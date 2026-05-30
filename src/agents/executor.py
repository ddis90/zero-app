"""
Order Executor - Places and manages orders via Kite Connect.
All orders go through Risk Manager validation before execution.
"""

import logging
from datetime import datetime
from typing import Optional

from kiteconnect import KiteConnect

from src.strategies.base import Signal, SignalType
from src.agents.risk_manager import RiskManager, Position

logger = logging.getLogger(__name__)


class OrderExecutor:
    """
    Executes trading orders via Kite Connect API.
    
    Features:
    - All orders validated by RiskManager before placement
    - Supports market, limit, and SL orders
    - GTT (Good Till Triggered) for persistent stop-losses
    - Spread order execution for options
    - Paper trading mode for testing
    """

    def __init__(self, kite: KiteConnect, risk_manager: RiskManager, paper_trade: bool = False):
        self.kite = kite
        self.risk_manager = risk_manager
        self.paper_trade = paper_trade
        self.order_history: list[dict] = []

    def execute_signal(self, signal: Signal) -> Optional[dict]:
        """
        Execute a trading signal after risk validation.
        
        Returns order details dict or None if blocked.
        """
        # Calculate position size if not set
        if signal.quantity == 0:
            signal.quantity = self.risk_manager.calculate_position_size(
                entry_price=signal.entry_price,
                stop_loss=signal.stop_loss,
                strategy=signal.strategy_name,
            )

        if signal.quantity == 0:
            logger.warning(f"Position size is 0 for {signal.symbol}. Skipping.")
            return None

        # Validate with risk manager
        allowed, reason = self.risk_manager.validate_trade(
            symbol=signal.symbol,
            quantity=signal.quantity,
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            strategy=signal.strategy_name,
        )

        if not allowed:
            logger.warning(f"Order BLOCKED by risk manager: {reason}")
            return None

        # Execute based on signal type
        if signal.signal_type == SignalType.BUY:
            return self._place_buy_order(signal)
        elif signal.signal_type == SignalType.SELL:
            if signal.metadata and "spread_type" in signal.metadata:
                return self._place_spread_order(signal)
            return self._place_sell_order(signal)
        elif signal.signal_type == SignalType.EXIT:
            return self._place_exit_order(signal)

        return None

    def _place_buy_order(self, signal: Signal) -> dict:
        """Place a buy order (equity delivery or intraday)."""
        order_params = {
            "tradingsymbol": signal.symbol,
            "exchange": signal.exchange,
            "transaction_type": "BUY",
            "quantity": signal.quantity,
            "order_type": "LIMIT",
            "price": signal.entry_price,
            "product": "CNC",  # Delivery (Cash and Carry)
            "validity": "DAY",
        }

        order_id = self._place_order(order_params)

        if order_id:
            # Register position with risk manager
            position = Position(
                symbol=signal.symbol,
                exchange=signal.exchange,
                quantity=signal.quantity,
                entry_price=signal.entry_price,
                entry_time=datetime.now(),
                stop_loss=signal.stop_loss,
                target=signal.target,
                strategy=signal.strategy_name,
                position_type="long",
            )
            self.risk_manager.add_position(position)

            # Place GTT stop-loss order (persistent, broker-side)
            self._place_gtt_stop_loss(signal)

        return {"order_id": order_id, "signal": signal, "status": "placed"}

    def _place_sell_order(self, signal: Signal) -> dict:
        """Place a sell order."""
        order_params = {
            "tradingsymbol": signal.symbol,
            "exchange": signal.exchange,
            "transaction_type": "SELL",
            "quantity": signal.quantity,
            "order_type": "LIMIT",
            "price": signal.entry_price,
            "product": "CNC",
            "validity": "DAY",
        }

        order_id = self._place_order(order_params)
        return {"order_id": order_id, "signal": signal, "status": "placed"}

    def _place_spread_order(self, signal: Signal) -> dict:
        """
        Place a credit spread order (two legs).
        Leg 1: Sell closer strike (collect premium)
        Leg 2: Buy further strike (hedge)
        """
        metadata = signal.metadata
        results = {}

        # Leg 1: Sell option (premium collection)
        sell_params = {
            "tradingsymbol": metadata["sell_symbol"],
            "exchange": signal.exchange,
            "transaction_type": "SELL",
            "quantity": signal.quantity,
            "order_type": "LIMIT",
            "price": 0,  # Will be set to LTP
            "product": "NRML",  # Normal (F&O margin)
            "validity": "DAY",
        }

        # Leg 2: Buy option (hedge)
        buy_params = {
            "tradingsymbol": metadata["buy_symbol"],
            "exchange": signal.exchange,
            "transaction_type": "BUY",
            "quantity": signal.quantity,
            "order_type": "LIMIT",
            "price": 0,
            "product": "NRML",
            "validity": "DAY",
        }

        # Execute both legs
        sell_order_id = self._place_order(sell_params)
        buy_order_id = self._place_order(buy_params)

        if sell_order_id and buy_order_id:
            position = Position(
                symbol=signal.symbol,
                exchange=signal.exchange,
                quantity=signal.quantity,
                entry_price=metadata["premium_collected"],
                entry_time=datetime.now(),
                stop_loss=signal.stop_loss,
                target=signal.target,
                strategy=signal.strategy_name,
                position_type="short",
            )
            self.risk_manager.add_position(position)

        return {
            "sell_order_id": sell_order_id,
            "buy_order_id": buy_order_id,
            "signal": signal,
            "status": "spread_placed",
        }

    def _place_exit_order(self, signal: Signal) -> dict:
        """Place an exit order and close position in risk manager."""
        # Determine transaction type based on position
        positions = self.risk_manager.open_positions
        position = next((p for p in positions if p.symbol == signal.symbol), None)

        if not position:
            logger.warning(f"No position to exit for {signal.symbol}")
            return {"status": "no_position"}

        transaction_type = "SELL" if position.position_type == "long" else "BUY"
        product = "CNC" if signal.exchange == "NSE" else "NRML"

        order_params = {
            "tradingsymbol": signal.symbol,
            "exchange": signal.exchange,
            "transaction_type": transaction_type,
            "quantity": signal.quantity,
            "order_type": "MARKET",  # Exit at market for certainty
            "product": product,
            "validity": "DAY",
        }

        order_id = self._place_order(order_params)

        if order_id:
            self.risk_manager.close_position(
                symbol=signal.symbol,
                exit_price=signal.entry_price,
                exit_reason=signal.reason,
            )

        return {"order_id": order_id, "signal": signal, "status": "exit_placed"}

    def _place_gtt_stop_loss(self, signal: Signal):
        """
        Place GTT (Good Till Triggered) order for stop-loss.
        Persists on broker side even if our system goes down.
        """
        if self.paper_trade:
            logger.info(f"[PAPER] GTT SL for {signal.symbol} @ ₹{signal.stop_loss:.1f}")
            return

        try:
            trigger_values = [signal.stop_loss]
            orders = [{
                "transaction_type": "SELL",
                "quantity": signal.quantity,
                "order_type": "LIMIT",
                "product": "CNC",
                "price": signal.stop_loss * 0.99,  # Slight buffer below trigger
            }]

            gtt_id = self.kite.place_gtt(
                trigger_type="single",
                tradingsymbol=signal.symbol,
                exchange=signal.exchange,
                trigger_values=trigger_values,
                last_price=signal.entry_price,
                orders=orders,
            )
            logger.info(f"GTT stop-loss placed for {signal.symbol}: ID {gtt_id}")
        except Exception as e:
            logger.error(f"GTT placement failed for {signal.symbol}: {e}")

    def _place_order(self, params: dict) -> Optional[str]:
        """Place order via Kite or simulate in paper trade mode."""
        if self.paper_trade:
            order_id = f"PAPER_{datetime.now().strftime('%H%M%S')}_{params['tradingsymbol']}"
            logger.info(
                f"[PAPER TRADE] {params['transaction_type']} {params['quantity']} "
                f"{params['tradingsymbol']} @ ₹{params.get('price', 'MKT')}"
            )
            self.order_history.append({**params, "order_id": order_id, "time": datetime.now()})
            return order_id

        try:
            order_id = self.kite.place_order(
                variety="regular",
                **params,
            )
            logger.info(
                f"Order placed: {params['transaction_type']} {params['quantity']} "
                f"{params['tradingsymbol']} | ID: {order_id}"
            )
            self.order_history.append({**params, "order_id": order_id, "time": datetime.now()})
            return order_id
        except Exception as e:
            logger.error(f"Order placement failed: {e}")
            return None

    def get_order_status(self, order_id: str) -> dict:
        """Check status of a placed order."""
        if self.paper_trade:
            return {"status": "COMPLETE", "order_id": order_id}
        try:
            orders = self.kite.orders()
            for order in orders:
                if order["order_id"] == order_id:
                    return order
            return {"status": "NOT_FOUND"}
        except Exception as e:
            logger.error(f"Failed to fetch order status: {e}")
            return {"status": "ERROR", "error": str(e)}

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order."""
        if self.paper_trade:
            logger.info(f"[PAPER] Cancelled order {order_id}")
            return True
        try:
            self.kite.cancel_order(variety="regular", order_id=order_id)
            logger.info(f"Order cancelled: {order_id}")
            return True
        except Exception as e:
            logger.error(f"Order cancellation failed: {e}")
            return False
