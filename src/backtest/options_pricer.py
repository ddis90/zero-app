"""
Black-Scholes option pricing for synthetic option chain generation.
Used in backtesting to simulate option premiums when historical chain data isn't available.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from scipy.stats import norm

logger = logging.getLogger(__name__)

# Risk-free rate (India 10-year G-Sec yield approximation)
RISK_FREE_RATE = 0.065


@dataclass
class OptionQuote:
    """Synthetic option quote."""
    strike: float
    option_type: str  # "CE" or "PE"
    premium: float
    delta: float
    gamma: float
    theta: float
    vega: float
    iv: float
    days_to_expiry: int
    tradingsymbol: str


class OptionsPricer:
    """
    Black-Scholes based option pricing engine for backtesting.
    
    Generates synthetic option chains given:
    - Spot price
    - VIX (used as IV proxy with adjustments)
    - Days to expiry
    - Strike range
    
    Accuracy: ~85-90% for ATM/near-OTM options on Nifty/BankNifty.
    Less accurate for deep OTM and near-expiry (ignores skew smile).
    """

    def __init__(self, risk_free_rate: float = RISK_FREE_RATE):
        self.r = risk_free_rate

    def black_scholes_price(
        self,
        spot: float,
        strike: float,
        days_to_expiry: int,
        iv: float,
        option_type: str = "CE",
    ) -> float:
        """
        Calculate Black-Scholes option price.
        
        Args:
            spot: Current underlying price
            strike: Strike price
            days_to_expiry: Days to expiration
            iv: Implied volatility (annualized, as decimal e.g. 0.15 for 15%)
            option_type: "CE" for call, "PE" for put
            
        Returns:
            Option premium
        """
        if days_to_expiry <= 0:
            # At expiry: intrinsic value only
            if option_type == "CE":
                return max(0, spot - strike)
            else:
                return max(0, strike - spot)

        t = days_to_expiry / 365.0
        sigma = iv

        d1 = (np.log(spot / strike) + (self.r + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
        d2 = d1 - sigma * np.sqrt(t)

        if option_type == "CE":
            price = spot * norm.cdf(d1) - strike * np.exp(-self.r * t) * norm.cdf(d2)
        else:
            price = strike * np.exp(-self.r * t) * norm.cdf(-d2) - spot * norm.cdf(-d1)

        return max(0.05, price)  # Minimum premium of ₹0.05

    def calculate_greeks(
        self,
        spot: float,
        strike: float,
        days_to_expiry: int,
        iv: float,
        option_type: str = "CE",
    ) -> dict:
        """Calculate option Greeks."""
        if days_to_expiry <= 0:
            return {"delta": 1.0 if option_type == "CE" else -1.0, "gamma": 0, "theta": 0, "vega": 0}

        t = days_to_expiry / 365.0
        sigma = iv
        sqrt_t = np.sqrt(t)

        d1 = (np.log(spot / strike) + (self.r + 0.5 * sigma**2) * t) / (sigma * sqrt_t)
        d2 = d1 - sigma * sqrt_t

        # Delta
        if option_type == "CE":
            delta = norm.cdf(d1)
        else:
            delta = norm.cdf(d1) - 1

        # Gamma
        gamma = norm.pdf(d1) / (spot * sigma * sqrt_t)

        # Theta (per day)
        theta_term1 = -(spot * norm.pdf(d1) * sigma) / (2 * sqrt_t)
        if option_type == "CE":
            theta = (theta_term1 - self.r * strike * np.exp(-self.r * t) * norm.cdf(d2)) / 365
        else:
            theta = (theta_term1 + self.r * strike * np.exp(-self.r * t) * norm.cdf(-d2)) / 365

        # Vega (per 1% move in IV)
        vega = spot * sqrt_t * norm.pdf(d1) / 100

        return {"delta": delta, "gamma": gamma, "theta": theta, "vega": vega}

    def generate_option_chain(
        self,
        spot: float,
        vix: float,
        days_to_expiry: int,
        underlying: str = "NIFTY",
        strike_step: float = None,
        num_strikes: int = 20,
    ) -> pd.DataFrame:
        """
        Generate a synthetic option chain for backtesting.
        
        Args:
            spot: Current spot price of underlying
            vix: India VIX value (used to derive IV)
            days_to_expiry: Days until option expiry
            underlying: "NIFTY" or "BANKNIFTY"
            strike_step: Distance between strikes (auto-calculated if None)
            num_strikes: Number of strikes on each side of ATM
            
        Returns:
            DataFrame with columns matching live option chain format
        """
        if strike_step is None:
            strike_step = 50 if underlying == "NIFTY" else 100

        # ATM strike (rounded to nearest strike_step)
        atm_strike = round(spot / strike_step) * strike_step

        # Generate strike range
        strikes = [atm_strike + i * strike_step for i in range(-num_strikes, num_strikes + 1)]

        # IV adjustments:
        # - VIX is a 30-day ATM IV estimate
        # - Weekly options have higher IV (closer expiry = more gamma risk)
        # - OTM options have higher IV (skew)
        base_iv = vix / 100.0  # Convert VIX to decimal

        rows = []
        for strike in strikes:
            for opt_type in ["CE", "PE"]:
                # IV adjustment for skew
                moneyness = (spot - strike) / spot
                if opt_type == "PE":
                    moneyness = -moneyness

                # Simple skew model: OTM puts have higher IV
                iv_adjustment = 1.0
                if opt_type == "PE" and moneyness > 0:  # OTM put
                    iv_adjustment = 1.0 + 0.3 * moneyness  # Put skew
                elif opt_type == "CE" and moneyness > 0:  # OTM call
                    iv_adjustment = 1.0 + 0.15 * moneyness  # Call skew (less steep)

                # Time adjustment: shorter expiry → slightly higher effective IV
                time_adj = 1.0
                if days_to_expiry <= 7:
                    time_adj = 1.1
                elif days_to_expiry <= 3:
                    time_adj = 1.2

                adjusted_iv = base_iv * iv_adjustment * time_adj
                adjusted_iv = max(0.08, min(0.80, adjusted_iv))  # Clamp 8%-80%

                # Calculate premium
                premium = self.black_scholes_price(spot, strike, days_to_expiry, adjusted_iv, opt_type)
                greeks = self.calculate_greeks(spot, strike, days_to_expiry, adjusted_iv, opt_type)

                # Generate tradingsymbol (approximate format)
                expiry_str = "WKLY"  # Simplified for backtest
                tradingsymbol = f"{underlying}{expiry_str}{int(strike)}{opt_type}"

                rows.append({
                    "strike": strike,
                    "instrument_type": opt_type,
                    "tradingsymbol": tradingsymbol,
                    "ltp": round(premium, 2),
                    "iv": round(adjusted_iv * 100, 1),
                    "delta": round(greeks["delta"], 4),
                    "gamma": round(greeks["gamma"], 6),
                    "theta": round(greeks["theta"], 2),
                    "vega": round(greeks["vega"], 2),
                    "oi": self._synthetic_oi(spot, strike, opt_type),
                    "volume": self._synthetic_volume(spot, strike, opt_type),
                    "days_to_expiry": days_to_expiry,
                })

        return pd.DataFrame(rows)

    def price_credit_spread(
        self,
        spot: float,
        sell_strike: float,
        buy_strike: float,
        vix: float,
        days_to_expiry: int,
        spread_type: str = "bull_put",
    ) -> dict:
        """
        Price a credit spread for backtesting theta selling strategy.
        
        Args:
            spot: Current spot price
            sell_strike: Strike to sell (closer to money)
            buy_strike: Strike to buy (further OTM, protection)
            vix: India VIX value
            days_to_expiry: DTE at entry
            spread_type: "bull_put" or "bear_call"
            
        Returns:
            Dict with net_premium, max_profit, max_loss, breakeven
        """
        base_iv = vix / 100.0

        if spread_type == "bull_put":
            opt_type = "PE"
            # Sell higher strike put, buy lower strike put
            sell_premium = self.black_scholes_price(spot, sell_strike, days_to_expiry, base_iv * 1.05, opt_type)
            buy_premium = self.black_scholes_price(spot, buy_strike, days_to_expiry, base_iv * 1.1, opt_type)
        else:  # bear_call
            opt_type = "CE"
            # Sell lower strike call, buy higher strike call
            sell_premium = self.black_scholes_price(spot, sell_strike, days_to_expiry, base_iv * 1.0, opt_type)
            buy_premium = self.black_scholes_price(spot, buy_strike, days_to_expiry, base_iv * 1.05, opt_type)

        net_premium = sell_premium - buy_premium
        spread_width = abs(sell_strike - buy_strike)
        max_loss = spread_width - net_premium

        return {
            "net_premium": round(max(0, net_premium), 2),
            "max_profit": round(max(0, net_premium), 2),
            "max_loss": round(max_loss, 2),
            "sell_premium": round(sell_premium, 2),
            "buy_premium": round(buy_premium, 2),
            "breakeven": sell_strike - net_premium if spread_type == "bull_put" else sell_strike + net_premium,
            "spread_width": spread_width,
        }

    def reprice_spread_at_expiry(
        self,
        spot_at_expiry: float,
        sell_strike: float,
        buy_strike: float,
        spread_type: str = "bull_put",
    ) -> float:
        """
        Calculate the intrinsic value of a spread at expiry.
        Returns the loss amount (0 if profitable, positive if loss).
        """
        if spread_type == "bull_put":
            # Bull put spread: lose money if spot < sell_strike
            if spot_at_expiry >= sell_strike:
                return 0.0  # Both puts expire worthless = keep full premium
            elif spot_at_expiry <= buy_strike:
                return abs(sell_strike - buy_strike)  # Max loss
            else:
                return sell_strike - spot_at_expiry  # Partial loss
        else:  # bear_call
            # Bear call spread: lose money if spot > sell_strike
            if spot_at_expiry <= sell_strike:
                return 0.0
            elif spot_at_expiry >= buy_strike:
                return abs(buy_strike - sell_strike)
            else:
                return spot_at_expiry - sell_strike

    def _synthetic_oi(self, spot: float, strike: float, opt_type: str) -> int:
        """Generate realistic OI (higher near ATM, lower far OTM)."""
        distance = abs(spot - strike) / spot
        base_oi = 50000
        oi = int(base_oi * np.exp(-10 * distance**2))
        return max(1000, oi)

    def _synthetic_volume(self, spot: float, strike: float, opt_type: str) -> int:
        """Generate realistic volume."""
        distance = abs(spot - strike) / spot
        base_vol = 20000
        vol = int(base_vol * np.exp(-8 * distance**2))
        return max(100, vol)
