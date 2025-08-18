from __future__ import annotations

from datetime import datetime
from typing import Dict

from pandas import DataFrame
import talib.abstract as ta

from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import informative


class SMCUsdtFuturesStrategy(IStrategy):
    """
    5m Trend-Pullback strategy for Binance USDT perpetual futures.

    Design:
    - Trade with the 5m trend (EMA alignment) with 15m HTF confirmation.
    - Enter on shallow pullbacks to EMA20 with volatility gating via ATR.
    - Exits via Chandelier Stop; fixed TP/SL apply as configured.

    Notes:
    - Signals: enter_long/enter_short (futures mode). Ensure config has trading_mode=futures.
    - Parameters are intentionally simple for robust evaluation on 5m.
    """

    timeframe = "5m"
    can_short = True

    # Core parameters
    ema_fast = 20
    ema_slow = 100
    atr_length = 14
    atr_mult = 1.5  # for chandelier stop

    # Higher timeframe filter (HTF)
    htf_timeframe = "15m"
    htf_ema_fast = 50
    htf_ema_slow = 200

    # Basic risk settings (adjust to taste and leverage)
    minimal_roi = {"0": 0.03}
    stoploss = -0.03

    # Need enough data for HTF EMA(200) on 15m -> 200 * (15/5) = 600 5m candles
    startup_candle_count: int = 1000

    use_exit_signal = True
    exit_profit_only = False
    use_custom_stoploss = False

    @informative(htf_timeframe)
    def populate_indicators_15m(self, dataframe: DataFrame, metadata: Dict) -> DataFrame:
        # HTF trend filter via EMA alignment
        dataframe["ema_fast"] = ta.EMA(dataframe, timeperiod=self.htf_ema_fast)
        dataframe["ema_slow"] = ta.EMA(dataframe, timeperiod=self.htf_ema_slow)
        dataframe["htf_bull"] = (dataframe["ema_fast"] > dataframe["ema_slow"]).astype(int)
        dataframe["htf_bear"] = (dataframe["ema_fast"] < dataframe["ema_slow"]).astype(int)
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: Dict) -> DataFrame:
        # Trend and momentum
        dataframe["ema_fast"] = ta.EMA(dataframe, timeperiod=self.ema_fast)
        dataframe["ema_slow"] = ta.EMA(dataframe, timeperiod=self.ema_slow)

        # Volatility (ATR)
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=self.atr_length)

        # Chandelier stop levels
        highest_close = dataframe["close"].rolling(self.atr_length).max()
        lowest_close = dataframe["close"].rolling(self.atr_length).min()
        dataframe["chand_long"] = highest_close - self.atr_mult * dataframe["atr"]
        dataframe["chand_short"] = lowest_close + self.atr_mult * dataframe["atr"]

        # Pullback flags
        # price near ema_fast within 0.25*ATR
        dataframe["near_ema"] = (abs(dataframe["close"] - dataframe["ema_fast"]) <= 0.25 * dataframe["atr"]).astype(int)

        # Ensure HTF columns exist after merge
        for col in ("htf_bull_15m", "htf_bear_15m"):
            if col not in dataframe.columns:
                dataframe[col] = 0
        dataframe[["htf_bull_15m", "htf_bear_15m"]] = dataframe[["htf_bull_15m", "htf_bear_15m"]].fillna(0)

        return dataframe

    def _compute_fvg(self, df: DataFrame) -> DataFrame:
        """Add Fair Value Gap columns on LTF using 3-candle definition (no lookahead)."""
        # Bullish FVG: current low > high of 2 candles ago
        bull_gap = df["low"] - df["high"].shift(2)
        df["bullish_fvg"] = (bull_gap > (df["close"] * self.fvg_min_ratio)).astype(int)

        # Bearish FVG: current high < low of 2 candles ago
        bear_gap = df["low"].shift(2) - df["high"]
        df["bearish_fvg"] = (bear_gap > (df["close"] * self.fvg_min_ratio)).astype(int)

        return df

    def _compute_liquidity_sweeps(self, df: DataFrame) -> DataFrame:
        """Detect liquidity sweeps using prior N-bar highs/lows and current wick/close."""
        prev_high = df["high"].shift(1).rolling(self.bos_lookback).max()
        prev_low = df["low"].shift(1).rolling(self.bos_lookback).min()

        # Sweep high: take prior high but close back below that level
        df["sweep_high"] = ((df["high"] > prev_high) & (df["close"] < prev_high)).astype(int)

        # Sweep low: take prior low but close back above that level
        df["sweep_low"] = ((df["low"] < prev_low) & (df["close"] > prev_low)).astype(int)

        return df

    def populate_entry_trend(self, dataframe: DataFrame, metadata: Dict) -> DataFrame:
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0

        bull_trend = dataframe["ema_fast"] > dataframe["ema_slow"]
        bear_trend = dataframe["ema_fast"] < dataframe["ema_slow"]
        htf_bull = dataframe["htf_bull_15m"] > 0
        htf_bear = dataframe["htf_bear_15m"] > 0

        # Enter on pullback to ema_fast in trend direction
        dataframe.loc[bull_trend & (dataframe["near_ema"] > 0) & htf_bull, "enter_long"] = 1
        dataframe.loc[bear_trend & (dataframe["near_ema"] > 0) & htf_bear, "enter_short"] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: Dict) -> DataFrame:
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0

        # Exit if price closes beyond chandelier levels
        exit_long = (dataframe["close"] < dataframe["chand_long"]) 
        exit_short = (dataframe["close"] > dataframe["chand_short"]) 

        dataframe.loc[exit_long, "exit_long"] = 1
        dataframe.loc[exit_short, "exit_short"] = 1

        return dataframe

    # Optional: control leverage (defaults to 1.0 if omitted)
    def leverage(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> float:
        # Hard-fixed leverage 5x, capped by exchange max
        return float(min(5.0, max_leverage))

    # No custom stoploss; exits via ROI/SL and exit signals

