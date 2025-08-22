from __future__ import annotations

from datetime import datetime
from typing import Dict, Any

import numpy as np
from pandas import DataFrame
import talib.abstract as ta

from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import informative


class VolumeRotorClockStrategy(IStrategy):
    """
    Volume-dominance strategy inspired by the TradingView indicator
    "Volume Rotor Clock [hapharmonic]".

    Core idea:
    - Per-bar buy/sell volume split using the price's relative close in the bar range:
      buy = volume * (close - low) / (high - low)
      sell = volume * (high - close) / (high - low)
      Fallback when (high == low): buy = sell = volume / 2
    - Compute rolling sums of buy/sell over a short lookback window.
    - Enter long on bullish flip (buy_sum > sell_sum) with minimum dominance percentage
      and optional volume-above-MA gating. Enter short on the bearish mirror.
    - Exit on opposite flip events.

    Notes:
    - Futures-compatible (can_short = True).
    - Keep exits signal-driven; ROI left permissive by default.
    """

    timeframe = "5m"
    can_short = False  # default long-only; can be toggled by trade_shorts below

    # ROI/SL and trailing – exits governed by ROI/SL/Trailing (no exit signals)
    minimal_roi: Dict[str, float] = {"0": 0.04}  # 4%
    stoploss = -0.02  # -2%
    trailing_stop = True
    trailing_stop_positive = 0.005  # 0.5%
    trailing_stop_positive_offset = 0.015  # 1.5% arming offset
    trailing_only_offset_is_reached = True

    process_only_new_candles = True
    use_exit_signal = False
    exit_profit_only = False
    ignore_roi_if_entry_signal = True
    use_custom_stoploss = True

    # Indicator parameters
    lookback_bars: int = 3  # match Pine default (i_lookbackBars)
    vol_ma_len: int = 20
    vol_ma_mult: float = 1.8  # stricter: require volume > vol_ma * mult
    require_vol_above_ma: bool = True
    min_buy_pct_dom_long: float = 55.0  # minimum dominance percentage to accept longs
    min_sell_pct_dom_short: float = 55.0  # minimum dominance percentage to accept shorts
    min_dom_gap_pct: float = 15.0  # require |buy_pct - sell_pct| >= gap

    # Volatility gating
    atr_length: int = 14
    min_atr_pct: float = 0.0020  # 0.20%

    # Trend filter
    ema_fast: int = 50
    ema_slow: int = 200
    min_trend_sep: float = 0.004  # 0.4%

    # Stability / candle quality
    consecutive_dom_bars: int = 3
    min_close_pos_long: float = 0.65  # close near high for longs
    min_close_pos_short: float = 0.65  # close near low for shorts (use 1-close_pos)
    min_candle_atr_mult: float = 0.6

    # Dynamic risk management
    be_trigger: float = 0.012  # go break-even once profit >= 1.2%
    be_min_lock: float = 0.0005  # keep at least +0.05% locked when at BE

    # Higher timeframe (HTF) alignment
    htf_timeframe: str = "15m"
    htf_fast: int = 50
    htf_slow: int = 200
    htf_min_sep: float = 0.003

    # Entry overextension guard (vs. EMA fast)
    max_entry_extension: float = 0.01  # 1% max distance to ema_fast

    # Allow enabling shorts explicitly
    trade_shorts: bool = False

    @informative(htf_timeframe)
    def populate_indicators_15m(self, dataframe: DataFrame, metadata: Dict) -> DataFrame:
        dataframe["ema_fast"] = ta.EMA(dataframe, timeperiod=int(self.htf_fast))
        dataframe["ema_slow"] = ta.EMA(dataframe, timeperiod=int(self.htf_slow))
        sep = (dataframe["ema_fast"] - dataframe["ema_slow"]) / dataframe["ema_slow"].replace(0, np.nan)
        dataframe["htf_bull"] = ((dataframe["ema_fast"] > dataframe["ema_slow"]) & (sep >= float(self.htf_min_sep))).astype("int8")
        dataframe["htf_bear"] = ((dataframe["ema_fast"] < dataframe["ema_slow"]) & (sep <= -float(self.htf_min_sep))).astype("int8")
        return dataframe

    # Need enough history for rolling calculations
    startup_candle_count: int = 200

    def populate_indicators(self, dataframe: DataFrame, metadata: Dict) -> DataFrame:
        # Guard: ensure necessary columns exist
        for col in ("open", "high", "low", "close", "volume"):
            if col not in dataframe.columns:
                dataframe[col] = np.nan

        # Denominator (high-low); handle empty range
        denom = (dataframe["high"] - dataframe["low"]).replace(0, np.nan)

        # Buy/Sell volume split per bar
        buy_vol = dataframe["volume"] * (dataframe["close"] - dataframe["low"]) / denom
        sell_vol = dataframe["volume"] * (dataframe["high"] - dataframe["close"]) / denom

        # Fallback when denom is NaN (i.e., high == low)
        fallback = dataframe["volume"] / 2.0
        buy_vol = buy_vol.fillna(fallback)
        sell_vol = sell_vol.fillna(fallback)

        lb = int(self.lookback_bars) + 1  # Pine sums current bar + N lookback bars
        dataframe["buy_sum"] = buy_vol.rolling(window=lb, min_periods=1).sum()
        dataframe["sell_sum"] = sell_vol.rolling(window=lb, min_periods=1).sum()
        dataframe["tot_sum"] = dataframe["buy_sum"] + dataframe["sell_sum"]

        # Dominance percentages
        tot_safe = dataframe["tot_sum"].replace(0, np.nan)
        dataframe["buy_pct"] = 100.0 * (dataframe["buy_sum"] / tot_safe)
        dataframe["sell_pct"] = 100.0 - dataframe["buy_pct"]

        # Volume MA gating (optional)
        dataframe["vol_ma"] = dataframe["volume"].rolling(self.vol_ma_len, min_periods=1).mean()
        dataframe["vol_above_ma"] = (
            dataframe["volume"] > (self.vol_ma_mult * dataframe["vol_ma"])
        ).astype("int8")

        # Volatility (ATR%)
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=int(self.atr_length))
        dataframe["atr_pct"] = (dataframe["atr"] / dataframe["close"].replace(0, np.nan)).astype(float)
        dataframe["atr_ok"] = (dataframe["atr_pct"] >= float(self.min_atr_pct)).astype("int8")

        # Trend EMAs & separation
        dataframe["ema_fast"] = ta.EMA(dataframe, timeperiod=int(self.ema_fast))
        dataframe["ema_slow"] = ta.EMA(dataframe, timeperiod=int(self.ema_slow))
        sep = (dataframe["ema_fast"] - dataframe["ema_slow"]) / dataframe["ema_slow"].replace(0, np.nan)
        dataframe["trend_bull"] = ((dataframe["ema_fast"] > dataframe["ema_slow"]) & (sep >= float(self.min_trend_sep))).astype("int8")
        dataframe["trend_bear"] = ((dataframe["ema_fast"] < dataframe["ema_slow"]) & (sep <= -float(self.min_trend_sep))).astype("int8")

        # Candle quality
        dataframe["candle_range"] = (dataframe["high"] - dataframe["low"]).abs()
        dataframe["close_pos"] = (dataframe["close"] - dataframe["low"]) / dataframe["candle_range"].replace(0, np.nan)
        dataframe["range_atr_ok"] = (dataframe["candle_range"] >= (float(self.min_candle_atr_mult) * dataframe["atr"]))
        dataframe["close_pos_long_ok"] = (dataframe["close_pos"] >= float(self.min_close_pos_long))
        dataframe["close_pos_short_ok"] = ((1.0 - dataframe["close_pos"]) >= float(self.min_close_pos_short))

        # Merge HTF flags if missing
        for col in ("htf_bull_15m", "htf_bear_15m"):
            if col not in dataframe.columns:
                dataframe[col] = 0
        dataframe[["htf_bull_15m", "htf_bear_15m"]] = dataframe[["htf_bull_15m", "htf_bear_15m"]].fillna(0)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: Dict) -> DataFrame:
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0

        # Core dominance conditions
        dom_gap = (dataframe["buy_pct"] - dataframe["sell_pct"]).abs()
        bull_dom_raw = (
            (dataframe["buy_sum"] > dataframe["sell_sum"]) &
            (dataframe["buy_pct"] >= float(self.min_buy_pct_dom_long)) &
            (dom_gap >= float(self.min_dom_gap_pct))
        )
        bear_dom_raw = (
            (dataframe["sell_sum"] > dataframe["buy_sum"]) &
            (dataframe["sell_pct"] >= float(self.min_sell_pct_dom_short)) &
            (dom_gap >= float(self.min_dom_gap_pct))
        )

        # Optional volume gating and ATR gating
        if self.require_vol_above_ma:
            bull_dom_raw = bull_dom_raw & (dataframe["vol_above_ma"] > 0)
            bear_dom_raw = bear_dom_raw & (dataframe["vol_above_ma"] > 0)
        bull_dom_raw = bull_dom_raw & (dataframe["atr_ok"] > 0)
        bear_dom_raw = bear_dom_raw & (dataframe["atr_ok"] > 0)

        # Trend, HTF and candle-quality gating
        bull_dom_raw = bull_dom_raw & (dataframe["trend_bull"] > 0) & (dataframe["htf_bull_15m"] > 0) & dataframe["range_atr_ok"] & dataframe["close_pos_long_ok"]
        bear_dom_raw = bear_dom_raw & (dataframe["trend_bear"] > 0) & (dataframe["htf_bear_15m"] > 0) & dataframe["range_atr_ok"] & dataframe["close_pos_short_ok"]

        # Stability: require consecutive dominance bars
        cons = int(self.consecutive_dom_bars)
        if cons > 1:
            bull_dom = bull_dom_raw & bull_dom_raw.shift(1)
            bear_dom = bear_dom_raw & bear_dom_raw.shift(1)
        else:
            bull_dom = bull_dom_raw
            bear_dom = bear_dom_raw

        # Overextension guard (avoid chasing beyond EMA fast)
        ext = (dataframe["close"] - dataframe["ema_fast"]).abs() / dataframe["close"].replace(0, np.nan)
        not_overext = ext <= float(self.max_entry_extension)

        # Flip events (edge detection)
        bull_dom = bull_dom.astype("boolean")
        bear_dom = bear_dom.astype("boolean")
        prev_bull = bull_dom.shift(1)
        prev_bear = bear_dom.shift(1)
        bull_evt = bull_dom & (~prev_bull.fillna(False))
        bear_evt = bear_dom & (~prev_bear.fillna(False))

        dataframe.loc[bull_evt & not_overext & (dataframe["volume"] > 0), "enter_long"] = 1
        if self.trade_shorts and self.can_short:
            dataframe.loc[bear_evt & not_overext & (dataframe["volume"] > 0), "enter_short"] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: Dict) -> DataFrame:
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0

        # Recompute dominance booleans using existing columns and gates
        dom_gap = (dataframe["buy_pct"] - dataframe["sell_pct"]).abs()
        bull_dom_raw = (
            (dataframe["buy_sum"] > dataframe["sell_sum"]) &
            (dataframe["buy_pct"] >= float(self.min_buy_pct_dom_long)) &
            (dom_gap >= float(self.min_dom_gap_pct)) &
            (dataframe["atr_ok"] > 0) &
            (dataframe["trend_bull"] > 0) & (dataframe["htf_bull_15m"] > 0) & dataframe["range_atr_ok"] & dataframe["close_pos_long_ok"]
        )
        bear_dom_raw = (
            (dataframe["sell_sum"] > dataframe["buy_sum"]) &
            (dataframe["sell_pct"] >= float(self.min_sell_pct_dom_short)) &
            (dom_gap >= float(self.min_dom_gap_pct)) &
            (dataframe["atr_ok"] > 0) &
            (dataframe["trend_bear"] > 0) & (dataframe["htf_bear_15m"] > 0) & dataframe["range_atr_ok"] & dataframe["close_pos_short_ok"]
        )
        cons = int(self.consecutive_dom_bars)
        if cons > 1:
            bull_dom = bull_dom_raw & bull_dom_raw.shift(1)
            bear_dom = bear_dom_raw & bear_dom_raw.shift(1)
        else:
            bull_dom = bull_dom_raw
            bear_dom = bear_dom_raw

        # Exit on opposite flip events
        bull_dom = bull_dom.astype("boolean")
        bear_dom = bear_dom.astype("boolean")
        prev_bull = bull_dom.shift(1)
        prev_bear = bear_dom.shift(1)
        bull_evt = bull_dom & (~prev_bull.fillna(False))
        bear_evt = bear_dom & (~prev_bear.fillna(False))

        dataframe.loc[bear_evt, "exit_long"] = 1
        dataframe.loc[bull_evt, "exit_short"] = 1

        return dataframe

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
        # Fixed leverage with exchange cap
        return float(min(2.0, max_leverage))

    @property
    def protections(self):
        # Cooldown to avoid immediate re-entries during chop
        return [
            {
                "method": "CooldownPeriod",
                "stop_duration_candles": 10,
            }
        ]

    def _dominance_raw_flags(self, dataframe: DataFrame) -> tuple[Any, Any]:
        """Compute raw dominance flags (pre consecutive-bar stability)."""
        dom_gap = (dataframe["buy_pct"] - dataframe["sell_pct"]).abs()
        bull_dom_raw = (
            (dataframe["buy_sum"] > dataframe["sell_sum"]) &
            (dataframe["buy_pct"] >= float(self.min_buy_pct_dom_long)) &
            (dom_gap >= float(self.min_dom_gap_pct)) &
            (dataframe["atr_ok"] > 0) &
            (dataframe["trend_bull"] > 0) & dataframe["range_atr_ok"] & dataframe["close_pos_long_ok"]
        )
        bear_dom_raw = (
            (dataframe["sell_sum"] > dataframe["buy_sum"]) &
            (dataframe["sell_pct"] >= float(self.min_sell_pct_dom_short)) &
            (dom_gap >= float(self.min_dom_gap_pct)) &
            (dataframe["atr_ok"] > 0) &
            (dataframe["trend_bear"] > 0) & dataframe["range_atr_ok"] & dataframe["close_pos_short_ok"]
        )
        return bull_dom_raw, bear_dom_raw

    def custom_stoploss(
        self,
        pair: str,
        trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        after_fill: bool,
        **kwargs: Any,
    ) -> float | None:
        """
        Dynamic stop:
        - Before BE: defer to framework static stoploss (return None).
        - After BE trigger: lock at max(be_min_lock, 0.5 * ATR%).
        """
        # If no profit info, defer to framework stoploss
        if current_profit is None:
            return None

        # Try to fetch ATR%
        atr_pct = None
        try:
            if self.dp:
                df = self.dp.get_analyzed_dataframe(pair, self.timeframe)
                atr_pct = float(df.loc[:current_time]["atr_pct"].iloc[-1])
        except Exception:
            atr_pct = None

        if current_profit >= float(self.be_trigger):
            lock = float(self.be_min_lock)
            if atr_pct is not None and np.isfinite(atr_pct):
                lock = max(lock, 0.5 * atr_pct)
            return float(max(lock, 0.0001))

        # Before BE: use configured static stoploss
        return None

    def custom_exit(
        self,
        pair: str,
        trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs: Any,
    ) -> str | bool | None:
        """
        Early exit on opposite dominance flip to avoid full SL.
        Only triggers if not yet at BE.
        """
        try:
            if not self.dp:
                return None
            df = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            last = df.loc[:current_time].iloc[-1]
        except Exception:
            return None

        # If already at/above BE trigger, let trailing/ROI handle
        if current_profit is not None and current_profit >= float(self.be_trigger):
            return None

        # Opposite dominance raw flip check
        try:
            # Compute raw flags on the last row context
            bull_raw, bear_raw = self._dominance_raw_flags(df)
            opp_bear = bool(bear_raw.loc[last.name])
            opp_bull = bool(bull_raw.loc[last.name])
        except Exception:
            return None

        if getattr(trade, "is_short", False):
            # For shorts, exit if bull dominance appears
            if opp_bull:
                return "opp_flip_early"
        else:
            # For longs, exit if bear dominance appears
            if opp_bear:
                return "opp_flip_early"

        return None


