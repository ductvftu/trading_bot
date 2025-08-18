from __future__ import annotations

from datetime import datetime
from typing import Any, Dict

import numpy as np
from pandas import DataFrame
import talib.abstract as ta

from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import informative


class StructureBreakStrategy(IStrategy):
    """
    Break-of-Structure (BOS) strategy:
    - Base timeframe: 3m
    - Signals generated on 3m; HTF confirmations via 5m and 15m
    - HTF gating: 5m and 15m EMA trend alignment with separation
    - Trend-strength: 8/21/50 EMA stack with 3-bar momentum
    - Vol/Vol gating: ATR% >= min_atr_pct and volume surge/trend with price momentum
    - Entries on BOS over confirmed swing levels with optional Retest/FVG filters
    - Exits on counter BOS; optional early exit on minor counter BOS
    """

    timeframe = "3m"
    can_short = True
    process_only_new_candles = True

    # Strategy configuration (effective values)
    STRATEGY_CONFIG: Dict[str, Any] = {
        "pivot_window": 5,
        "lookback": 160,
        "require_close_break": True,
        "require_htf_confirmation": True,
        "fast_ema": 9,
        "slow_ema": 21,
        "min_atr_pct": 0.0016,  # 0.16%
        "minor_pivot_window": 3,
        "minor_lookback": 60,
        "require_retest_hold": True,
        "retest_lookback_bars": 3,
        "retest_tol_bps": 15.0,
        "use_fvg_filter": True,
        "fvg_tolerance_bps": 15.0,
        "bos_buffer_bps": 6.0,
        "pre_break_bars": 2,
        "early_exit_on_minor_break": False,
        "exit_on_retest_failure": True,
        # New quality filters
        "rsi_period": 14,
        "adx_period": 14,
        "min_adx": 18.0,
        "rsi_long_min": 48.0,
        "rsi_long_max": 72.0,
        "rsi_short_min": 28.0,
        "rsi_short_max": 52.0,
        "min_break_atr_mult": 0.6,
        "min_close_pos_in_bar": 0.6,  # for long; use (1 - value) for short
        "max_break_extension_bps": 20.0,
    }

    # Backtest configuration (documented for reference)
    BACKTEST_CONFIG: Dict[str, Any] = {
        "fee_bps": 7.0,
        "leverage": 4.0,
        "initial_equity": 1000.0,
        "use_atr_stops": True,
        "use_swing_stops": False,
        "atr_period": 14,
        "stop_atr": 1.2,
        "target_atr": 1.0,
        "max_stop_atr": 1.6,
        "min_stop_atr": 0.5,
        "swing_pivot_window": 5,
        "swing_lookback": 200,
        "stop_buffer_bps": 3.0,
        "target_buffer_bps": 50.0,
        "signal_exit_consecutive": 6,
        "break_even_atr_trigger": 0.3,
        "cooldown_bars_after_sl": 0,
        "sl_tp_eval_timeframe": "3min",
        "exit_on_opposite_signal": False,
        "max_hold_minutes": 45,
    }

    # Risk handled via custom stoploss and signal exits - disable ROI exits
    minimal_roi: Dict[str, float] = {"0": 1000.0}
    stoploss = -0.99
    use_custom_stoploss = True
    use_exit_signal = True
    exit_profit_only = False

    htf_1 = "5m"
    htf_2 = "15m"

    startup_candle_count: int = 600

    @staticmethod
    def _bps_to_ratio(bps: float) -> float:
        return float(bps) / 10000.0

    @staticmethod
    def _safe_fill(df: DataFrame, cols: list[str]) -> DataFrame:
        for c in cols:
            if c not in df.columns:
                df[c] = 0
        return df

    @staticmethod
    def _pivot_extrema(series: DataFrame, window: int) -> DataFrame:
        # Past-only pivots: t-1 is a pivot if it was max/min over the last N bars
        max_mask = series.shift(1) == series.shift(1).rolling(window).max()
        min_mask = series.shift(1) == series.shift(1).rolling(window).min()
        return max_mask.astype(int), min_mask.astype(int)

    # HTF informative
    @informative(htf_1)
    def populate_indicators_5m(self, dataframe: DataFrame, metadata: Dict) -> DataFrame:
        cfg = self.STRATEGY_CONFIG
        fast, slow = int(cfg["fast_ema"]), int(cfg["slow_ema"]) 
        dataframe["ema_fast"] = ta.EMA(dataframe, timeperiod=fast)
        dataframe["ema_slow"] = ta.EMA(dataframe, timeperiod=slow)
        sep = (dataframe["ema_fast"] - dataframe["ema_slow"]) / dataframe["ema_slow"].replace(0, np.nan)
        dataframe["trend_bull"] = ((dataframe["ema_fast"] > dataframe["ema_slow"]) & (sep >= 0.002)).astype(int)
        dataframe["trend_bear"] = ((dataframe["ema_fast"] < dataframe["ema_slow"]) & (sep <= -0.002)).astype(int)
        return dataframe

    @informative(htf_2)
    def populate_indicators_15m(self, dataframe: DataFrame, metadata: Dict) -> DataFrame:
        cfg = self.STRATEGY_CONFIG
        fast, slow = int(cfg["fast_ema"]), int(cfg["slow_ema"]) 
        dataframe["ema_fast"] = ta.EMA(dataframe, timeperiod=fast)
        dataframe["ema_slow"] = ta.EMA(dataframe, timeperiod=slow)
        sep = (dataframe["ema_fast"] - dataframe["ema_slow"]) / dataframe["ema_slow"].replace(0, np.nan)
        dataframe["trend_bull"] = ((dataframe["ema_fast"] > dataframe["ema_slow"]) & (sep >= 0.002)).astype(int)
        dataframe["trend_bear"] = ((dataframe["ema_fast"] < dataframe["ema_slow"]) & (sep <= -0.002)).astype(int)
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: Dict) -> DataFrame:
        cfg = self.STRATEGY_CONFIG
        fast, slow = int(cfg["fast_ema"]), int(cfg["slow_ema"]) 

        # Core EMAs and stack (3m base)
        dataframe["ema8"] = ta.EMA(dataframe, timeperiod=8)
        dataframe["ema21"] = ta.EMA(dataframe, timeperiod=21)
        dataframe["ema50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["ema_fast"] = ta.EMA(dataframe, timeperiod=fast)
        dataframe["ema_slow"] = ta.EMA(dataframe, timeperiod=slow)

        # ATR and Volume
        atr_period = int(self.BACKTEST_CONFIG["atr_period"])
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=atr_period)
        dataframe["atr_pct"] = dataframe["atr"] / dataframe["close"].replace(0, np.nan)
        dataframe["vol_sma20"] = dataframe["volume"].rolling(20).mean()
        dataframe["vol_sma50"] = dataframe["volume"].rolling(50).mean()

        # RSI / ADX
        rsi_period = int(cfg.get("rsi_period", 14))
        adx_period = int(cfg.get("adx_period", 14))
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=rsi_period)
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=adx_period)

        # 3-bar momentum
        dataframe["mom3_up"] = ((dataframe["close"] > dataframe["close"].shift(1)) & (dataframe["close"].shift(1) > dataframe["close"].shift(2))).astype(int)
        dataframe["mom3_dn"] = ((dataframe["close"] < dataframe["close"].shift(1)) & (dataframe["close"].shift(1) < dataframe["close"].shift(2))).astype(int)

        # Trend-strength stack
        stack_bull = (dataframe["close"] > dataframe[["ema8", "ema21", "ema50"]].max(axis=1))
        stack_bear = (dataframe["close"] < dataframe[["ema8", "ema21", "ema50"]].min(axis=1))
        dataframe["stack_bull"] = stack_bull.astype(int)
        dataframe["stack_bear"] = stack_bear.astype(int)

        # Volatility and Volume gating
        dataframe["atr_ok"] = (dataframe["atr_pct"] >= float(cfg["min_atr_pct"]))
        vol_surge = dataframe["volume"] > 1.2 * dataframe["vol_sma20"]
        vol_trend_up = (dataframe["vol_sma20"] > dataframe["vol_sma20"].shift(1)) & (dataframe["mom3_up"] > 0)
        vol_trend_dn = (dataframe["vol_sma20"] > dataframe["vol_sma20"].shift(1)) & (dataframe["mom3_dn"] > 0)
        dataframe["vol_ok_bull"] = (vol_surge | vol_trend_up).astype(int)
        dataframe["vol_ok_bear"] = (vol_surge | vol_trend_dn).astype(int)

        # Major swings from pivots within lookback (no lookahead)
        pivot_w = int(cfg["pivot_window"]) 
        ph, pl = self._pivot_extrema(dataframe["high"], pivot_w)
        pivot_highs = dataframe["high"].shift(1).where(ph == 1)
        pivot_lows = dataframe["low"].shift(1).where(pl == 1)
        dataframe["swing_high"] = pivot_highs.ffill()
        dataframe["swing_low"] = pivot_lows.ffill()

        # Minor swings for optional exits
        min_pivot_w = int(cfg["minor_pivot_window"]) 
        mph, mpl = self._pivot_extrema(dataframe["high"], min_pivot_w)
        minor_highs = dataframe["high"].shift(1).where(mph == 1)
        minor_lows = dataframe["low"].shift(1).where(mpl == 1)
        dataframe["minor_swing_high"] = minor_highs.ffill()
        dataframe["minor_swing_low"] = minor_lows.ffill()

        # BOS definition (close vs intrabar)
        bos_buf = self._bps_to_ratio(cfg.get("bos_buffer_bps", 0.0))
        pre_n = int(cfg.get("pre_break_bars", 0))
        up_level = dataframe["swing_high"] * (1.0 + bos_buf)
        dn_level = dataframe["swing_low"] * (1.0 - bos_buf)
        if bool(cfg["require_close_break"]):
            raw_bos_long = dataframe["close"] > up_level
            raw_bos_short = dataframe["close"] < dn_level
        else:
            raw_bos_long = dataframe["high"] > up_level
            raw_bos_short = dataframe["low"] < dn_level
        if pre_n > 0:
            prev_max_high = dataframe["high"].shift(1).rolling(pre_n).max()
            prev_min_low = dataframe["low"].shift(1).rolling(pre_n).min()
            pre_below = (prev_max_high < up_level.shift(1))
            pre_above = (prev_min_low > dn_level.shift(1))
            bos_long = (raw_bos_long & pre_below)
            bos_short = (raw_bos_short & pre_above)
        else:
            bos_long = raw_bos_long
            bos_short = raw_bos_short
        dataframe["bos_long"] = bos_long.astype(int)
        dataframe["bos_short"] = bos_short.astype(int)

        # Break candle quality
        candle_range = (dataframe["high"] - dataframe["low"]).abs()
        min_break_atr = float(cfg.get("min_break_atr_mult", 0.0)) * dataframe["atr"]
        close_pos = (dataframe["close"] - dataframe["low"]) / candle_range.replace(0, np.nan)
        min_close_pos = float(cfg.get("min_close_pos_in_bar", 0.0))
        # Edge detection for break
        bos_long_b = bos_long.astype("boolean")
        bos_short_b = bos_short.astype("boolean")
        break_long_evt = bos_long_b & (~bos_long_b.shift(1).fillna(False))
        break_short_evt = bos_short_b & (~bos_short_b.shift(1).fillna(False))
        # Extension control
        max_ext = self._bps_to_ratio(cfg.get("max_break_extension_bps", 0.0))
        ext_long_ok = ((dataframe["close"] - dataframe["swing_high"]) / dataframe["close"].replace(0, np.nan)) <= max_ext
        ext_short_ok = ((dataframe["swing_low"] - dataframe["close"]) / dataframe["close"].replace(0, np.nan)) <= max_ext

        # Optional: FVG filter (coarse 3-candle)
        if bool(cfg["use_fvg_filter"]):
            bull_gap = dataframe["low"] - dataframe["high"].shift(2)
            bear_gap = dataframe["low"].shift(2) - dataframe["high"]
            fvg_tol = self._bps_to_ratio(cfg["fvg_tolerance_bps"]) * dataframe["close"]
            dataframe["fvg_bull"] = (bull_gap > fvg_tol).astype(int)
            dataframe["fvg_bear"] = (bear_gap > fvg_tol).astype(int)
        else:
            dataframe["fvg_bull"] = 1
            dataframe["fvg_bear"] = 1

        # Retest-and-hold AFTER break within window (no lookahead)
        if bool(cfg["require_retest_hold"]):
            ret_win = int(cfg["retest_lookback_bars"]) 
            tol = self._bps_to_ratio(cfg["retest_tol_bps"]) * dataframe["close"]
            bos_long_b = bos_long.astype("boolean")
            bos_short_b = bos_short.astype("boolean")
            break_long_evt2 = bos_long_b & (~bos_long_b.shift(1).fillna(False))
            break_short_evt2 = bos_short_b & (~bos_short_b.shift(1).fillna(False))
            recent_long_break = break_long_evt2.shift(1).rolling(ret_win).max().fillna(0) > 0
            recent_short_break = break_short_evt2.shift(1).rolling(ret_win).max().fillna(0) > 0
            ret_long = (
                recent_long_break
                & (dataframe["low"] <= (dataframe["swing_high"] + tol))
                & (dataframe["close"] > dataframe["swing_high"]) 
            )
            ret_short = (
                recent_short_break
                & (dataframe["high"] >= (dataframe["swing_low"] - tol))
                & (dataframe["close"] < dataframe["swing_low"]) 
            )
            dataframe["retest_ok_long"] = ret_long.astype(int)
            dataframe["retest_ok_short"] = ret_short.astype(int)
        else:
            dataframe["retest_ok_long"] = 1
            dataframe["retest_ok_short"] = 1

        # HTF confirmation flags exist after merge
        for col in ("trend_bull_5m", "trend_bull_15m", "trend_bear_5m", "trend_bear_15m"):
            if col not in dataframe.columns:
                dataframe[col] = 0
        dataframe[["trend_bull_5m", "trend_bull_15m", "trend_bear_5m", "trend_bear_15m"]] = \
            dataframe[["trend_bull_5m", "trend_bull_15m", "trend_bear_5m", "trend_bear_15m"]].ffill()

        # Final signal gates on 3m
        rsi = dataframe["rsi"]
        adx = dataframe["adx"]
        adx_ok = adx >= float(cfg.get("min_adx", 0.0))
        rsi_long_ok = (rsi >= float(cfg.get("rsi_long_min", 0))) & (rsi <= float(cfg.get("rsi_long_max", 1000)))
        rsi_short_ok = (rsi <= float(cfg.get("rsi_short_max", 1000))) & (rsi >= float(cfg.get("rsi_short_min", 0)))

        break_quality_long = (
            break_long_evt
            & (candle_range >= min_break_atr)
            & (close_pos >= float(cfg.get("min_close_pos_in_bar", 0.0)))
            & ext_long_ok
        )
        break_quality_short = (
            break_short_evt
            & (candle_range >= min_break_atr)
            & ((1.0 - close_pos) >= float(cfg.get("min_close_pos_in_bar", 0.0)))
            & ext_short_ok
        )

        core_long = (
            (dataframe["bos_long"] > 0)
            & (dataframe["atr_ok"] > 0)
            & (dataframe["vol_ok_bull"] > 0)
            & (dataframe["stack_bull"] > 0)
            & (dataframe["mom3_up"] > 0)
            & adx_ok
            & rsi_long_ok
            & break_quality_long
        )
        core_short = (
            (dataframe["bos_short"] > 0)
            & (dataframe["atr_ok"] > 0)
            & (dataframe["vol_ok_bear"] > 0)
            & (dataframe["stack_bear"] > 0)
            & (dataframe["mom3_dn"] > 0)
            & adx_ok
            & rsi_short_ok
            & break_quality_short
        )
        long_gate = core_long & (dataframe["retest_ok_long"] > 0) & (dataframe["fvg_bull"] > 0)
        short_gate = core_short & (dataframe["retest_ok_short"] > 0) & (dataframe["fvg_bear"] > 0)
        if bool(self.STRATEGY_CONFIG.get("require_htf_confirmation", True)):
            either_long_ok = ((dataframe["trend_bull_5m"] > 0) | (dataframe["trend_bull_15m"] > 0))
            either_short_ok = ((dataframe["trend_bear_5m"] > 0) | (dataframe["trend_bear_15m"] > 0))
            long_gate = long_gate & either_long_ok
            short_gate = short_gate & either_short_ok

        long_bool = long_gate.astype("boolean")
        short_bool = short_gate.astype("boolean")
        prev_long = long_bool.shift(1).fillna(False)
        prev_short = short_bool.shift(1).fillna(False)
        dataframe["sig_long"] = (long_bool & (~prev_long)).astype("int8")
        dataframe["sig_short"] = (short_bool & (~prev_short)).astype("int8")

        # Exits: Counter major BOS
        exit_long_major = (dataframe["bos_short"] > 0)
        exit_short_major = (dataframe["bos_long"] > 0)
        exit_long = exit_long_major
        exit_short = exit_short_major

        exit_long_b = exit_long.astype("boolean")
        exit_short_b = exit_short.astype("boolean")
        prev_exit_long = exit_long_b.shift(1).fillna(False)
        prev_exit_short = exit_short_b.shift(1).fillna(False)
        dataframe["sig_exit_long"] = (exit_long_b & (~prev_exit_long)).astype("int8")
        dataframe["sig_exit_short"] = (exit_short_b & (~prev_exit_short)).astype("int8")

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: Dict) -> DataFrame:
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0
        dataframe.loc[dataframe["sig_long"] > 0, "enter_long"] = 1
        dataframe.loc[dataframe["sig_short"] > 0, "enter_short"] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: Dict) -> DataFrame:
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0
        dataframe.loc[dataframe["sig_exit_long"] > 0, "exit_long"] = 1
        dataframe.loc[dataframe["sig_exit_short"] > 0, "exit_short"] = 1
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
        return float(min(self.BACKTEST_CONFIG["leverage"], max_leverage))

    def custom_stoploss(
        self,
        pair: str,
        trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        after_fill: bool,
        **kwargs,
    ) -> float | None:
        # Prefer swing-based stops when enabled, else fall back to ATR-based; cap with min/max ATR multiples
        try:
            df = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            last = df.loc[:current_time].iloc[-1]
        except Exception:
            last = None

        buffer = self._bps_to_ratio(self.BACKTEST_CONFIG.get("stop_buffer_bps", 0.0))

        if last is not None and bool(self.BACKTEST_CONFIG.get("use_swing_stops", False)):
            swing_low = last.get("swing_low", np.nan)
            swing_high = last.get("swing_high", np.nan)
            try:
                atr_pct_last = float(last.get("atr_pct", np.nan))
            except Exception:
                atr_pct_last = np.nan
            max_stop_atr = float(self.BACKTEST_CONFIG.get("max_stop_atr", 2.5))
            min_stop_atr = float(self.BACKTEST_CONFIG.get("min_stop_atr", 0.6))

            if not np.isnan(swing_low) and not np.isnan(swing_high):
                if getattr(trade, "is_short", False):
                    stop_price = float(swing_high) * (1.0 + buffer)
                    dist = max((stop_price - current_rate) / max(current_rate, 1e-9), 0.0)
                    if not np.isnan(atr_pct_last):
                        dist = min(dist, max_stop_atr * atr_pct_last)
                        dist = max(dist, min_stop_atr * atr_pct_last)
                    return dist
                else:
                    stop_price = float(swing_low) * (1.0 - buffer)
                    dist = max((current_rate - stop_price) / max(current_rate, 1e-9), 0.0)
                    if not np.isnan(atr_pct_last):
                        dist = min(dist, max_stop_atr * atr_pct_last)
                        dist = max(dist, min_stop_atr * atr_pct_last)
                    return dist

        # ATR-based stop evaluated from 3m ATR%
        try:
            atr_pct = float(df.loc[:current_time]["atr_pct"].iloc[-1]) if last is not None else np.nan
        except Exception:
            atr_pct = np.nan

        if np.isnan(atr_pct):
            return None

        stop_atr = float(self.BACKTEST_CONFIG.get("stop_atr", 2.5)) * atr_pct
        max_stop_atr = float(self.BACKTEST_CONFIG.get("max_stop_atr", 2.5))
        min_stop_atr = float(self.BACKTEST_CONFIG.get("min_stop_atr", 0.6))
        stop_atr = min(max(stop_atr, min_stop_atr * atr_pct), max_stop_atr * atr_pct)

        be_trigger = float(self.BACKTEST_CONFIG.get("break_even_atr_trigger", 0.5)) * atr_pct
        if current_profit is not None and current_profit >= be_trigger:
            return max(buffer, 0.0)

        return max(stop_atr + buffer, 0.0)

    def custom_exit(
        self,
        pair: str,
        trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> str | bool | None:
        # ATR-based take profit and time-based cut for stuck losers
        try:
            df = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            atr_pct = float(df.loc[:current_time]["atr_pct"].iloc[-1])
        except Exception:
            atr_pct = np.nan

        try:
            max_hold = int(self.BACKTEST_CONFIG.get("max_hold_minutes", 0))
            if max_hold and trade and trade.open_date_utc is not None:
                held = (current_time - trade.open_date_utc).total_seconds() / 60.0
                if held >= max_hold and (current_profit is not None) and current_profit < 0:
                    return "time_cut"
        except Exception:
            pass

        if not np.isnan(atr_pct):
            target_atr = float(self.BACKTEST_CONFIG.get("target_atr", 2.0)) * atr_pct
            buf = self._bps_to_ratio(self.BACKTEST_CONFIG.get("target_buffer_bps", 0.0))
            if (current_profit is not None) and (current_profit >= max(target_atr - buf, 0.0)):
                return "tp_atr"

        return None

