from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import talib.abstract as ta

from freqtrade.strategy import IStrategy, Trade
from freqtrade.strategy.strategy_helper import stoploss_from_absolute


class EMATrend1mStrategy(IStrategy):
    """
    1-minute EMA trend-following strategy implementing:
    - EMAs: 5, 10, 30, 60
    - Entry long: After >=7 consecutive closes below EMA30, price reverses and prints 3rd/4th
      consecutive close above EMA30 → enter at close.
    - Entry short: Mirror of long.
    - Exit: When EMA10 crosses/touches EMA30 (directional).
    - Stop-loss: At the most recent trough/peak (swing) of the prior opposite run that broke
      across EMA30 (converted via custom_stoploss to an absolute stop price).
    """

    INTERFACE_VERSION = 3

    timeframe = "5m"
    can_short: bool = True

    # ROI is effectively governed by exits; keep a permissive minimal ROI
    minimal_roi = {"0": 10.0}

    # Hard stoploss (5% with 10x leverage ≈ 0.5% price move)
    stoploss = -0.5

    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = True

    # We reference prior runs; give generous startup to form EMAs and runs
    startup_candle_count: int = 200

    # Core entry guards (for SMA crossover)
    slope_len: int = 5  # candles to measure SMA20 slope
    min_slope: float = 0.0045  # 0.15% over slope_len candles
    min_ema_sep: float = 0.0005  # EMA10 vs EMA30 separation
    # Trading session filter (UTC hours). Default: Asian session 23:00–08:59 UTC
    use_asian_session_only: bool = True
    asian_session_hours_utc: list[int] = [23, 0, 1, 2, 3, 4, 5, 6, 7, 8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]
    # Volume zone filter
    use_volume_zone: bool = True
    vol_zone_len: int = 30  # rolling window in candles
    vol_zone_mult: float = 1.4  # rolling vol must be >= X * day avg so far
    # Longer prior-run requirement
    min_prior_run_len: int = 7

    plot_config = {
        "main_plot": {
            "ema_5": {"color": "#03A9F4"},
            "ema_10": {"color": "#009688"},
            "ema_30": {"color": "#FF9800"},
            "ema_60": {"color": "#9E9E9E"},
            "reversal_trough": {"color": "#8BC34A"},
            "reversal_peak": {"color": "#E91E63"},
            "donchian_high": {"color": "#4CAF50"},
            "donchian_low": {"color": "#F44336"},
        },
        "subplots": {},
    }

    # Disable custom stoploss (pure crossover exits)
    use_custom_stoploss = False

    @staticmethod
    def _consecutive_true_counts(condition: Series) -> Series:
        """Return consecutive True counts, resetting to 0 on False."""
        # group runs by changes in condition
        run_id = (condition != condition.shift(1)).cumsum()
        # count within each run
        run_pos = condition.groupby(run_id).cumcount() + 1
        return run_pos.where(condition, 0)

    @staticmethod
    def _last_opposite_run_length_propagated(above_now: Series, consec_below: Series) -> Series:
        """
        For candles where above_now is True, propagate the length of the previous below-run
        (the run that ended at the switch). NaN elsewhere.
        """
        switch_up = above_now & ~above_now.shift(1, fill_value=False)
        at_switch_len = consec_below.shift(1).where(switch_up)
        # forward-fill only within above runs
        result = at_switch_len.copy()
        result = result.where(above_now)
        result = result.ffill()
        result = result.where(above_now)
        return result

    @staticmethod
    def _last_opposite_run_length_propagated_down(below_now: Series, consec_above: Series) -> Series:
        switch_down = below_now & ~below_now.shift(1, fill_value=False)
        at_switch_len = consec_above.shift(1).where(switch_down)
        result = at_switch_len.copy()
        result = result.where(below_now)
        result = result.ffill()
        result = result.where(below_now)
        return result

    @staticmethod
    def _run_id(condition: Series) -> Series:
        return (condition != condition.shift(1)).cumsum()

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Core SMAs for crossover
        dataframe["sma_20"] = dataframe["close"].rolling(window=20, min_periods=20).mean()
        dataframe["sma_200"] = dataframe["close"].rolling(window=200, min_periods=200).mean()
        # Slope of SMA20 over slope_len candles (relative)
        dataframe["sma20_slope"] = (
            (dataframe["sma_20"] - dataframe["sma_20"].shift(self.slope_len))
            / dataframe["sma_20"].shift(self.slope_len)
        )

        # Keep EMAs only for legacy fields used elsewhere (if any)
        dataframe["ema_5"] = dataframe["close"].ewm(span=5, adjust=False).mean()
        dataframe["ema_10"] = dataframe["close"].ewm(span=10, adjust=False).mean()
        dataframe["ema_30"] = dataframe["close"].ewm(span=30, adjust=False).mean()
        dataframe["ema_60"] = dataframe["close"].ewm(span=60, adjust=False).mean()

        # No informative merges or donchian channels – pure 1m logic

        # Asian session filter (UTC)
        if self.use_asian_session_only:
            # Ensure "date" column exists and is datetime
            if "date" in dataframe.columns:
                dataframe["hour"] = dataframe["date"].dt.hour
                dataframe["is_asian_session"] = dataframe["hour"].isin(self.asian_session_hours_utc)
            else:
                dataframe["is_asian_session"] = True

        # High-volume zone vs. current day's average so far (no lookahead)
        if self.use_volume_zone:
            if "date" in dataframe.columns and "volume" in dataframe.columns:
                dataframe["day"] = dataframe["date"].dt.floor("D")
                grp = dataframe.groupby("day", group_keys=False)
                cum_vol = grp["volume"].cumsum()
                cum_cnt = grp.cumcount() + 1
                dataframe["day_avg_vol_sofar"] = cum_vol / cum_cnt
                dataframe["vol_ma"] = dataframe["volume"].rolling(window=self.vol_zone_len, min_periods=1).mean()
                dataframe["is_high_vol_zone"] = dataframe["vol_ma"] >= (self.vol_zone_mult * dataframe["day_avg_vol_sofar"])
            else:
                dataframe["is_high_vol_zone"] = True

        # Crossover helpers
        above = dataframe["sma_20"] > dataframe["sma_200"]
        dataframe["cross_up"] = above & (~above.shift(1, fill_value=False))
        dataframe["cross_down"] = (~above) & (above.shift(1, fill_value=False))

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Long entry: SMA20 crosses above SMA200 with sufficient positive slope
        long_condition = (
            (dataframe["cross_up"]) & (dataframe["sma20_slope"] > self.min_slope)
        )

        dataframe.loc[long_condition & (dataframe["volume"] > 0), "enter_long"] = 1

        # Short entry: SMA20 crosses below SMA200 with sufficient negative slope
        short_condition = (
            (dataframe["cross_down"]) & (dataframe["sma20_slope"] < -self.min_slope)
        )

        dataframe.loc[short_condition & (dataframe["volume"] > 0), "enter_short"] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Exit long on next bearish crossover
        exit_long_condition = dataframe["cross_down"]

        dataframe.loc[(exit_long_condition) & (dataframe["volume"] > 0), "exit_long"] = 1

        # Exit short on next bullish crossover
        exit_short_condition = dataframe["cross_up"]

        dataframe.loc[(exit_short_condition) & (dataframe["volume"] > 0), "exit_short"] = 1

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
        """Use 10x leverage (capped by exchange max)."""
        return float(min(10.0, max_leverage))

    @property
    def protections(self):
        """
        Apply a cooldown after each trade closes to avoid immediate re-entry during chop.
        Uses candles for timeframe-agnostic tuning (e.g., 50 candles at 5m ≈ 250 minutes).
        """
        return [
            {
                "method": "CooldownPeriod",
                "stop_duration_candles": 50,
            }
        ]

    def custom_stoploss(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        after_fill: bool,
        **kwargs: Any,
    ) -> float | None:
        """
        Use the prior-run swing (trough/peak) across EMA30 as the absolute stop price.
        While in-trade, if new swings form that increase protection (higher trough for long,
        lower peak for short), the stop will only ratchet in the favorable direction.
        """
        if self.dp is None:
            return None

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or dataframe.empty:
            return None

        # Restrict to candles since the trade opened (inclusive)
        df = dataframe.loc[dataframe["date"] >= trade.open_date_utc]
        if df.empty:
            return None

        if trade.is_short:
            # Candidate stop is the minimum of prior-run reversal peaks since entry (lower is tighter for shorts)
            # We want the maximum protection (lowest absolute price above current), so take the minimum of
            # valid reversal peaks and clip to be >= current_rate to avoid invalid stop
            base_stop = df["reversal_peak"].dropna().min()
            atr = df["atr_14"].iloc[-1] if "atr_14" in df.columns else np.nan
            stop_price = base_stop + (self.atr_mult_sl * atr if pd.notna(atr) else 0.0) if pd.notna(base_stop) else np.nan
        else:
            # Long: use the maximum of reversal troughs since entry (ratchet up only)
            base_stop = df["reversal_trough"].dropna().max()
            atr = df["atr_14"].iloc[-1] if "atr_14" in df.columns else np.nan
            stop_price = base_stop - (self.atr_mult_sl * atr if pd.notna(atr) else 0.0) if pd.notna(base_stop) else np.nan

        if stop_price is None or np.isnan(stop_price):
            return None

        # Convert absolute stop price to relative distance required by framework
        return stoploss_from_absolute(
            stop_rate=float(stop_price),
            current_rate=float(current_rate),
            is_short=trade.is_short,
            leverage=trade.leverage,
        )


