# --- Do not remove these imports ---
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from pandas import DataFrame
from typing import Optional, Union
from functools import reduce

from freqtrade.strategy import (
    IStrategy,
    Trade,
    Order,
    PairLocks,
    informative,  # @informative decorator
    # Hyperopt Parameters
    BooleanParameter,
    CategoricalParameter,
    DecimalParameter,
    IntParameter,
    RealParameter,
    # timeframe helpers
    timeframe_to_minutes,
    timeframe_to_next_date,
    timeframe_to_prev_date,
    # Strategy helper functions
    merge_informative_pair,
    stoploss_from_absolute,
    stoploss_from_open,
)

import talib.abstract as ta
from technical import qtpylib

class SimpleStrategy(IStrategy):
    """
    Estrategia de trading que se especializa en posiciones cortas basadas en el cruce de 2 medias moviles exponenciales.
    """

    INTERFACE_VERSION = 3

    can_short: bool = True

    minimal_roi = {
        "0": 1
    }

    position_adjustment_enable = True

    stoploss = -1

    # Procesar solo nuevas velas
    process_only_new_candles = True

    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    startup_candle_count: int = 100

    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }

    order_time_in_force = {"entry": "GTC", "exit": "GTC"}

    plot_config = {
        "main_plot": {
            "ema_5": {"color": "#00ff00"},    # Lime
            "ema_10": {"color": "#0000ff"},   # Blue
            "ema_50": {"color": "#ffa500"},   # Orange
            "ema_100": {"color": "#ff0000"},  # Red
        },
        "subplots": {
            "RSI": {
                "rsi": {"color": "red"},
            },
        },
    }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # EMAs requested for "EMA Strategy PRO" conversion
        dataframe['ema_5'] = ta.EMA(dataframe, timeperiod=5)
        dataframe['ema_10'] = ta.EMA(dataframe, timeperiod=10)
        dataframe['ema_50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema_100'] = ta.EMA(dataframe, timeperiod=100)
        
        # We also need a way to check for crossovers easily
        # (Though we can use qtpylib directly in populate_entry_trend)
        
        # Keep old names if necessary for some logic, 
        # but the request implies a full replacement of logic.
        dataframe['ema_long'] = dataframe['ema_100']
        dataframe['ema_short'] = dataframe['ema_50']

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Trend Conditions
        bull_trend = dataframe['ema_50'] > dataframe['ema_100']
        bear_trend = dataframe['ema_50'] < dataframe['ema_100']

        # Long: (Bull Trend + crossover(EMA 5, EMA 10)) OR (crossover(EMA 50, EMA 100))
        dataframe.loc[
            (
                (bull_trend & qtpylib.crossed_above(dataframe['ema_5'], dataframe['ema_10'])) |
                qtpylib.crossed_above(dataframe['ema_50'], dataframe['ema_100'])
            ),
            'enter_long'] = 1

        # Short: (Bear Trend + crossunder(EMA 5, EMA 10)) OR (crossunder(EMA 50, EMA 100))
        dataframe.loc[
            (
                (bear_trend & qtpylib.crossed_below(dataframe['ema_5'], dataframe['ema_10'])) |
                qtpylib.crossed_below(dataframe['ema_50'], dataframe['ema_100'])
            ),
            'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Exit signals can be added here if needed. 
        # For now, we rely on ROI and Stoploss.
        return dataframe

    def adjust_trade_position(self, trade: Trade, current_time: datetime, current_rate: float,
                             current_profit: float, min_stake: float, max_stake: float,
                             **kwargs) -> Optional[float]:
        # # Close position if profit is below stoploss threshold
        # if current_profit <= self.stoploss_threshold.value:
        #     return trade.stake_amount * -1

        # # Limit the number of DCA adjustments
        # if trade.nr_of_successful_entries >= self.max_dca_adjustments.value:
        #     return None

        # # DCA strategy based on current profit thresholds
        # if current_profit <= self.dca_threshold_3.value:
        #     return trade.stake_amount * self.dca_multiplier_3.value
        # elif current_profit <= self.dca_threshold_2.value:
        #     return trade.stake_amount * self.dca_multiplier_2.value
        # elif current_profit <= self.dca_threshold_1.value:
        #     return trade.stake_amount * self.dca_multiplier_1.value

        return None