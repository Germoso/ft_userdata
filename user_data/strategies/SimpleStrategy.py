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

    startup_candle_count: int = 30

    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }

    order_time_in_force = {"entry": "GTC", "exit": "GTC"}

    plot_config = {
        "main_plot": {},
        "subplots": {
            "RSI": {
                "rsi": {"color": "red"},
            },
        },
    }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['ema_long'] = ta.EMA(dataframe, timeperiod=200)
        dataframe['ema_short'] = ta.EMA(dataframe, timeperiod=100)
        dataframe['ema_filter'] = ta.EMA(dataframe, timeperiod=1000) # Esto se puede usar para sacar stop loss dinámico si el preico cierra por encima de esa ema

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        conditions.append(dataframe['ema_long']> dataframe['ema_short'])
        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        conditions.append(dataframe['ema_long']< dataframe['ema_short'])
        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'exit_short'] = 1
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