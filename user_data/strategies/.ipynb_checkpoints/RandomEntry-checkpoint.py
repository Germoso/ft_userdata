# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these imports ---
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from pandas import DataFrame
from typing import Dict, Optional, Union, Tuple

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
    AnnotationType,
)

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
from technical import qtpylib


class RandomEntry(IStrategy):
    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 3

    # Optimal timeframe for the strategy.
    timeframe = "5m"

    # Can this strategy go short?
    can_short: bool = True

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    minimal_roi = {
        "0": 0.05
    }

    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    stoploss = -0.10
    
    # Trailing stoploss
    trailing_stop = False
    # trailing_only_offset_is_reached = False
    # trailing_stop_positive = 0.01
    # trailing_stop_positive_offset = 0.0  # Disabled / not configured

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True

    # These values can be overridden in the config.
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 1

    # Strategy parameters
    # Define parameters for random entry - probability of generating a long signal (0.5 = 50%)
    long_probability = DecimalParameter(0.1, 0.9, default=0.5, space="buy", optimize=True)


    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Generate a random number between 0 and 1 for each candle
        dataframe['random_value'] = np.random.random(size=len(dataframe))
        
        # If random value is less than long_probability, generate long signal
        # Otherwise generate short signal
        dataframe['random_long'] = dataframe['random_value'] < self.long_probability.value
        dataframe['random_short'] = ~dataframe['random_long']
        
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Inicializar columnas de entrada
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0
        
        # Entrar en long cuando random_long es True
        dataframe.loc[dataframe['random_long'], 'enter_long'] = 1
        
        # Entrar en short cuando random_short es True
        dataframe.loc[dataframe['random_short'], 'enter_short'] = 1
        
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe