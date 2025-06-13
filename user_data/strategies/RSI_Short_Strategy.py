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

class RSIShortStrategy(IStrategy):
    """
    Estrategia de trading que se especializa en posiciones cortas basadas en el indicador RSI.
    
    Esta estrategia:
    - Opera exclusivamente con posiciones cortas
    - Entra al mercado cuando el RSI indica condiciones de sobrecompra (RSI > 70)
    - Utiliza una gestión dinámica de posiciones que permite promediar a la baja
    - Implementa un sistema de trailing stop para proteger beneficios
    - Ajusta posiciones basándose en el nivel de pérdida actual
    - Cierra operaciones cuando se alcanza el objetivo de beneficio configurado
    """

    INTERFACE_VERSION = 3

    can_short: bool = True

    minimal_roi = {
        "0": 0.1 
    }

    position_adjustment_enable = True

    stoploss = -1

    timeframe = "5m"

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

    # Hyperopt parameters for entry
    short_rsi = IntParameter(low=60, high=90, default=70, space="buy", optimize=True, load=True)
    short_rsi_decreasing = IntParameter(low=1, high=10, default=2, space="buy", optimize=True, load=True)    

    # Hyperopt parameters for position adjustment
    max_dca_adjustments = IntParameter(low=1, high=10, default=5, space="buy", optimize=True, load=True)
    stoploss_threshold = DecimalParameter(low=-1.0, high=-0.1, default=-1.0, space="sell", optimize=True, load=True)
    dca_threshold_1 = DecimalParameter(low=-10, high=010, default=-1, space="buy", optimize=True, load=True)
    dca_threshold_2 = DecimalParameter(low=-10, high=10, default=-2, space="buy", optimize=True, load=True)
    dca_threshold_3 = DecimalParameter(low=-10, high=10, default=-4, space="buy", optimize=True, load=True)
    dca_multiplier_1 = DecimalParameter(low=1.0, high=3.0, default=1.0, space="buy", optimize=True, load=True)
    dca_multiplier_2 = DecimalParameter(low=1.0, high=3.0, default=2.0, space="buy", optimize=True, load=True)
    dca_multiplier_3 = DecimalParameter(low=1.0, high=3.0, default=2.0, space="buy", optimize=True, load=True)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["rsi"] = ta.RSI(dataframe)
        dataframe['rsi_prev_1'] = dataframe['rsi'].shift(1)
        dataframe['rsi_prev_2'] = dataframe['rsi'].shift(2)
        dataframe['rsi_decreasing'] = (dataframe['rsi'] < dataframe['rsi_prev_1']).astype('int')
        dataframe['rsi_increasing'] = (dataframe['rsi'] > dataframe['rsi_prev_1']).astype('int')
        
        dataframe['ema_slow'] = ta.EMA(dataframe, timeperiod=200)
        dataframe['ema_fast'] = ta.EMA(dataframe, timeperiod=5)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        
        conditions.append(dataframe['rsi'] > self.short_rsi.value)
        conditions.append(dataframe['rsi_decreasing'] > 0)
        conditions.append(dataframe['volume'] > 0)

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe

    def adjust_trade_position(self, trade: Trade, current_time: datetime, current_rate: float,
                             current_profit: float, min_stake: float, max_stake: float,
                             **kwargs) -> Optional[float]:
        # Close position if profit is below stoploss threshold
        if current_profit <= self.stoploss_threshold.value:
            return trade.stake_amount * -1
        
        # Limit the number of DCA adjustments
        if trade.nr_of_successful_entries >= self.max_dca_adjustments.value:
            return None

        # DCA strategy based on current profit thresholds
        if current_profit <= self.dca_threshold_3.value:
            return trade.stake_amount * self.dca_multiplier_3.value
        elif current_profit <= self.dca_threshold_2.value:
            return trade.stake_amount * self.dca_multiplier_2.value
        elif current_profit <= self.dca_threshold_1.value:
            return trade.stake_amount * self.dca_multiplier_1.value
        
        return None