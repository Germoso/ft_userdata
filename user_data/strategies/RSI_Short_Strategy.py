# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
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

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
from technical import qtpylib


class RSIShortStrategy(IStrategy):
    """
    Estrategia personalizada que solo hace shorts cuando el RSI está en sobreventa
    y cierra con un 1% de profit.
    
    Esta estrategia:
    - Solo realiza operaciones en corto (shorts)
    - Entra cuando el RSI está en sobreventa (RSI > 70)
    - Cierra la posición cuando se alcanza un 1% de beneficio
    """

    # Strategy interface version
    INTERFACE_VERSION = 3

    # Esta estrategia solo hace shorts
    can_short: bool = True

    # Minimal ROI - configurado para cerrar con 1% de beneficio
    minimal_roi = {
        "0": 0.1 
    }

    position_adjustment_enable = True

    # Stoploss
    stoploss = -1

    # Trailing stoploss
    # trailing_stop = True
    # trailing_stop_positive = 0.005
    # trailing_stop_positive_offset = 0.01
    # trailing_only_offset_is_reached = False

    # Timeframe
    timeframe = "5m"

    # Procesar solo nuevas velas
    process_only_new_candles = True

    # Configuración de señales de salida
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Parámetros optimizables
    short_rsi = IntParameter(low=60, high=90, default=60, space="sell", optimize=True, load=True)
    
    # Número de velas necesarias antes de producir señales válidas
    startup_candle_count: int = 30

    # Configuración de tipos de órdene
    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }

    # Configuración de tiempo en vigor de las órdenes
    order_time_in_force = {"entry": "GTC", "exit": "GTC"}

    # Configuración de gráficos
    plot_config = {
        "main_plot": {},
        "subplots": {
            "RSI": {
                "rsi": {"color": "red"},
            },
        },
    }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Calcula los indicadores necesarios para la estrategia.
        """
        # RSI
        dataframe["rsi"] = ta.RSI(dataframe)
        
        # RSI valores previos
        dataframe['rsi_prev_1'] = dataframe['rsi'].shift(1)
        dataframe['rsi_prev_2'] = dataframe['rsi'].shift(2)
        
        # Dirección del RSI
        dataframe['rsi_decreasing'] = (dataframe['rsi'] < dataframe['rsi_prev_1']).astype('int')
        dataframe['rsi_increasing'] = (dataframe['rsi'] > dataframe['rsi_prev_1']).astype('int')

        # EMA
        dataframe['ema_slow'] = ta.EMA(dataframe, timeperiod=2000)
        dataframe['ema_fast'] = ta.EMA(dataframe, timeperiod=500)
        
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Define las condiciones para entrar en posiciones cortas.
        Solo entramos en corto cuando el RSI está en sobreventa (RSI > 70).
        """


        #Entrar siempre a short
        dataframe['enter_short'] = 1

        conditions = []
        
        # Condición principal: RSI en sobreventa
        conditions.append(dataframe['rsi'] > self.short_rsi.value)
        
        # RSI debe estar disminuyendo
        conditions.append(dataframe['rsi_decreasing'] > 0)

        # EMA 
        conditions.append(dataframe['ema_slow'] > dataframe['ema_fast'])
        
        # Verificar que el volumen no sea 0
        conditions.append(dataframe['volume'] > 0)

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        No utilizamos señales de salida específicas ya que saldremos 
        cuando alcancemos el 1% de beneficio definido en minimal_roi.
        """
        return dataframe

    def adjust_trade_position(self, trade: Trade, current_time: datetime, current_rate: float,
                             current_profit: float, min_stake: float, max_stake: float,
                             **kwargs) -> Optional[float]:
        """
        Ajusta la posición de una operación existente.
        Permite añadir a la posición si se cumplen ciertas condiciones basadas en:
        - Beneficio actual
        - Indicadores técnicos (RSI, EMA)
        - Tiempo transcurrido desde la entrada
        - Número de ajustes previos
        Implementa un stoploss que cierra toda la posición cuando se alcanza una pérdida determinada.
        """
        # Obtener el dataframe con los indicadores
        dataframe = kwargs.get('dataframe', None)
        
        # Definir el umbral de stoploss para cerrar toda la posición
        stoploss_threshold = -1  # Cerrar toda la posición si la pérdida es del 100% o mayor
        
        # Si alcanzamos o superamos el umbral de stoploss, cerrar toda la posición
        if current_profit <= stoploss_threshold:
            # Retornar -1 para cerrar toda la posición
            return trade.stake_amount * -1
        
        # Definir el número máximo de ajustes de posición
        max_adjustments = 5
        
        # Si ya hemos ajustado la posición el número máximo de veces, no hacer nada
        if trade.nr_of_successful_entries >= max_adjustments:
            return None
            
        # Verificar si tenemos acceso al dataframe

        # Si no tenemos acceso al dataframe, usar la estrategia básica basada solo en beneficio
        if current_profit <= -0.30:  # Si estamos en -30% o peor
            # Añadir una posición grande para promediar a la baja
            return trade.stake_amount * 2
        elif current_profit <= -0.2:  # Si estamos entre -15% y -30%
            # Añadir una posición moderada
            return trade.stake_amount * 2
        elif current_profit <= -0.1:  # Si estamos entre -5% y -15%
            # Añadir una posición pequeña
            return trade.stake_amount
        
        # Si no cumplimos ninguna condición, no ajustar la posición
        return None

    def confirm_trade_exit(self, pair: str, trade: 'Trade', order_type: str, amount: float,
                          rate: float, time_in_force: str, exit_reason: str, **kwargs) -> bool:
        """
        Confirma la salida de una operación.
        Solo permitimos salir si hemos alcanzado al menos un 1% de beneficio.
        """
        # Calcular beneficio actual
        current_profit = trade.calc_profit_ratio(rate)
        
        # Solo salir si tenemos al menos un 1% de beneficio
        if current_profit >= 0.1:
            return True
            
        # En caso contrario, no salir
        return False
