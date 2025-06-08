# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these imports ---
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from pandas import DataFrame
from typing import Optional, Union, Dict, List
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


class ShortGridTradingBot(IStrategy):
    """
    Short Grid Trading Bot con niveles configurables por porcentaje.
    
    Esta estrategia:
    - Establece una cuadrícula (grid) de órdenes de venta en corto a intervalos porcentuales definidos
    - Permite configurar el porcentaje entre niveles de entrada y salida
    - Solo opera en posiciones cortas
    - Gestiona cada nivel de la grid de forma independiente
    - No utiliza indicadores técnicos, solo opera basado en niveles de precio
    """

    # Strategy interface version
    INTERFACE_VERSION = 3

    # Solo permitir operaciones en corto
    can_short: bool = True
    can_long: bool = False

    # Minimal ROI - configurado para no usar ROI automático
    minimal_roi = {
        "0": 100  # No usar ROI, la estrategia gestiona las salidas
    }

    # Habilitar ajuste de posición para añadir entradas
    position_adjustment_enable = True
    
    # Stoploss - No usar stoploss global, la estrategia gestiona el riesgo
    stoploss = -1.0

    # Timeframe
    timeframe = "5m"

    # Procesar solo nuevas velas
    process_only_new_candles = True

    # Configuración de señales de salida
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Parámetros optimizables para la grid
    # Porcentaje entre niveles de entrada en corto (cuando el precio sube)
    grid_short_entry_pct = DecimalParameter(0.5, 5.0, default=1.0, space="sell", optimize=True, load=True)
    
    # Porcentaje de beneficio objetivo para cada nivel
    profit_target_pct = DecimalParameter(0.5, 5.0, default=1.0, space="sell", optimize=True, load=True)
    
    # Número máximo de niveles en la grid
    max_grid_levels = IntParameter(2, 20, default=5, space="sell", optimize=True, load=True)
    
    # Número de velas necesarias antes de producir señales válidas
    startup_candle_count: int = 1  # No necesitamos muchas velas para una estrategia basada solo en precio

    # Configuración de tipos de órdenes
    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }

    # Configuración de tiempo en vigor de las órdenes
    order_time_in_force = {"entry": "GTC", "exit": "GTC"}

    # Diccionario para almacenar información de la grid para cada par
    grid_info = {}

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        No utilizamos indicadores técnicos, solo operamos basados en niveles de precio.
        """
        # No añadimos indicadores técnicos
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Define las condiciones para entrar en posiciones cortas.
        Para un grid bot de shorts, generamos una señal inicial para la primera entrada.
        """
        # Inicializar columna de entrada en corto
        dataframe['enter_short'] = 0
        
        # Generar señal para la primera entrada
        # Simplemente tomamos la primera vela como punto de entrada inicial
        dataframe.loc[0, 'enter_short'] = 1
        
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Define las condiciones para salir de posiciones cortas.
        Para un grid bot, las salidas son gestionadas por adjust_trade_position y custom_exit.
        """
        # No utilizamos señales de salida basadas en dataframe
        dataframe['exit_short'] = 0
        
        return dataframe

    def adjust_trade_position(self, trade: Trade, current_time: datetime, current_rate: float,
                             current_profit: float, min_stake: float, max_stake: float,
                             **kwargs) -> Optional[float]:
        """
        Implementa la lógica de grid trading para posiciones cortas.
        Añade nuevas posiciones cortas cuando el precio sube a los niveles definidos.
        """
        # Verificar si ya hemos alcanzado el número máximo de niveles en la grid
        if trade.nr_of_successful_entries >= self.max_grid_levels.value:
            return None
        
        # Obtener o inicializar la información de la grid para este par
        pair = trade.pair
        if pair not in self.grid_info:
            self.grid_info[pair] = {
                'base_price': trade.open_rate,
                'grid_levels': [trade.open_rate],
                'level_profits': {}
            }
        
        # Calcular el porcentaje de grid para entradas en corto
        grid_pct = self.grid_short_entry_pct.value / 100.0
        
        # Obtener el precio de la última entrada
        last_entry_price = self.grid_info[pair]['grid_levels'][-1]
        
        # Calcular el próximo nivel de precio para la grid
        # Para shorts: añadir posición cuando el precio sube
        next_entry_price = last_entry_price * (1 + grid_pct)
        
        # Si el precio ha alcanzado el siguiente nivel de la grid, añadir una nueva posición corta
        if current_rate >= next_entry_price:
            # Registrar este nuevo nivel en nuestra grid
            self.grid_info[pair]['grid_levels'].append(next_entry_price)
            
            # Calcular el tamaño de la nueva entrada (igual al tamaño original)
            new_entry_size = trade.stake_amount
            
            # Asegurar que no exceda el máximo permitido
            return min(new_entry_size, max_stake)
        
        # Si no se cumplen las condiciones, no añadir nueva entrada
        return None

    def confirm_trade_exit(self, pair: str, trade: 'Trade', order_type: str, amount: float,
                          rate: float, time_in_force: str, exit_reason: str, **kwargs) -> bool:
        """
        Confirma la salida de una operación.
        Para un grid bot de shorts, permitimos salidas parciales cuando un nivel alcanza su objetivo.
        """
        # Calcular beneficio actual
        current_profit = trade.calc_profit_ratio(rate)
        
        # Obtener el objetivo de beneficio
        profit_target = self.profit_target_pct.value / 100.0
        
        # Si es una salida parcial (cierre de un nivel específico)
        if amount != trade.amount and current_profit >= profit_target:
            return True
        
        # Si es una salida completa, verificar si todas las entradas están en beneficio
        if amount == trade.amount and current_profit >= profit_target:
            return True
        
        # En caso contrario, no permitir la salida
        return False
    
    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                   current_profit: float, **kwargs) -> Optional[str]:
        """
        Implementa salidas personalizadas para niveles específicos de la grid.
        Para shorts, cerramos posiciones cuando el precio baja y alcanzamos el objetivo de beneficio.
        """
        # Obtener el objetivo de beneficio
        profit_target = self.profit_target_pct.value / 100.0
        
        # Verificar si hemos alcanzado el objetivo de beneficio
        if current_profit >= profit_target:
            # Registrar este beneficio en nuestro seguimiento de la grid
            if pair in self.grid_info and 'level_profits' in self.grid_info[pair]:
                level = len(self.grid_info[pair]['level_profits']) + 1
                self.grid_info[pair]['level_profits'][level] = current_profit
            
            return "grid_profit_target_reached"
        
        return None
    
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                           proposed_stake: float, min_stake: float, max_stake: float, **kwargs) -> float:
        """
        Personaliza el tamaño de la posición para cada nivel de la grid.
        """
        # Usar el tamaño propuesto por la estrategia
        return proposed_stake
