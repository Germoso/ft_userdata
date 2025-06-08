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


class GridTradingBot(IStrategy):
    """
    Grid Trading Bot con niveles de compra y venta configurables por porcentaje.
    
    Esta estrategia:
    - Establece una cuadrícula (grid) de órdenes de compra y venta a intervalos porcentuales definidos
    - Permite configurar el porcentaje entre niveles de compra y venta
    - Funciona tanto en posiciones largas como cortas
    - Gestiona cada nivel de la grid de forma independiente
    """

    # Strategy interface version
    INTERFACE_VERSION = 3

    # Habilitar operaciones en largo y corto
    can_short: bool = True

    # Minimal ROI - configurado para cerrar con el porcentaje definido
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
    # Porcentaje entre niveles de compra
    grid_buy_pct = DecimalParameter(0.5, 5.0, default=1.0, space="buy", optimize=True, load=True)
    
    # Porcentaje entre niveles de venta
    grid_sell_pct = DecimalParameter(0.5, 5.0, default=1.0, space="sell", optimize=True, load=True)
    
    # Número máximo de niveles en la grid
    max_grid_levels = IntParameter(2, 20, default=5, space="buy", optimize=True, load=True)
    
    # Porcentaje de beneficio objetivo para cada nivel
    profit_target_pct = DecimalParameter(0.5, 5.0, default=1.0, space="sell", optimize=True, load=True)
    
    # Número de velas necesarias antes de producir señales válidas
    startup_candle_count: int = 30

    # Configuración de tipos de órdenes
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
        "main_plot": {
            "sma_200": {"color": "blue"},
            "ema_50": {"color": "red"},
        },
        "subplots": {
            "MACD": {
                "macd": {"color": "blue"},
                "macdsignal": {"color": "orange"},
            },
            "RSI": {
                "rsi": {"color": "red"},
            },
        },
    }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Calcula los indicadores necesarios para la estrategia.
        """
        # SMA
        dataframe['sma_200'] = ta.SMA(dataframe, timeperiod=200)
        
        # EMA
        dataframe['ema_50'] = ta.EMA(dataframe, timeperiod=50)
        
        # RSI
        dataframe['rsi'] = ta.RSI(dataframe)
        
        # MACD
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']
        
        # Bollinger Bands
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']
        
        # Tendencia
        dataframe['uptrend'] = (dataframe['close'] > dataframe['ema_50']) & (dataframe['ema_50'] > dataframe['sma_200'])
        dataframe['downtrend'] = (dataframe['close'] < dataframe['ema_50']) & (dataframe['ema_50'] < dataframe['sma_200'])
        
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Define las condiciones para entrar en posiciones.
        Para un grid bot, generamos señales de entrada iniciales basadas en indicadores.
        """
        # Inicializar columnas de entrada
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0
        
        # Condiciones para entrar en largo
        long_conditions = []
        
        # Condición principal para largo: precio por debajo de la banda inferior de Bollinger
        long_conditions.append(dataframe['close'] < dataframe['bb_lowerband'])
        
        # RSI en sobreventa
        long_conditions.append(dataframe['rsi'] < 30)
        
        # Verificar que el volumen no sea 0
        long_conditions.append(dataframe['volume'] > 0)
        
        # Aplicar condiciones para largo
        if long_conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, long_conditions),
                'enter_long'] = 1
        
        # Condiciones para entrar en corto
        short_conditions = []
        
        # Condición principal para corto: precio por encima de la banda superior de Bollinger
        short_conditions.append(dataframe['close'] > dataframe['bb_upperband'])
        
        # RSI en sobrecompra
        short_conditions.append(dataframe['rsi'] > 70)
        
        # Verificar que el volumen no sea 0
        short_conditions.append(dataframe['volume'] > 0)
        
        # Aplicar condiciones para corto
        if short_conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, short_conditions),
                'enter_short'] = 1
        
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Define las condiciones para salir de posiciones.
        Para un grid bot, las salidas son gestionadas principalmente por adjust_trade_position.
        """
        # Inicializar columnas de salida
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0
        
        # Condiciones para salir de largo
        long_exit_conditions = []
        
        # Salir de largo si el precio cruza por encima de la banda superior de Bollinger
        long_exit_conditions.append(dataframe['close'] > dataframe['bb_upperband'])
        
        # RSI en sobrecompra
        long_exit_conditions.append(dataframe['rsi'] > 70)
        
        # Aplicar condiciones para salir de largo
        if long_exit_conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, long_exit_conditions),
                'exit_long'] = 1
        
        # Condiciones para salir de corto
        short_exit_conditions = []
        
        # Salir de corto si el precio cruza por debajo de la banda inferior de Bollinger
        short_exit_conditions.append(dataframe['close'] < dataframe['bb_lowerband'])
        
        # RSI en sobreventa
        short_exit_conditions.append(dataframe['rsi'] < 30)
        
        # Aplicar condiciones para salir de corto
        if short_exit_conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, short_exit_conditions),
                'exit_short'] = 1
        
        return dataframe

    def adjust_trade_position(self, trade: Trade, current_time: datetime, current_rate: float,
                             current_profit: float, min_stake: float, max_stake: float,
                             **kwargs) -> Optional[float]:
        """
        Implementa la lógica de grid trading.
        Añade nuevas posiciones cuando el precio alcanza los niveles definidos.
        """
        # Obtener el dataframe con los indicadores
        dataframe = kwargs.get('dataframe', None)
        
        # Si no tenemos acceso al dataframe, no podemos tomar decisiones informadas
        if dataframe is None or len(dataframe) == 0:
            return None
        
        # Obtener la última fila del dataframe (datos actuales)
        current_candle = dataframe.iloc[-1]
        
        # Verificar si ya hemos alcanzado el número máximo de niveles en la grid
        if trade.nr_of_successful_entries >= self.max_grid_levels.value:
            return None
        
        # Determinar si estamos en una posición larga o corta
        is_short = trade.is_short
        
        # Calcular el porcentaje de grid a utilizar según la dirección
        grid_pct = self.grid_sell_pct.value if is_short else self.grid_buy_pct.value
        
        # Obtener el precio de la última entrada
        last_entry_price = trade.open_rate
        
        # Si ya hay entradas adicionales, encontrar la última
        if trade.nr_of_successful_entries > 0 and hasattr(trade, 'orders') and len(trade.orders) > 0:
            # Encontrar la última orden de entrada ejecutada
            entry_orders = [o for o in trade.orders if o.ft_order_side == 'entry' and o.status == 'closed']
            if entry_orders:
                last_entry = sorted(entry_orders, key=lambda x: x.order_date)[-1]
                last_entry_price = last_entry.price
        
        # Calcular el próximo nivel de precio para la grid
        # Para shorts: añadir posición cuando el precio sube
        # Para longs: añadir posición cuando el precio baja
        price_change_needed = grid_pct / 100.0
        
        if is_short:
            # Para shorts, el precio debe subir para añadir
            next_entry_price = last_entry_price * (1 + price_change_needed)
            should_add = current_rate >= next_entry_price
        else:
            # Para longs, el precio debe bajar para añadir
            next_entry_price = last_entry_price * (1 - price_change_needed)
            should_add = current_rate <= next_entry_price
        
        # Si el precio ha alcanzado el siguiente nivel de la grid, añadir una nueva posición
        if should_add:
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
        Para un grid bot, permitimos salidas parciales cuando un nivel alcanza su objetivo.
        """
        # Determinar si estamos en una posición larga o corta
        is_short = trade.is_short
        
        # Calcular beneficio actual
        current_profit = trade.calc_profit_ratio(rate)
        
        # Obtener el objetivo de beneficio
        profit_target = self.profit_target_pct.value / 100.0
        
        # Si es una salida parcial (cierre de un nivel específico)
        if amount != trade.amount:
            # Para shorts, el beneficio es positivo cuando el precio baja
            # Para longs, el beneficio es positivo cuando el precio sube
            if (is_short and current_profit >= profit_target) or (not is_short and current_profit >= profit_target):
                return True
        
        # Si es una salida completa, verificar si todas las entradas están en beneficio
        if amount == trade.amount and current_profit >= profit_target:
            return True
        
        # En caso contrario, no permitir la salida
        return False
    
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                           proposed_stake: float, min_stake: float, max_stake: float, **kwargs) -> float:
        """
        Personaliza el tamaño de la posición para cada nivel de la grid.
        """
        # Usar el tamaño propuesto por la estrategia
        return proposed_stake
    
    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                   current_profit: float, **kwargs) -> Optional[str]:
        """
        Implementa salidas personalizadas para niveles específicos de la grid.
        """
        # Determinar si estamos en una posición larga o corta
        is_short = trade.is_short
        
        # Obtener el objetivo de beneficio
        profit_target = self.profit_target_pct.value / 100.0
        
        # Verificar si hemos alcanzado el objetivo de beneficio
        if (is_short and current_profit >= profit_target) or (not is_short and current_profit >= profit_target):
            return "grid_profit_target_reached"
        
        return None
