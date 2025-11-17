# --- Martingale DCA Strategy for Freqtrade ---
# Author: ChatGPT (GPT-5)
# Date: 2025-11-09

from freqtrade.strategy import IStrategy, DecimalParameter, IntParameter
from pandas import DataFrame
import numpy as np

class MartingaleDcaStrategy(IStrategy):
    INTERFACE_VERSION = 3

    # --- Configuración básica ---
    timeframe = '5m'
    startup_candle_count = 50
    use_custom_stoploss = True
    position_adjustment_enable = True

    # --- Parámetros principales ---
    minimal_roi = {"0": 0.015}  # 1.5%
    stoploss = -0.10             # -10%
    trailing_stop = False

    # --- Parámetros de DCA/Martingala ---
    dca_levels = 3                   # cuántas veces hará DCA
    dca_trigger = -0.02              # -2% para reentrar
    martingale_multiplier = 2.0      # multiplica el tamaño
    base_order_size = 1.0            # tamaño inicial (porcentaje del saldo)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Indicadores simples: EMA rápida y lenta
        dataframe['ema_fast'] = dataframe['close'].ewm(span=9, adjust=False).mean()
        dataframe['ema_slow'] = dataframe['close'].ewm(span=21, adjust=False).mean()
        dataframe['ema_cross'] = dataframe['ema_fast'] - dataframe['ema_slow']
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe['ema_fast'] > dataframe['ema_slow']) &
            (dataframe['ema_cross'].shift(1) <= 0),  # cruce alcista reciente
            'buy'
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # dataframe.loc[
        #     (dataframe['close'] > dataframe['ema_fast']) &
        #     (dataframe['ema_fast'] < dataframe['ema_slow']),
        #     'sell'
        # ] = 1
        return dataframe

    # --- Ajuste de posición (Martingala / DCA) ---
    def adjust_trade_position(self, trade, current_time, current_rate, current_profit, **kwargs):
        """
        Se llama automáticamente cuando una operación está abierta.
        Si el precio cae más de X%, abre otra posición mayor.
        """
        if trade.nr_of_successful_buys >= self.dca_levels:
            return None  # límite de DCA alcanzado

        # porcentaje desde el precio promedio actual
        trigger = self.dca_trigger * (trade.nr_of_successful_buys + 1)

        if current_profit <= trigger:
            # calcula nuevo tamaño
            stake_amount = self.base_order_size * (self.martingale_multiplier ** trade.nr_of_successful_buys)
            return stake_amount
        return None

    # --- Stop loss personalizado (para cerrar si es necesario) ---
    def custom_stoploss(self, pair, trade, current_time, current_rate, current_profit, **kwargs):
        # Si la pérdida excede el stoploss configurado, cerrar
        if current_profit < self.stoploss:
            return 0.01  # cerrar inmediatamente
        return 1  # mantener abierta