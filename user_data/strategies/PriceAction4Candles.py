# Importar librerías de Freqtrade
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import talib.abstract as ta

class PriceAction4Candles(IStrategy):
    # Parámetros básicos
    minimal_roi = {"0": 0.1}  # Ajusta según quieras
    stoploss = -0.1           # Ajusta según riesgo
    timeframe = '1h'          # Temporalidad recomendada: 1h, 4h o 1d
    startup_candle_count = 210 # SMA200 + 10 para lookback

    # Parámetros configurables
    ema_long_period = 200
    ema_short_period = 100
    lookback_candles = 4
    only_long = True

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['ema_short'] = ta.EMA(dataframe, timeperiod=self.ema_short_period)
        dataframe['ema_long'] = ta.EMA(dataframe, timeperiod=self.ema_long_period)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['long_condition'] = (
            (dataframe['ema_short'] > dataframe['ema_long']) | (not self.only_long)
        ) & (
            dataframe['close'] > dataframe['high'].shift(1).rolling(self.lookback_candles).max()
        )

        dataframe.loc[dataframe['long_condition'], 'buy'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Salida cuando cierra por debajo del mínimo de las últimas 4 velas
        dataframe['sell_condition'] = (
            dataframe['close'] < dataframe['low'].shift(1).rolling(self.lookback_candles).min()
        )

        dataframe.loc[dataframe['sell_condition'], 'sell'] = 1
        return dataframe