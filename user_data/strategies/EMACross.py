from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta

class EMACross(IStrategy):

    timeframe = '1m'

    stoploss = -0.10

    can_short = True

    minimal_roi = {"0": 0.01}

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['ema_slow'] = ta.EMA(dataframe, timeperiod=20)
        dataframe['ema_fast'] = ta.EMA(dataframe, timeperiod=9)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe['ema_slow'] < dataframe['ema_fast']),
            'enter_long'] = 1 

        dataframe.loc[
            (dataframe['ema_slow'] > dataframe['ema_fast']),
            'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe['ema_slow'] > dataframe['ema_fast']),
            'exit_long'] = 1

        dataframe.loc[
            (dataframe['ema_slow'] < dataframe['ema_fast']),
            'exit_short'] = 1

        return dataframe