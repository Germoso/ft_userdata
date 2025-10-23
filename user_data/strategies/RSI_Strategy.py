from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta


class RSI_Strategy(IStrategy):
    timeframe = '1h'
    stoploss = -0.10
    can_short = True
    minimal_roi = {
        "0": 0.10
    }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # LONG
        dataframe.loc[
            (dataframe['rsi'] < 30),
            'enter_long'
        ] = 1

        # SHORT
        dataframe.loc[
            (dataframe['rsi'] > 70),
            'enter_short'
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # EXIT LONG
        dataframe.loc[
            (dataframe['rsi'] > 70),
            'exit_long'
        ] = 1

        # EXIT SHORT
        dataframe.loc[
            (dataframe['rsi'] < 30),
            'exit_short'
        ] = 1

        return dataframe