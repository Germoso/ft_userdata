from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta

class RSI_Shorteable_Coin_Alerts(IStrategy):
    stoploss = -1
    can_short = True
    minimal_roi = {"0": 1}

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=7)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe['rsi'] > 60),
            'enter_long'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe['rsi'] > 20),
            'exit_long'] = 1

        return dataframe