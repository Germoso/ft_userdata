from freqtrade.strategy import IStrategy, stoploss_from_absolute
from pandas import DataFrame
import talib.abstract as ta
from datetime import datetime
from freqtrade.persistence import Trade

class BollingerReversionStrategy(IStrategy):
    minimal_roi = {"0": 0.05}
    stoploss = -1
    timeframe = '1m'
    position_adjustment_enable = True
    bollinger_period = 20
    bollinger_deviation = 2.0

    # Example specific variables
    max_entry_position_adjustment = 3
    # This number is explained a bit further down
    max_dca_multiplier = 5.5

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        bollinger = ta.BBANDS(
            dataframe,
            timeperiod=self.bollinger_period,
            nbdevup=self.bollinger_deviation,
            nbdevdn=self.bollinger_deviation,
            matype=0
        )
        dataframe['bb_upper'] = bollinger['upperband']
        dataframe['bb_lower'] = bollinger['lowerband']
        dataframe['bb_middle'] = bollinger['middleband']
        dataframe['close_prev'] = dataframe['close'].shift(1)
        return dataframe



    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                          current_rate: float, current_profit: float,
                          min_stake: float | None, max_stake: float,
                          current_entry_rate: float, current_exit_rate: float,
                          current_entry_profit: float, current_exit_profit: float,
                          **kwargs
                          ) -> float | None | tuple[float | None, str | None]:

        # Si hay órdenes abiertas, no hacer nada
        if trade.has_open_orders:
            return None

        # Configuración del DCA por drawdown
        dca_step = 0.03  # Cada 3% de caída
        dca_multipliers = [0.5, 1.0, 1.5]  # Tamaños progresivos
        max_dcas = self.max_entry_position_adjustment

        # Cuántos DCA ya hicimos
        count_entries = trade.nr_of_successful_entries

        # ¿Ya alcanzamos el máximo?
        if count_entries >= max_dcas:
            return None

        # Drawdown actual (ejemplo: -0.07 → 7% caída)
        dd = abs(current_profit)

        # Qué nivel de DCA le toca según la caída
        # ejemplo: dd = 0.07, dca_step = 0.03 → level = 2
        dca_level = int(dd / dca_step)

        # Si el nivel calculado no coincide con la cantidad actual de entradas,
        # no es el momento de hacer DCA todavía.
        if dca_level <= count_entries:
            return None

        # Tamaño base del primer trade
        base_stake = trade.select_filled_orders(trade.entry_side)[0].stake_amount_filled

        # Seleccionar multiplicador correspondiente
        # Si dca_level se pasa del listado, usa el último
        multiplier = dca_multipliers[min(dca_level - 1, len(dca_multipliers) - 1)]

        stake = base_stake * multiplier

        return stake, f"DCA_Drawdown_{dd:.2%}"



    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['close_prev'] < dataframe['bb_lower']) &
                (dataframe['close'] > dataframe['bb_lower'])
            ), 'buy'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe



    # Usa esta función para gestionar las salidas
    def custom_exit(self, pair: str, trade, current_time, current_rate, current_profit, **kwargs):
        # Obtener dataframe analizado
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

        # Última vela
        last_candle = dataframe.iloc[-1].squeeze()

        # Señal técnica de salida
        sell_signal = (
            (last_candle['close_prev'] > last_candle['bb_upper']) and
            (last_candle['close'] < last_candle['bb_upper'])
        )

        # Solo sale si hay señal técnica y profit positivo
        if sell_signal and current_profit > 0:
            return "sell_signal"

        return None