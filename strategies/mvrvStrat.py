# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these libs ---
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame

from freqtrade.strategy import (
    BooleanParameter,
    CategoricalParameter,
    DecimalParameter,
    IStrategy,
    IntParameter,
)

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import logging

logger = logging.getLogger(__name__)

class mvrvStrategy(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = "1h"
    minimal_roi = {"0": 1}
    stoploss = -1
    can_short = False
    trailing_stop = False
    process_only_new_candles = True
    startup_candle_count: int = 30

    # Hyperparameters
    entry_mvrv_ratio = DecimalParameter(0.0, 10.0, default=1.0, space="buy")
    exit_mvrv_ratio = DecimalParameter(0.0, 10.0, default=2.0, space="sell")
    volume_threshold = IntParameter(1, 100, default=1, space="buy_sell")

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        url = 'https://raw.githubusercontent.com/wtriantis/FreqData/main/mvrv_mayer.csv'

        # Load the data and check columns
        additional_data = pd.read_csv(url)
        logger.info(f"Columns in additional data: {additional_data.columns}")

        # Assuming the correct column name for date is 'timestamp'
        if 'timestamp' in additional_data.columns:
            additional_data['timestamp'] = pd.to_datetime(additional_data['timestamp'])
            additional_data.set_index('timestamp', inplace=True)
        elif 'date' in additional_data.columns:
            additional_data['date'] = pd.to_datetime(additional_data['date'])
            additional_data.set_index('date', inplace=True)
        else:
            raise ValueError("The CSV file does not contain a 'timestamp' or 'date' column")

        # Resample to the same frequency as the OHLCV data
        additional_data = additional_data.resample(self.timeframe).ffill()

        # Join the additional data with the dataframe and forward fill to match OHLCV data
        dataframe.set_index('date', inplace=True)
        dataframe = dataframe.join(additional_data[['BTC: MVRV Ratio']], how='left').fillna(method='ffill')
        dataframe.reset_index(inplace=True)

        # Log the dataframe columns and some sample data
        logger.info(f"Dataframe columns after joining additional data: {dataframe.columns}")
        logger.info(f"Sample dataframe data:\n{dataframe.head()}")

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (qtpylib.crossed_above(dataframe["BTC: MVRV Ratio"], self.exit_mvrv_ratio.value))
                & (dataframe["volume"] > self.volume_threshold.value)
            ),
            "enter_long",
        ] = 1
        logger.info(f"Entry signals: {dataframe[dataframe['enter_long'] == 1]}")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe["BTC: MVRV Ratio"] < self.entry_mvrv_ratio.value)
                & (dataframe["volume"] > self.volume_threshold.value)
            ),
            "exit_long",
        ] = 1
        logger.info(f"Exit signals: {dataframe[dataframe['exit_long'] == 1]}")
        return dataframe
