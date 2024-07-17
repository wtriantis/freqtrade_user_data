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

class LongSpotStrategy(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = "1h"
    minimal_roi = {"0": 1}
    stoploss = -1
    can_short = False
    trailing_stop = False
    process_only_new_candles = True
    startup_candle_count: int = 30

    # Hyperparameters
    entry_sharpe_signal = DecimalParameter(0.0, 1.0, default=0.5, space="buy")
    exit_sharpe_signal = DecimalParameter(0.0, 1.0, default=0.5, space="sell")
    volume_threshold = IntParameter(1, 100, default=1, space="buy_sell")

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        url_1hr = 'https://raw.githubusercontent.com/spmcelrath/images/main/bitcoin-bitcoin-sharpe-signal-short-1hr.csv'
        url_1d = 'https://raw.githubusercontent.com/spmcelrath/images/main/bitcoin-bitcoin-sharpe-signal-short-1d.csv'

        # Load Sharpe signals from URLs
        sharpe_signals_1hr = pd.read_csv(url_1hr, index_col='timestamp', parse_dates=True)
        sharpe_signals_1d = pd.read_csv(url_1d, index_col='timestamp', parse_dates=True)

        # Merge the signals based on the timeframe
        if self.timeframe == "1h":
            sharpe_signals = sharpe_signals_1hr
        elif self.timeframe == "1d":
            sharpe_signals = sharpe_signals_1d
        else:
            raise ValueError(f"Unsupported timeframe: {self.timeframe}")

        # Rename 'value' column to 'sharpe_signal' for consistency
        sharpe_signals.rename(columns={'value': 'sharpe_signal'}, inplace=True)

        # Log the first few rows after renaming
        logger.info(f"Sharpe signals after renaming column:\n{sharpe_signals.head()}")

        # Resample to the same frequency as the OHLCV data
        sharpe_signals = sharpe_signals.resample(self.timeframe).ffill()

        # Join the signals with the dataframe and forward fill to match OHLCV data
        dataframe.set_index('date', inplace=True)
        dataframe = dataframe.join(sharpe_signals, how='left').fillna(method='ffill')
        dataframe.reset_index(inplace=True)

        # Log the dataframe columns and some sample data
        logger.info(f"Dataframe columns after joining Sharpe signals: {dataframe.columns}")
        logger.info(f"Sample dataframe data:\n{dataframe.head()}")

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe["sharpe_signal"] < self.entry_sharpe_signal.value)
                & (dataframe["volume"] > self.volume_threshold.value)
            ),
            "enter_long",
        ] = 1
        logger.info(f"Entry signals: {dataframe[dataframe['enter_long'] == 1]}")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (qtpylib.crossed_above(dataframe["sharpe_signal"], self.exit_sharpe_signal.value))
                & (dataframe["volume"] > self.volume_threshold.value)
            ),
            "exit_long",
        ] = 1
        logger.info(f"Exit signals: {dataframe[dataframe['exit_long'] == 1]}")
        return dataframe
