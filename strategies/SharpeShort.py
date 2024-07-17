# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these libs ---
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame
import os
from datetime import datetime

from freqtrade.strategy import (
    IStrategy,
)

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import logging

logger = logging.getLogger(__name__)

class ShortSharpeStrategy(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = "1h"
    minimal_roi = {"0": 1}
    stoploss = -.1
    can_short = True
    trailing_stop = False
    process_only_new_candles = True
    startup_candle_count: int = 30

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        url_1hr = 'https://raw.githubusercontent.com/spmcelrath/images/main/bitcoin-bitcoin-sharpe-signal-short-1hr.csv'

        # Load Sharpe signals from URLs
        sharpe_signals = pd.read_csv(url_1hr, index_col='timestamp', parse_dates=True)

        # Rename 'value' column to 'sharpe_signal' for consistency
        sharpe_signals.rename(columns={'value': 'sharpe_signal'}, inplace=True)

        # Resample to the same frequency as the OHLCV data
        sharpe_signals = sharpe_signals.resample(self.timeframe).ffill()

        # Join the signals with the dataframe and forward fill to match OHLCV data
        dataframe.set_index('date', inplace=True)
        dataframe = dataframe.join(sharpe_signals, how='left').fillna(method='ffill')
        dataframe.reset_index(inplace=True)

        # Save the desired columns to a CSV file
        save_columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'sharpe_signal']
        save_dataframe = dataframe[save_columns]

        # Create a unique file name with a timestamp
        save_path = "user_data/backtest_results/results.csv"
        
        # Attempt to save the DataFrame to a CSV file
        save_dataframe.to_csv(save_path, index=False)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Short entry
        dataframe.loc[
            (
                (qtpylib.crossed_above(dataframe["sharpe_signal"], 0.5))
                & (dataframe["volume"] > 0)
            ),
            "enter_short",
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Short exit
        dataframe.loc[
            (
                (dataframe["sharpe_signal"] < 0.5)
                & (dataframe["volume"] > 0)
            ),
            "exit_short",
        ] = 1

        return dataframe
