# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these libs ---
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame
from functools import reduce

from freqtrade.strategy import (
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

class glassnodeWeightedStrategy(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = "1d"
    minimal_roi = {"0": 1}
    stoploss = -1
    can_short = False
    trailing_stop = False
    process_only_new_candles = True
    startup_candle_count: int = 30

    # Hyperparameters for weights
    buy_weight_mvrv = DecimalParameter(0, 1, default=0.5, space="buy")
    buy_weight_spss = DecimalParameter(0, 1, default=0.5, space="buy")
    buy_weight_nupl = DecimalParameter(0, 1, default=0.5, space="buy")
    buy_weight_realized = DecimalParameter(0, 1, default=0.5, space="buy")
    buy_weight_short_term_activity = DecimalParameter(0, 1, default=0.5, space="buy")
    buy_weight_short_term_ratio = DecimalParameter(0, 1, default=0.5, space="buy")
    buy_weight_miners = DecimalParameter(0, 1, default=0.5, space="buy")
    buy_weight_holder_spending = DecimalParameter(0, 1, default=0.5, space="buy")
    buy_weight_holder_mvrv = DecimalParameter(0, 1, default=0.5, space="buy")
    buy_weight_exchange_volume = DecimalParameter(0, 1, default=0.5, space="buy")

    sell_weight_mvrv = DecimalParameter(0, 1, default=0.5, space="sell")
    sell_weight_spss = DecimalParameter(0, 1, default=0.5, space="sell")
    sell_weight_nupl = DecimalParameter(0, 1, default=0.5, space="sell")
    sell_weight_realized = DecimalParameter(0, 1, default=0.5, space="sell")
    sell_weight_short_term_activity = DecimalParameter(0, 1, default=0.5, space="sell")
    sell_weight_short_term_ratio = DecimalParameter(0, 1, default=0.5, space="sell")
    sell_weight_miners = DecimalParameter(0, 1, default=0.5, space="sell")
    sell_weight_holder_spending = DecimalParameter(0, 1, default=0.5, space="sell")
    sell_weight_holder_mvrv = DecimalParameter(0, 1, default=0.5, space="sell")
    sell_weight_exchange_volume = DecimalParameter(0, 1, default=0.5, space="sell")

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        mvrv_url = 'https://raw.githubusercontent.com/wtriantis/FreqData/main/bitcoin-mvrv-and-mayer-multiple-pricing-models-signal.csv'
        spss_url = 'https://raw.githubusercontent.com/wtriantis/FreqData/main/bitcoin-supply-profitability-state-signal.csv'
        nupl_url = 'https://raw.githubusercontent.com/wtriantis/FreqData/main/bitcoin-net-unrealized-profit-loss-signal.csv'
        realized_url = 'https://raw.githubusercontent.com/wtriantis/FreqData/main/bitcoin-realized-profit-loss-ratio-14d-ma-signal.csv'
        short_term_activity_url = 'https://raw.githubusercontent.com/wtriantis/FreqData/main/bitcoin-short-term-holder-activity-in-profit-loss-signal.csv'
        short_term_ratio_url = 'https://raw.githubusercontent.com/wtriantis/FreqData/main/bitcoin-short-term-holder-supply-profit-loss-ratio-signal.csv'
        miners_url = 'https://raw.githubusercontent.com/wtriantis/FreqData/main/bitcoin-miners-fee-revenue-binary-indicator-signal.csv'
        holder_spending_url = 'https://raw.githubusercontent.com/wtriantis/FreqData/main/bitcoin-long-term-holder-spending-binary-indicator-7d-signal.csv'
        holder_mvrv_url = 'https://raw.githubusercontent.com/wtriantis/FreqData/main/bitcoin-long-term-holder-mvrv-signal.csv'
        exchange_volume_url = 'https://raw.githubusercontent.com/wtriantis/FreqData/main/bitcoin-exchange-volume-momentum-signal.csv'

        def load_and_clean_data(url, prefix):
            # Load the data
            additional_data = pd.read_csv(url)
            logger.info(f"Columns in additional data from {url}: {additional_data.columns}")

            # Handle timestamp columns
            for col in additional_data.columns:
                if 'timestamp' in col:
                    additional_data[col] = pd.to_datetime(additional_data[col], errors='coerce')

            # Combine all timestamp columns into one, taking the first non-null value
            additional_data['timestamp'] = additional_data.apply(lambda row: next((row[col] for col in additional_data.columns if 'timestamp' in col and pd.notnull(row[col])), pd.NaT), axis=1)

            # Drop the original timestamp columns
            additional_data = additional_data.drop(columns=[col for col in additional_data.columns if 'timestamp' in col and col != 'timestamp'])

            # Set the new timestamp column as the index
            additional_data.set_index('timestamp', inplace=True)

            # Add prefix to columns
            additional_data = additional_data.add_prefix(prefix + '_')

            return additional_data

        # Load and clean datasets with respective prefixes
        mvrv_data = load_and_clean_data(mvrv_url, 'mvrv')
        spss_data = load_and_clean_data(spss_url, 'spss')
        nupl_data = load_and_clean_data(nupl_url, 'nupl')
        realized_data = load_and_clean_data(realized_url, 'realized')
        short_term_activity_data = load_and_clean_data(short_term_activity_url, 'short_term_activity')
        short_term_ratio_data = load_and_clean_data(short_term_ratio_url, 'short_term_ratio')
        miners_data = load_and_clean_data(miners_url, 'miners')
        holder_spending_data = load_and_clean_data(holder_spending_url, 'holder_spending')
        holder_mvrv_data = load_and_clean_data(holder_mvrv_url, 'holder_mvrv')
        exchange_volume_data = load_and_clean_data(exchange_volume_url, 'exchange_volume')

        # Combine all datasets
        additional_data = (
            mvrv_data
            .join(spss_data, how='outer')
            .join(nupl_data, how='outer')
            .join(realized_data, how='outer')
            .join(short_term_activity_data, how='outer')
            .join(short_term_ratio_data, how='outer')
            .join(miners_data, how='outer')
            .join(holder_spending_data, how='outer')
            .join(holder_mvrv_data, how='outer')
            .join(exchange_volume_data, how='outer')
        )

        # Resample to the same frequency as the OHLCV data
        additional_data = additional_data.resample(self.timeframe).ffill()

        # Join the additional data with the dataframe and forward fill to match OHLCV data
        dataframe.set_index('date', inplace=True)
        dataframe = dataframe.join(additional_data, how='left').fillna(method='ffill')
        dataframe.reset_index(inplace=True)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['enter_long'] = (
            self.buy_weight_mvrv.value * dataframe.get('mvrv_Very Low Risk', 0) +
            self.buy_weight_spss.value * dataframe.get('spss_Very Low Risk', 0) +
            self.buy_weight_nupl.value * dataframe.get('nupl_Very Low Risk', 0) +
            self.buy_weight_realized.value * dataframe.get('realized_Very Low Risk', 0) +
            self.buy_weight_short_term_activity.value * dataframe.get('short_term_activity_Very Low Risk', 0) +
            self.buy_weight_short_term_ratio.value * dataframe.get('short_term_ratio_Very Low Risk', 0) +
            self.buy_weight_miners.value * dataframe.get('miners_Very Low Risk', 0) +
            self.buy_weight_holder_spending.value * dataframe.get('holder_spending_Very Low Risk', 0) +
            self.buy_weight_holder_mvrv.value * dataframe.get('holder_mvrv_Very Low Risk', 0) +
            self.buy_weight_exchange_volume.value * dataframe.get('exchange_volume_Very Low Risk', 0)
        )

        dataframe['enter_long'] = dataframe['enter_long'] > 0.5  # You can adjust this threshold

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['exit_long'] = (
            self.sell_weight_mvrv.value * dataframe.get('mvrv_Very High Risk', 0) +
            self.sell_weight_spss.value * dataframe.get('spss_Very High Risk', 0) +
            self.sell_weight_nupl.value * dataframe.get('nupl_Very High Risk', 0) +
            self.sell_weight_realized.value * dataframe.get('realized_Very High Risk', 0) +
            self.sell_weight_short_term_activity.value * dataframe.get('short_term_activity_Very High Risk', 0) +
            self.sell_weight_short_term_ratio.value * dataframe.get('short_term_ratio_Very High Risk', 0) +
            self.sell_weight_miners.value * dataframe.get('miners_Very High Risk', 0) +
            self.sell_weight_holder_spending.value * dataframe.get('holder_spending_Very High Risk', 0) +
            self.sell_weight_holder_mvrv.value * dataframe.get('holder_mvrv_Very High Risk', 0) +
            self.sell_weight_exchange_volume.value * dataframe.get('exchange_volume_Very High Risk', 0)
        )

        dataframe['exit_long'] = dataframe['exit_long'] > 0.5  # You can adjust this threshold

        return dataframe