# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these libs ---
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame
from functools import reduce

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

class glassnodeBooleanStrategy(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = "1d"
    minimal_roi = {"0": 1}
    stoploss = -1
    can_short = False
    trailing_stop = False
    process_only_new_candles = True
    startup_candle_count: int = 30

    # 
    # Hyperparameters
    buy_very_low_risk_mvrv = BooleanParameter(default=True, space="buy")
    buy_low_risk_mvrv = BooleanParameter(default=False, space="buy")
    sell_very_high_risk_mvrv = BooleanParameter(default=True, space="sell")
    sell_high_risk_mvrv = BooleanParameter(default=False, space="sell")
    
    buy_very_low_risk_spss = BooleanParameter(default=True, space="buy")
    buy_low_risk_spss = BooleanParameter(default=False, space="buy")
    sell_very_high_risk_spss = BooleanParameter(default=True, space="sell")
    sell_high_risk_spss = BooleanParameter(default=False, space="sell")
    
    buy_very_low_risk_nupl = BooleanParameter(default=True, space="buy")
    buy_low_risk_nupl = BooleanParameter(default=False, space="buy")
    sell_very_high_risk_nupl = BooleanParameter(default=True, space="sell")
    sell_high_risk_nupl = BooleanParameter(default=False, space="sell")

    buy_very_low_risk_realized = BooleanParameter(default=True, space="buy")
    buy_low_risk_realized = BooleanParameter(default=False, space="buy")
    sell_very_high_risk_realized = BooleanParameter(default=True, space="sell")
    sell_high_risk_realized = BooleanParameter(default=False, space="sell")

    buy_very_low_risk_short_term_activity = BooleanParameter(default=True, space="buy")
    buy_low_risk_short_term_activity = BooleanParameter(default=False, space="buy")
    sell_very_high_risk_short_term_activity = BooleanParameter(default=True, space="sell")
    sell_high_risk_short_term_activity = BooleanParameter(default=False, space="sell")

    buy_very_low_risk_short_term_ratio = BooleanParameter(default=True, space="buy")
    buy_low_risk_short_term_ratio = BooleanParameter(default=False, space="buy")
    sell_very_high_risk_short_term_ratio = BooleanParameter(default=True, space="sell")
    sell_high_risk_short_term_ratio = BooleanParameter(default=False, space="sell")

    buy_very_low_risk_miners = BooleanParameter(default=True, space="buy")
    buy_low_risk_miners = BooleanParameter(default=False, space="buy")
    sell_very_high_risk_miners = BooleanParameter(default=True, space="sell")
    sell_high_risk_miners = BooleanParameter(default=False, space="sell")

    buy_very_low_risk_holder_spending = BooleanParameter(default=True, space="buy")
    buy_low_risk_holder_spending = BooleanParameter(default=False, space="buy")
    sell_very_high_risk_holder_spending = BooleanParameter(default=True, space="sell")
    sell_high_risk_holder_spending = BooleanParameter(default=False, space="sell")

    buy_very_low_risk_holder_mvrv = BooleanParameter(default=True, space="buy")
    buy_low_risk_holder_mvrv = BooleanParameter(default=False, space="buy")
    sell_very_high_risk_holder_mvrv = BooleanParameter(default=True, space="sell")
    sell_high_risk_holder_mvrv = BooleanParameter(default=False, space="sell")

    buy_very_low_risk_exchange_volume = BooleanParameter(default=True, space="buy")
    buy_low_risk_exchange_volume = BooleanParameter(default=False, space="buy")
    sell_very_high_risk_exchange_volume = BooleanParameter(default=True, space="sell")
    sell_high_risk_exchange_volume = BooleanParameter(default=False, space="sell")

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

        # Initialize 'enter_long' and 'exit_long' columns to 0
        dataframe['enter_long'] = 0
        dataframe['exit_long'] = 0

        # Log the dataframe columns and some sample data
        logger.info(f"Dataframe columns after joining additional data: {dataframe.columns}")
        logger.info(f"Sample dataframe data:\n{dataframe.head()}")

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        if self.buy_very_low_risk_mvrv.value and 'mvrv_Very Low Risk' in dataframe.columns:
            conditions.append(dataframe['mvrv_Very Low Risk'] > 0)
        if self.buy_low_risk_mvrv.value and 'mvrv_Low Risk' in dataframe.columns:
            conditions.append(dataframe['mvrv_Low Risk'] > 0)
        if self.buy_very_low_risk_spss.value and 'spss_Very Low Risk' in dataframe.columns:
            conditions.append(dataframe['spss_Very Low Risk'] > 0)
        if self.buy_low_risk_spss.value and 'spss_Low Risk' in dataframe.columns:
            conditions.append(dataframe['spss_Low Risk'] > 0)
        if self.buy_very_low_risk_nupl.value and 'nupl_Very Low Risk' in dataframe.columns:
            conditions.append(dataframe['nupl_Very Low Risk'] > 0)
        if self.buy_low_risk_nupl.value and 'nupl_Low Risk' in dataframe.columns:
            conditions.append(dataframe['nupl_Low Risk'] > 0)
        if self.buy_very_low_risk_realized.value and 'realized_Very Low Risk' in dataframe.columns:
            conditions.append(dataframe['realized_Very Low Risk'] > 0)
        if self.buy_low_risk_realized.value and 'realized_Low Risk' in dataframe.columns:
            conditions.append(dataframe['realized_Low Risk'] > 0)
        if self.buy_very_low_risk_short_term_activity.value and 'short_term_activity_Very Low Risk' in dataframe.columns:
            conditions.append(dataframe['short_term_activity_Very Low Risk'] > 0)
        if self.buy_low_risk_short_term_activity.value and 'short_term_activity_Low Risk' in dataframe.columns:
            conditions.append(dataframe['short_term_activity_Low Risk'] > 0)
        if self.buy_very_low_risk_short_term_ratio.value and 'short_term_ratio_Very Low Risk' in dataframe.columns:
            conditions.append(dataframe['short_term_ratio_Very Low Risk'] > 0)
        if self.buy_low_risk_short_term_ratio.value and 'short_term_ratio_Low Risk' in dataframe.columns:
            conditions.append(dataframe['short_term_ratio_Low Risk'] > 0)
        if self.buy_very_low_risk_miners.value and 'miners_Very Low Risk' in dataframe.columns:
            conditions.append(dataframe['miners_Very Low Risk'] > 0)
        if self.buy_low_risk_miners.value and 'miners_Low Risk' in dataframe.columns:
            conditions.append(dataframe['miners_Low Risk'] > 0)
        if self.buy_very_low_risk_holder_spending.value and 'holder_spending_Very Low Risk' in dataframe.columns:
            conditions.append(dataframe['holder_spending_Very Low Risk'] > 0)
        if self.buy_low_risk_holder_spending.value and 'holder_spending_Low Risk' in dataframe.columns:
            conditions.append(dataframe['holder_spending_Low Risk'] > 0)
        if self.buy_very_low_risk_holder_mvrv.value and 'holder_mvrv_Very Low Risk' in dataframe.columns:
            conditions.append(dataframe['holder_mvrv_Very Low Risk'] > 0)
        if self.buy_low_risk_holder_mvrv.value and 'holder_mvrv_Low Risk' in dataframe.columns:
            conditions.append(dataframe['holder_mvrv_Low Risk'] > 0)
        if self.buy_very_low_risk_exchange_volume.value and 'exchange_volume_Very Low Risk' in dataframe.columns:
            conditions.append(dataframe['exchange_volume_Very Low Risk'] > 0)
        if self.buy_low_risk_exchange_volume.value and 'exchange_volume_Low Risk' in dataframe.columns:
            conditions.append(dataframe['exchange_volume_Low Risk'] > 0)

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'enter_long'
            ] = 1
        logger.info(f"Entry signals: {dataframe[dataframe['enter_long'] == 1]}")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        if self.sell_very_high_risk_mvrv.value and 'mvrv_Very High Risk' in dataframe.columns:
            conditions.append(dataframe['mvrv_Very High Risk'] > 0)
        if self.sell_high_risk_mvrv.value and 'mvrv_High Risk' in dataframe.columns:
            conditions.append(dataframe['mvrv_High Risk'] > 0)
        if self.sell_very_high_risk_spss.value and 'spss_Very High Risk' in dataframe.columns:
            conditions.append(dataframe['spss_Very High Risk'] > 0)
        if self.sell_high_risk_spss.value and 'spss_High Risk' in dataframe.columns:
            conditions.append(dataframe['spss_High Risk'] > 0)
        if self.sell_very_high_risk_nupl.value and 'nupl_Very High Risk' in dataframe.columns:
            conditions.append(dataframe['nupl_Very High Risk'] > 0)
        if self.sell_high_risk_nupl.value and 'nupl_High Risk' in dataframe.columns:
            conditions.append(dataframe['nupl_High Risk'] > 0)
        if self.sell_very_high_risk_realized.value and 'realized_Very High Risk' in dataframe.columns:
            conditions.append(dataframe['realized_Very High Risk'] > 0)
        if self.sell_high_risk_realized.value and 'realized_High Risk' in dataframe.columns:
            conditions.append(dataframe['realized_High Risk'] > 0)
        if self.sell_very_high_risk_short_term_activity.value and 'short_term_activity_Very High Risk' in dataframe.columns:
            conditions.append(dataframe['short_term_activity_Very High Risk'] > 0)
        if self.sell_high_risk_short_term_activity.value and 'short_term_activity_High Risk' in dataframe.columns:
            conditions.append(dataframe['short_term_activity_High Risk'] > 0)
        if self.sell_very_high_risk_short_term_ratio.value and 'short_term_ratio_Very High Risk' in dataframe.columns:
            conditions.append(dataframe['short_term_ratio_Very High Risk'] > 0)
        if self.sell_high_risk_short_term_ratio.value and 'short_term_ratio_High Risk' in dataframe.columns:
            conditions.append(dataframe['short_term_ratio_High Risk'] > 0)
        if self.sell_very_high_risk_miners.value and 'miners_Very High Risk' in dataframe.columns:
            conditions.append(dataframe['miners_Very High Risk'] > 0)
        if self.sell_high_risk_miners.value and 'miners_High Risk' in dataframe.columns:
            conditions.append(dataframe['miners_High Risk'] > 0)
        if self.sell_very_high_risk_holder_spending.value and 'holder_spending_Very High Risk' in dataframe.columns:
            conditions.append(dataframe['holder_spending_Very High Risk'] > 0)
        if self.sell_high_risk_holder_spending.value and 'holder_spending_High Risk' in dataframe.columns:
            conditions.append(dataframe['holder_spending_High Risk'] > 0)
        if self.sell_very_high_risk_holder_mvrv.value and 'holder_mvrv_Very High Risk' in dataframe.columns:
            conditions.append(dataframe['holder_mvrv_Very High Risk'] > 0)
        if self.sell_high_risk_holder_mvrv.value and 'holder_mvrv_High Risk' in dataframe.columns:
            conditions.append(dataframe['holder_mvrv_High Risk'] > 0)
        if self.sell_very_high_risk_exchange_volume.value and 'exchange_volume_Very High Risk' in dataframe.columns:
            conditions.append(dataframe['exchange_volume_Very High Risk'] > 0)
        if self.sell_high_risk_exchange_volume.value and 'exchange_volume_High Risk' in dataframe.columns:
            conditions.append(dataframe['exchange_volume_High Risk'] > 0)

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'exit_long'
            ] = 1

        logger.info(f"Exit signals: {dataframe[dataframe['exit_long'] == 1]}")
        return dataframe