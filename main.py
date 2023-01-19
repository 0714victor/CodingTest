import pandas as pd
import numpy as np
import warnings
import re
import time

MONTH_CODES = "FGHJKMNQUVXZ"

MONTH_NAMES = [
    "JAN",
    "FEB",
    "MAR",
    "APR",
    "MAY",
    "JUN",
    "JUL",
    "AUG",
    "SEP",
    "OCT",
    "NOV",
    "DEC",
]

MONTH_NUMS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

MONTH_NAME_TO_CODE = {k: v for k, v in zip(MONTH_NAMES, MONTH_CODES)}

FIELDS_MAP = {
    "Trade Date": "date",
    "Risk Free Interest Rate": "RATE",
    "Open Implied Volatility": "PRICE_OPEN",
    "Last Implied Volatility": "PRICE_LAST",
    "High Implied Volatility": "PRICE_HIGH",
    "Previous Close Price": "PRICE_CLOSE_PREV",
    "Close Implied Volatility": "IMPLIEDVOL_BLACK",
    "Strike Price": "STRIKE",
    "Option Premium": "PREMIUM",
    "General Value6": "UNDL_PRICE_SETTLE",
    "General Value7": "UNDL_PRICE_LAST",
}

FLOAT_FIELDS = [
    "PRICE_OPEN",
    "PRICE_LAST",
    "PRICE_HIGH",
    "PRICE_CLOSE_PREV",
    "IMPLIEDVOL_BLACK",
    "PREMIUM",
    "RATE",
    "STRIKE",
    "UNDL_PRICE_SETTLE",
    "UNDL_PRICE_LAST",
]


def transform(raw_data_: pd.DataFrame, instruments_: pd.DataFrame) -> pd.DataFrame:
    """
    Create a function called transform that returns a normalized table.
    Do not mutate the input.
    The runtime of the transform function should be below 1 second.

    :param raw_data_: dataframe of all features associated with instruments, with associated timestamps
    :param instruments_: dataframe of all traded instruments
    """
    df = raw_data_.copy()
    if 'Error' in df.columns:
        df = df[df['error'] != 'Not Found']
    df['contract'] = np.where(df['Term'] == None, df['Period'], df['Term'])
    df['Trade Date'] = pd.to_datetime(df['Trade Date'])
    df['Expiration Date ts'] = pd.to_datetime(df['Expiration Date'])
    if df['Trade Date'].isnull().values.any():
        warnings.warn('some nulls in Trade Date')
        df = df.dropna(subset = ['Trade Date'])
    if not df[df['Trade Date'] > df['Expiration Date ts']].empty:
        warnings.warn('some trades in expired instruments')
        df = df[df['Trade Date'] <= df['Expiration Date ts']]
    if df['contract'].isnull().values.any():
        warnings.warn('some nulls in contract')
        df = df.dropna(subset = ['contract'])

    base_list = instruments_['Base']
    def get_base(ric: str) -> str:
        for b in base_list:
            if ric.startswith(b) & ric[len(b)].isnumeric():
                return b
    df['base'] = df['RIC'].map(get_base)

    def get_moneyness(ric: str) -> int:
        for b in base_list:
            if ric.startswith(b):
                s = ric[len(b):]
                res = re.findall('\d+', s)
                return res[0]
    df['moneyness'] = df['RIC'].map(get_moneyness)
    df = df.merge(instruments_, left_on = 'base', right_on = 'Base')
    df['contract_year'] = df['Expiration Date'].map(lambda x: x[6:10])
    df['contract_month'] = df['Period'].map(lambda x: x[0:3])

    def adjust_contract_year(row):
        if row['contract_year'][3] != row['Period'][3]:
            return int(row['contract_year']) + 1
        return row['contract_year']
    df['contract_year'] = df.apply(lambda row: adjust_contract_year(row), axis=1)

    df['month_code'] = df['contract_month'].map(MONTH_NAME_TO_CODE.get)
    df = df.rename(columns = FIELDS_MAP)
    df['Bloomberg Ticker'] = df['Bloomberg Ticker'].map(lambda x: x + '_' if len(x) == 1 else x)
    df['symbol'] = ('FUTURE_VOL_' + df['Exchange'] + '_' + df['Bloomberg Ticker'] +
                    df['month_code'] + df['contract_year'].astype(str) + '_' + df['moneyness'].astype(str))
    df['source'] = 'refinitiv'
    names = ['date', 'symbol', 'source'] + FLOAT_FIELDS
    df = df[names]
    df = df.set_index(['date', 'symbol', 'source'])
    stack_name = (['value'] * len(FLOAT_FIELDS))
    idx = pd.MultiIndex.from_arrays([stack_name, FLOAT_FIELDS], names = ['value', 'field'])
    df.columns = idx
    df = df.stack(level = ['field'])
    df.columns.names = [None]

    df = df.sort_values(by = 'field', kind = 'stable', key = lambda x: x.map(FLOAT_FIELDS.index))
    df = df.reset_index()
    df['value'] = df['value'].astype(str).map(lambda x: x.replace(',','')).astype(float)

    return df
    pass


if __name__ == '__main__':
    raw_data = pd.read_csv("raw_data.csv")
    instruments = pd.read_csv("instruments.csv")
    st = time.process_time()
    output = transform(raw_data, instruments)
    et = time.process_time()
    print(f"Wall time: {100 * (et-st)} ms")
    expected_output = pd.read_csv(
        "expected_output.csv",
        index_col=0,
        parse_dates=['date']
    )
    pd.testing.assert_frame_equal(output, expected_output)