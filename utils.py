from collections.abc import Mapping
from operator import itemgetter

import numpy as np
import pandas as pd
import unidecode

NULL = frozenset([np.nan, pd.NA, None])


def numeric_cols(data: pd.DataFrame) -> list:
    numeric = data.dtypes.map(pd.api.types.is_numeric_dtype)
    return data.columns[numeric].to_list()


def counter_index(data):
    copy = pd.Series(data.index.sort_values())
    counts = copy.groupby(copy.values).count()
    counter = np.hstack(counts.map(np.arange).values)
    index = pd.MultiIndex.from_arrays([counter, copy.values])
    return index


def normalize_list_likes(data):
    if not isinstance(data, pd.Series):
        raise TypeError("`data` must be pd.Series")
    list_like = data.map(pd.api.types.is_list_like)
    size = data.map(len, na_action="ignore").max()
    filler = [np.nan] * size
    extended = data.copy()
    extended[~list_like] = extended.loc[~list_like].map(lambda x: [x])
    extended = extended.map(lambda x: (list(x) + filler)[:size])
    return extended


def map_list_likes(data, column, mapper):
    def transform(list_):
        if isinstance(mapper, Mapping):
            return [mapper[x] if x not in NULL else x for x in list_]
        else:
            return [mapper(x) if x not in NULL else x for x in list_]

    df = data.copy()
    df[column] = df.loc[:, column].map(transform, na_action="ignore")
    return df


def explode_wide(data, column):
    df = data.copy()
    n_cols = df[column].map(len).max()
    extended = normalize_list_likes(df[column])
    insert_after = df.columns.get_loc(column) + 1
    for i in range(n_cols):
        new_col = df[column].map(itemgetter(i))
        df.insert(insert_after + i, f"{column}_{i}", new_col)
    return df.drop(columns=column)


def triangle_mask(data: pd.DataFrame, upper=True):
    """pandas cookbook"""
    base = np.ones_like(data.values, dtype=np.bool_)
    if upper:
        mask = np.triu(base, k=1)
    else:
        mask = np.tril(base, k=-1)
    return mask
