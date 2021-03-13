import os
import pandas as pd
import numpy as np


def get_exts(fname):
    root, ext = os.path.splitext(fname)
    if os.path.splitext(root)[1]:
        return get_exts(root) + [ext]
    else:
        return [ext]


def nan_info(data: pd.DataFrame):
    df = data.isna().sum().to_frame("Total")
    df["Percent"] = (df["Total"] / data.shape[0]) * 100
    return df.sort_values("Total", ascending=False)


def nan_rows(data: pd.DataFrame):
    return data[data.isna().any(axis=1)]


def who_is_nan(data: pd.DataFrame, col: str, name_col: str):
    return nan_rows(data)[data[col].isna()][name_col]


def print_categorical_uniques(data: pd.DataFrame, cut=50, skip=None):
    cat_cols = (data.dtypes == "object") & (data.nunique() <= cut)
    cat_cols = data.columns[cat_cols]
    if skip:
        cat_cols = [x for x in cat_cols if x not in skip]
    for col in cat_cols:
        print(col)
        print("-" * len(col))
        print(data[col].unique())
        print("\n")


def show_numerical_value_counts(data: pd.DataFrame):
    is_numeric = data.dtypes.map(pd.api.types.is_numeric_dtype)
    num_only = data[data.columns[is_numeric]]
    for col in num_only.columns:
        display(num_only[col].value_counts().head())


def counter_index(data):
    copy = pd.Series(data.index.sort_values())
    counts = copy.groupby(copy.values).count()
    counter = np.hstack(counts.map(np.arange).values)
    index = pd.MultiIndex.from_arrays([counter, copy.values])
    return index


def explode_wide(data, column):
    expl = data[column].explode()
    expl.index = counter_index(expl)
    expl = expl.unstack(0)
    expl.columns = expl.columns.map(lambda x: f"{column}_{x+1}")
    return pd.concat([data, expl], axis=1).drop(column, axis=1)