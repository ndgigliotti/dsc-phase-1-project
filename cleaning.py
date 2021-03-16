import re
import json
import unidecode
from string import punctuation
import pandas as pd
import utils

RE_PUNCT = re.compile(f"[{re.escape(punctuation)}]")
RE_WHITESPACE = re.compile(r"\s+")


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
    for col in utils.numeric_cols(data):
        display(data[col].value_counts().head())


def process_strings(strings: pd.Series) -> pd.Series:
    df = strings.str.lower()
    df = df.str.replace(RE_PUNCT, "").str.replace(RE_WHITESPACE, " ")
    df = df.map(unidecode.unidecode, na_action="ignore")
    return df


def detect_json_list(x):
    return isinstance(x, str) and bool(re.fullmatch(r"\[.*\]", x))


def coerce_list_likes(data):
    if not isinstance(data, pd.Series):
        raise TypeError("`data` must be pd.Series")
    json_strs = data.map(detect_json_list, na_action="ignore")
    clean = data.copy()
    clean[json_strs] = clean.loc[json_strs].map(json.loads)
    list_like = clean.map(pd.api.types.is_list_like)
    clean[~list_like] = clean.loc[~list_like].map(lambda x: [x], na_action="ignore")
    clean = clean.map(list, na_action="ignore")
    return clean


def find_outliers(data: pd.Series) -> pd.Series:
    q1 = data.quantile(0.25, interpolation="midpoint")
    q3 = data.quantile(0.75, interpolation="midpoint")
    iqr = q3 - q1
    min_cut = q1 - 1.5 * iqr
    max_cut = q3 + 1.5 * iqr
    return (data < min_cut) | (data > max_cut)


def outlier_info(data: pd.DataFrame) -> pd.DataFrame:
    df = data[utils.numeric_cols(data)]
    df = df.apply(find_outliers).sum().to_frame("Total")
    df["Percent"] = (df["Total"] / data.shape[0]) * 100
    return df.sort_values("Total", ascending=False)