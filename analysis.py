import pandas as pd
import numpy as np
import utils


def pairwise_corr(data):
    corr_df = data.corr()
    mask = utils.triangle_mask(corr_df)
    corr_df = corr_df.where(mask)
    return corr_df.stack()


def expl_combo_table(data, column):
    cat_names = data[column].explode().sort_values().unique()
    cat_combos = pd.DataFrame(columns=cat_names, index=data.index)
    for index, row in cat_combos.iterrows():
        cats_at_index = data.loc[index, column]
        cat_combos.loc[index] = row.index.isin(cats_at_index)
    return cat_combos.apply(lambda x: x.astype(np.bool_))


def combo_table(source, index_col, columns_col):
    combos = pd.DataFrame(index=source.tconst.unique(), columns=source.nconst.unique())
    mapping = source.set_index(index_col)[columns_col]
    for index, row in combos.iterrows():
        cats_at_index = mapping[index]
        if isinstance(cats_at_index, pd.Series):
            cats_at_index = mapping[index].values
            combos.loc[index] = row.index.isin(cats_at_index)
        else:
            combos.loc[index] = row.index == cats_at_index
    return combos.astype(np.bool_)