import pandas as pd
import numpy as np
import utils


def pairwise_corr(data):
    corr_df = data.corr()
    mask = utils.triangle_mask(corr_df)
    corr_df = corr_df.where(mask)
    return corr_df.stack()


def expl_combo_table(data, column, dtype=np.bool_):
    cat_names = data[column].explode().sort_values().unique()
    cat_combos = pd.DataFrame(columns=cat_names, index=data.index)
    for index, row in cat_combos.iterrows():
        cats_at_index = data.loc[index, column]
        cat_combos.loc[index] = row.index.isin(cats_at_index)
    return cat_combos.apply(lambda x: x.astype(dtype))