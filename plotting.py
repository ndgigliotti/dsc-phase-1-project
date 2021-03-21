import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import utils


def multi_hist(data, include=None, xlabel=None, bins="auto", figsize=(15, 5)):
    if not include:
        include = utils.numeric_cols(data)
    fig, axes = plt.subplots(ncols=len(include), figsize=figsize)
    for col, ax in zip(include, axes.flat):
        ax = sns.histplot(data=data, x=col, bins=bins, ax=ax)
        ax.set_title(f"Distribution of `{col}`")
        if xlabel:
            ax.set_xlabel(xlabel)
    fig.tight_layout()
    return axes

def topn_ranking(data, label, rankby, topn, palette='deep', figsize=(5, 8)):
    fig, ax = plt.subplots(figsize=figsize)
    rank_df = data.sort_values(rankby, ascending=False).head(topn)
    ax = sns.barplot(data=rank_df, x=rankby, y=label, palette=palette, ax=ax)
    return ax

def heated_barplot(series, title, xlabel, ylabel, desat=0.6, ax=None, figsize=(8, 10)):
    if not ax:
        fig, ax = plt.subplots(figsize=figsize)
    series.sort_values(ascending=False, inplace=True)
    blues = sns.color_palette("Blues", (series <= 0).sum(), desat=desat)
    reds = sns.color_palette("Reds_r", (series > 0).sum(), desat=desat)
    palette = reds + blues
    ax = sns.barplot(
        x=series.values, y=series.index, palette=palette, orient="h", ec="gray", ax=ax
    )
    ax.axvline(0.0, color="gray", lw=1, ls="-")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return ax


def cat_correlation(crosstab, other, title, ylabel, **kwargs):
    corr = crosstab.corrwith(other).dropna().sort_values(ascending=False)
    xlabel = "Correlation"
    ax = heated_barplot(corr, title, xlabel, ylabel, **kwargs)
    return ax


def cat_corr_by_bins(corr, bin1, bin2, interval1, interval2, suptitle, **kwargs):
    fig, axes = plt.subplots(ncols=2, figsize=(15, 10))
    bins = [bin1, bin2]
    intervals = [interval1, interval2]
    for bin_, interval, ax in zip(bins, intervals, axes.flat):
        data = corr.loc[bin_].dropna().sort_values(ascending=False)
        left = utils.prettify_number(interval.left)
        right = utils.prettify_number(interval.right)
        title = f"{bin_}\n\${left} to \${right}"
        xlabel = "Correlation"
        ylabel = "Genre"
        ax = heated_barplot(data, title, xlabel, ylabel, ax=ax, **kwargs)
    fig.suptitle(suptitle, fontsize=16)
    fig.tight_layout()
    return axes


def boolean_violinplots(
    crosstab,
    y_series,
    suptitle,
    xlabels=None,
    ylabel=None,
    include=None,
    palette='deep',
    size=1,
    figsize=(12, 8),
):
    ncols = 2
    nrows = int(np.ceil(crosstab.shape[1] / 2))
    if include:
        crosstab = crosstab.loc[:, include]
        nrows = int(np.ceil(len(include) / 2))
    corr = crosstab.corrwith(y_series)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    for i, ax in enumerate(axes.flat):
        ax = sns.violinplot(
            x=crosstab.iloc[:, i], y=y_series, size=size, ax=ax, palette=palette
        )
        if xlabels:
            ax.set_xlabel(xlabel[i])
        if ylabel:
            ax.set_ylabel(ylabel)
        cat_corr = np.round(corr.iloc[i], 2)
        text = f"Corr: {cat_corr}"
        ax.text(
            0.975,
            1.025,
            text,
            horizontalalignment="right",
            verticalalignment="center",
            transform=ax.transAxes,
            fontsize=12,
        )
    fig.suptitle(suptitle)
    fig.tight_layout()
    return axes
