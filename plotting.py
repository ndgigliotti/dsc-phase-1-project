import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import utils


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
    size=1,
    figsize=(12, 8),
):
    ncols = 2
    nrows = int(np.ceil(crosstab.shape[1] / 2))
    if include:
        crosstab = crosstab.loc[:, include]
        nrows = int(np.ceil(len(include) / 2))
    crosstab = crosstab.astype(np.bool_)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    for i, ax in enumerate(axes.flat):
        ax = sns.violinplot(
            x=crosstab.iloc[:, i],
            y=y_series,
            size=size,
            ax=ax,
            palette="muted",
            split=True,
        )
        if xlabels:
            ax.set_xlabel(xlabel[i])
        if ylabel:
            ax.set_ylabel(ylabel)
    fig.suptitle(suptitle)
    fig.tight_layout()
    return axes
