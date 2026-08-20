"""
stats.py

Pairwise and family-wise statistical comparisons across an ordered/unordered
set of groups (basins, Saffir-Simpson categories, etc.) on a numeric metric.
Three test families: Mann-Whitney U (median shift), KS 2-sample (tail/shape
difference), Fisher's exact (extreme-exceedance rate).

Generalized from group_col='basin' in the original script to an explicit
group_col parameter, so the same functions serve both basin comparisons and
category comparisons (e.g. Saffir-Simpson) without duplicating this logic.

Usage
-----
    from ike_climatology.stats import run_test_family, calculate_pairwise_stats

    families = run_test_family(
        storm_level_df, group_col='basin', group_order=basin_order,
        metrics=['peak_ike', 'storm_time_avg_ike'],
        test_fn=calculate_pairwise_stats, test_label='Medians_MW',
    )
    for metric_label, p_matrix, effect_matrix in families:
        ...  # plot p_matrix / effect_matrix as a heatmap
"""

import itertools

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
import statsmodels.stats.multitest as smt


def rank_biserial_mwu(x, y):
    """Rank-biserial effect size for an independent-samples Mann-Whitney U test."""
    x = np.asarray(x)
    y = np.asarray(y)
    n1, n2 = len(x), len(y)
    u_stat, _ = scipy_stats.mannwhitneyu(x, y, alternative="two-sided")
    return 1 - (2 * u_stat) / (n1 * n2)


def rank_biserial_wilcoxon(x, mu=0.0):
    """Matched rank-biserial effect size for a one-sample Wilcoxon signed-rank test."""
    x = np.asarray(x) - mu
    x = x[x != 0]
    n = len(x)
    ranks = scipy_stats.rankdata(np.abs(x))
    pos_sum = ranks[x > 0].sum()
    neg_sum = ranks[x < 0].sum()
    return (pos_sum - neg_sum) / (n * (n + 1) / 2)


def calculate_pairwise_stats(df, metric, group_col, group_order=None):
    """Mann-Whitney U + rank-biserial effect size. Returns RAW (uncorrected) p-values."""
    if group_order is None:
        group_order = df[group_col].dropna().unique()

    groups = [df[df[group_col] == g][metric].dropna() for g in group_order]
    non_empty_groups = [g for g in groups if len(g) > 0]
    if len(non_empty_groups) >= 2:
        h_stat, p_omnibus = scipy_stats.kruskal(*non_empty_groups)
        print(f"      > Global Kruskal-Wallis p-value ({metric} by {group_col}): {p_omnibus:.2e}")

    pairs = list(itertools.combinations(group_order, 2))
    raw_p_values, effect_sizes = [], []

    for g1, g2 in pairs:
        data1 = df[df[group_col] == g1][metric].dropna()
        data2 = df[df[group_col] == g2][metric].dropna()
        if len(data1) == 0 or len(data2) == 0:
            raw_p_values.append(np.nan)
            effect_sizes.append(np.nan)
            continue
        U, p = scipy_stats.mannwhitneyu(data1, data2, alternative="two-sided")
        raw_p_values.append(p)
        r = 1 - (2 * U) / (len(data1) * len(data2))
        effect_sizes.append(abs(r))

    return pairs, raw_p_values, effect_sizes


def calculate_ks_stats(df, metric, group_col, group_order=None):
    """KS 2-sample test. Returns RAW (uncorrected) p-values + D-statistics."""
    if group_order is None:
        group_order = df[group_col].dropna().unique()

    pairs = list(itertools.combinations(group_order, 2))
    raw_p_values, d_stats = [], []

    for g1, g2 in pairs:
        data1 = df[df[group_col] == g1][metric].dropna()
        data2 = df[df[group_col] == g2][metric].dropna()
        if len(data1) == 0 or len(data2) == 0:
            raw_p_values.append(np.nan)
            d_stats.append(np.nan)
            continue
        stat, p = scipy_stats.ks_2samp(data1, data2)
        raw_p_values.append(p)
        d_stats.append(stat)

    return pairs, raw_p_values, d_stats


def calculate_exceedance_stats(df, metric, group_col, group_order=None, threshold_percentile=0.90):
    """
    Pairwise Fisher's exact tests comparing the probability of reaching an
    'extreme' threshold, using a BASIN-BALANCED threshold: each group's own
    quantile is computed first, then averaged across groups, so no single
    large-N group's sample size dominates the cutoff.
    """
    print(f"\n    - Running Extreme Exceedance Tests (> {threshold_percentile*100:.0f}th percentile, "
          f"balanced) for {metric} by {group_col}...")

    if group_order is None:
        group_order = df[group_col].dropna().unique()

    per_group_quantiles = [
        df[df[group_col] == g][metric].dropna().quantile(threshold_percentile)
        for g in group_order
    ]
    global_threshold = np.nanmean(per_group_quantiles)

    print(f"      > Balanced {threshold_percentile*100:.0f}th percentile threshold: {global_threshold:.2f}")
    print(f"      > (Per-group percentiles: "
          f"{dict(zip(group_order, np.round(per_group_quantiles, 2)))})")

    return _fisher_pairs(df, metric, group_col, group_order, global_threshold)


def calculate_exceedance_stats_global(df, metric, group_col, group_order=None, threshold_percentile=0.90):
    """
    Pairwise Fisher's exact tests comparing the probability of reaching an
    'extreme' threshold, using a TRUE GLOBAL percentile threshold across the
    whole sample (unlike calculate_exceedance_stats, which balances by group).
    """
    print(f"\n    - Running Extreme Exceedance Tests (> {threshold_percentile*100:.0f}th percentile, "
          f"global) for {metric} by {group_col}...")

    if group_order is None:
        group_order = df[group_col].dropna().unique()

    global_threshold = df[metric].dropna().quantile(threshold_percentile)
    print(f"      > True global {threshold_percentile*100:.0f}th percentile threshold: {global_threshold:.2f}")

    return _fisher_pairs(df, metric, group_col, group_order, global_threshold)


def _fisher_pairs(df, metric, group_col, group_order, global_threshold):
    print("      > Proportion of Extreme Cases per Group:")
    for g in group_order:
        g_data = df[df[group_col] == g][metric].dropna()
        total = len(g_data)
        extreme = (g_data > global_threshold).sum()
        pct = (extreme / total) * 100 if total > 0 else 0
        print(f"          {g}: {extreme}/{total} ({pct:.1f}%)")

    pairs = list(itertools.combinations(group_order, 2))
    raw_p_values = []

    for g1, g2 in pairs:
        g1_data = df[df[group_col] == g1][metric].dropna()
        g1_extreme = (g1_data > global_threshold).sum()
        g1_normal = (g1_data <= global_threshold).sum()

        g2_data = df[df[group_col] == g2][metric].dropna()
        g2_extreme = (g2_data > global_threshold).sum()
        g2_normal = (g2_data <= global_threshold).sum()

        if (g1_extreme + g1_normal) == 0 or (g2_extreme + g2_normal) == 0:
            raw_p_values.append(np.nan)
            continue

        table = [[g1_extreme, g1_normal], [g2_extreme, g2_normal]]
        _, p = scipy_stats.fisher_exact(table, alternative="two-sided")
        raw_p_values.append(p)

    return pairs, raw_p_values, global_threshold


def build_matrix(group_order, pairs, values, symmetric_fill=1.0):
    """Turn a flat list of pairwise values into a symmetric DataFrame for heatmap plotting."""
    mat = pd.DataFrame(
        np.full((len(group_order), len(group_order)), symmetric_fill),
        index=group_order, columns=group_order,
    )
    for (g1, g2), v in zip(pairs, values):
        mat.loc[g1, g2] = v
        mat.loc[g2, g1] = v
    return mat


def run_test_family(df, group_col, group_order, metrics, test_fn, test_label, extra_kwargs=None):
    """
    Run test_fn for every metric in metrics (all measured on the same
    group_col), pool the raw p-values from ALL metrics into ONE family,
    apply a single Holm-Bonferroni correction across that family, then
    re-split the corrected p-values back per-metric.
    Parameters
    ----------
    df : pandas.DataFrame
    group_col : str
    group_order : list
    metrics : list of str
    test_fn : callable
        One of calculate_pairwise_stats, calculate_ks_stats,
        calculate_exceedance_stats, calculate_exceedance_stats_global.
        Must return (pairs, raw_p_values, effect_or_extra).
    test_label : str
        Used to build each metric's output label, e.g. "peak_ike_Medians_MW".
    extra_kwargs : dict, optional
        Extra keyword arguments passed to test_fn (e.g. threshold_percentile).

    Returns
    -------
    list of (label, p_matrix, effect_matrix)
        effect_matrix is None for test_fn's that don't return per-pair
        effect sizes (the two exceedance/Fisher functions return a scalar
        threshold instead, which isn't plotted as an effect-size heatmap).
    """
    extra_kwargs = extra_kwargs or {}
    pooled_p, per_metric_slices = [], []

    for metric in metrics:
        pairs, raw_p, effect_or_extra = test_fn(
            df, metric=metric, group_col=group_col, group_order=group_order, **extra_kwargs
        )
        per_metric_slices.append((metric, pairs, effect_or_extra, len(raw_p)))
        pooled_p.extend(raw_p)

    pooled_p_arr = np.array(pooled_p, dtype=float)
    valid_mask = ~np.isnan(pooled_p_arr)
    p_corrected_full = np.full_like(pooled_p_arr, np.nan)

    if valid_mask.sum() > 0:
        _, p_corrected_valid, _, _ = smt.multipletests(pooled_p_arr[valid_mask], alpha=0.05, method="holm")
        p_corrected_full[valid_mask] = p_corrected_valid

    print(f"    - [{test_label}] Global Holm correction applied across "
          f"{int(valid_mask.sum())} valid tests ({len(metrics)} metrics x "
          f"{len(list(itertools.combinations(group_order, 2)))} pairs).")

    results = []
    idx = 0
    for metric, pairs, effect_or_extra, n in per_metric_slices:
        p_slice = p_corrected_full[idx: idx + n]
        idx += n

        label = f"{metric}_{test_label}"
        p_matrix = build_matrix(group_order, pairs, p_slice, symmetric_fill=1.0)

        effect_matrix = None
        if isinstance(effect_or_extra, list):
            effect_matrix = build_matrix(group_order, pairs, effect_or_extra, symmetric_fill=0.0)

        results.append((label, p_matrix, effect_matrix))

    return results
