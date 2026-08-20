"""
rate_of_change_matrix.py

Builds the phase-labeled, multi-variable rate-of-change table for the basin-comparison
rate-of-change matrix figure (2x2 grid: IKE/R34/Pmin/Vmax, North Atlantic
vs. West Pacific, no shear split).

"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


PHASE_ORDER = ["Early Tropical (>48hr to ET)", "Pre-ET (<48hr)"]


@dataclass(frozen=True)
class VariableSpec:
    """One variable to plot: which column, units, and how many timesteps to diff."""
    column: str
    units: str
    label: str
    diff_periods: int  # 1 = 6 hours given synoptic cadence

    @property
    def diff_col(self) -> str:
        return f"d_{self.column}_{self.diff_periods * 6}hr"


def assign_phase(df, pre_et_window_hrs=48):
    """
    Label each row Early Tropical / Pre-ET by time_to_et_hrs, dropping
    storms that never transition (time_to_et_hrs always NaN) entirely --
    same "per Chapter 2 methodology" restriction used by
    shear_rate_of_change.compute_rate_of_change_plot_df.
    """
    df = df[df["time_to_et_hrs"].notna()].copy()
    cond = [df["time_to_et_hrs"] <= pre_et_window_hrs]
    choices = ["Pre-ET (<48hr)"]
    df["phase"] = pd.Categorical(
        np.select(cond, choices, default="Early Tropical (>48hr to ET)"),
        categories=PHASE_ORDER,
        ordered=True,
    )
    return df


def add_rate_of_change_diffs(df, variables: dict):
    """
    Add one diff column per VariableSpec in `variables`, computed within
    each (basin, storm_year, storm_name) group. Unlike
    shear_rate_of_change.compute_rate_of_change_plot_df (which returns one
    narrow table per variable), this keeps every variable's diff as its
    own column on the same DataFrame, since the matrix figure plots all
    four from a single shared table.

    Parameters
    ----------
    df : pandas.DataFrame
        Output of assign_phase(...).
    variables : dict[str, VariableSpec]

    Returns
    -------
    pandas.DataFrame
    """
    df = df.sort_index()
    grouper = df.groupby(["basin", "storm_year", "storm_name"], sort=False)
    for spec in variables.values():
        df[spec.diff_col] = grouper[spec.column].diff(periods=spec.diff_periods)
    return df