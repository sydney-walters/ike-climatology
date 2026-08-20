"""
categories.py

Saffir-Simpson Hurricane Wind Scale (SSHWS) categorization of maximum
sustained wind, used by the category-breakdown figure(s).
"""

import numpy as np

# Category order used consistently across category-breakdown figures.
CATEGORY_ORDER = ["TS", "Cat 1", "Cat 2", "Cat 3", "Cat 4", "Cat 5"]

# Includes "TD" for completeness of the classification; figures that only
# plot tropical-storm-strength-and-above categories filter "TD" out
# explicitly rather than omitting it here.
ALL_CATEGORY_ORDER = ["TD"] + CATEGORY_ORDER


def get_saffir_simpson_cat(vmax_kts):
    """
    Categorize maximum sustained wind (knots) into a Saffir-Simpson
    category, or "TS"/"TD" for sub-hurricane intensity.

    Parameters
    ----------
    vmax_kts : float
        Maximum sustained wind, in knots. NaN returns NaN.

    Returns
    -------
    str or float
        One of "TD", "TS", "Cat 1".."Cat 5", or np.nan if vmax_kts is NaN.
    """
    if np.isnan(vmax_kts):
        return np.nan
    if vmax_kts < 34:
        return "TD"
    elif vmax_kts < 64:
        return "TS"
    elif vmax_kts < 83:
        return "Cat 1"
    elif vmax_kts < 96:
        return "Cat 2"
    elif vmax_kts < 113:
        return "Cat 3"
    elif vmax_kts < 137:
        return "Cat 4"
    else:
        return "Cat 5"
