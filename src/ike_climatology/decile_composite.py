"""
decile_composite.py

Per-storm IKE/Vmax/Pmin/R34 with tropical-phase masking, robust ET
anchoring, a wavenumber-1 R34 asymmetry index, and SHIPS shear
(magnitude + direction) merge -- feeding the upper-vs-lower-quartile
composite analysis of pre-ET, high-shear rapid IKE change (North
Atlantic vs. West Pacific).
"""

import os

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats as scipy_stats
from scipy.stats import mannwhitneyu

from .storm_metrics import M_PER_NM, KT_TO_MS
from .wind_asymmetry import robust_decode, load_specific_ships_file
from .stats import rank_biserial_mwu

TROPICAL_PHASES = ["TS", "HU", "TY", "ST", "TC"]
ET_PHASES = ["EX", "ET"]
SECONDS_PER_HOUR = 3600

DEFAULT_LAG_PREDICTORS = [
    "vmax_kts", "pmin_mb", "r34_nm", "r34_asymmetry",
    "storm_lat_deg", "storm_speed_kts", "total_ike_tj", "storm_heading_shifted", "shear_mag_ms",
]


def calculate_asymmetry_storm_metrics(storm_data, land_interaction_threshold_km, time_index):
    """
    Compute per-timestep IKE, tropical-phase-masked Vmax/Pmin/R34, a
    wavenumber-1 R34 asymmetry index, and robustly-anchored
    time_to_et_hrs, for a single storm.

    The asymmetry index is A1 / mean(R34), where A1 is the magnitude of
    the wavenumber-1 (NE-SW vs. SE-NW) component of the quadrant-resolved
    R34 field: 0 = perfectly symmetric, larger = more asymmetric.

    Returns
    -------
    pandas.DataFrame
        Indexed by time_index, with total_ike_tj, vmax_kts/pmin_mb/r34_nm/r34_km
        (NaN outside tropical phases), r34_asymmetry (NaN outside
        tropical phases or where any quadrant R34 is missing),
        storm_dir_deg, storm_heading_shifted (storm_dir_deg shifted to
        -180..180), storm_speed_kts, landfall_dist_km, had_landfall,
        storm_lat_deg/storm_lon_deg, and time_to_et_hrs.
    """
    rmw_m = storm_data["usa_rmw"].fillna(0).values * M_PER_NM
    vmax_ms = storm_data["usa_wind"].fillna(0).values * KT_TO_MS
    mpres_mb = storm_data["usa_pres"].fillna(np.nan).values

    r34_m = storm_data["usa_r34"].fillna(0).values * M_PER_NM
    r50_m = storm_data["usa_r50"].fillna(0).values * M_PER_NM
    r64_m = storm_data["usa_r64"].fillna(0).values * M_PER_NM

    # Quadrant-resolved R34 in native (nm) units, kept unfilled -- NaN
    # quadrants must stay NaN so the asymmetry index below can detect and
    # skip incomplete quadrant reports rather than silently treating a
    # missing quadrant as 0.
    r34_quad_nm = storm_data["usa_r34"].values

    storm_dir_deg = storm_data["storm_dir"].fillna(np.nan).values
    storm_speed_kts = storm_data["storm_speed"].fillna(0).values
    landfall_km = storm_data["landfall"].fillna(9999).values
    storm_lat_deg = storm_data["lat"].fillna(np.nan).values
    storm_lon_deg = storm_data["lon"].fillna(np.nan).values
    storm_heading_shifted = np.where(storm_dir_deg > 180, storm_dir_deg - 360, storm_dir_deg)

    try:
        status_np = np.vectorize(lambda x: x.decode("utf-8").strip())(storm_data["usa_status"].values)
    except Exception:
        status_np = np.array([str(s).strip() for s in storm_data["usa_status"].values])

    is_tropical = np.isin(status_np, TROPICAL_PHASES)
    num_timesteps = len(time_index)

    ike1826_list, ike2633_list, ikehur_list, total_ike_list = [], [], [], []

    for t in range(num_timesteps):
        rmw, mwind, status = rmw_m[t], vmax_ms[t], status_np[t]
        ike_q1, ike_q2, ike_q3 = [], [], []

        if status in TROPICAL_PHASES:
            if np.any(r34_m[t, :] > 0):
                for q in range(r34_m.shape[1]):
                    r34, r50 = r34_m[t, q], r50_m[t, q]
                    if r50 > 0:
                        ike_m, ike_a = 20, 0.25 * np.pi * (r34**2 - r50**2)
                    elif r50 == 0 and mwind > 26 and r34 > rmw:
                        ike_m, ike_a = 20, 0.25 * np.pi * (r34**2 - (0.75 * rmw) ** 2)
                    elif r50 == 0 and mwind < 26 and r34 > rmw:
                        ike_m, ike_a = 0.25 * mwind + 0.75 * 18, 0.25 * np.pi * (r34**2 - (0.75 * rmw) ** 2)
                    elif r50 == 0 and r34 <= rmw:
                        ike_m, ike_a = 18, 0.25 * np.pi * (r34**2 - (0.5 * r34) ** 2)
                    else:
                        ike_m, ike_a = 0, 0
                    ike_q1.append((0.5 * ike_a * ike_m**2) / 1e12)

            if np.any(r50_m[t, :] >= 0):
                for q in range(r50_m.shape[1]):
                    r50, r64 = r50_m[t, q], r64_m[t, q]
                    if r64 > 0:
                        ike_m, ike_a = 27.75, 0.25 * np.pi * (r50**2 - r64**2)
                    elif r64 == 0 and mwind > 33 and r50 > rmw:
                        ike_m, ike_a = 27.75, 0.25 * np.pi * (r50**2 - (0.75 * rmw) ** 2)
                    elif r64== 0 and mwind < 33 and r50 > rmw:
                        ike_m, ike_a = 0.25 * mwind + 0.75 * 26, 0.25 * np.pi * (r50**2 - (0.75 * rmw) ** 2)
                    elif r64 == 0 and r50 <= rmw:
                        ike_m, ike_a = 26, 0.25 * np.pi * (r50**2 - (0.5 * r50) ** 2)
                    else:
                        ike_m, ike_a = 0, 0
                    ike_q2.append((0.5 * ike_a * ike_m**2) / 1e12)

            if np.any(r64_m[t, :] >= 0):
                for q in range(r64_m.shape[1]):
                    r64 = r64_m[t, q]
                    if r64 > rmw:
                        ike_m, ike_a = 0.25 * mwind + 0.75 * 33, 0.25 * np.pi * (r64**2 - (0.75 * rmw) ** 2)
                    elif r64 == rmw:
                        ike_m, ike_a = 0.25 * mwind + 0.75 * 33, 0.25 * np.pi * (r64**2 - (0.75 * r64) ** 2)
                    elif r64 < rmw:
                        ike_m, ike_a = 0.1 * mwind + 0.9 * 33, 0.25 * np.pi * (r64**2 - (0.75 * r64) ** 2)
                    else:
                        ike_m, ike_a = 0, 0
                    ike_q3.append((0.5 * ike_a * ike_m**2) / 1e12)

        if status in TROPICAL_PHASES and np.nansum(ike_q1) > 0:
            ike1826_list.append(np.nansum(ike_q1))
            ike2633_list.append(np.nansum(ike_q2))
            ikehur_list.append(np.nansum(ike_q3))
            total_ike_list.append(np.nansum(ike_q1) + np.nansum(ike_q2) + np.nansum(ike_q3))
        else:
            ike1826_list.append(np.nan)
            ike2633_list.append(np.nan)
            ikehur_list.append(np.nan)
            total_ike_list.append(np.nan)

    # --- Robust ET anchoring: actual datetime delta, not assumed 6-hour spacing ---
    time_to_et_hrs = np.full(num_timesteps, np.nan)
    et_indices = np.where(np.isin(status_np, ET_PHASES))[0]
    if len(et_indices) > 0 and et_indices[0] > 0:
        first_et = et_indices[0]
        last_tropical = first_et - 1
        delta = (time_index[last_tropical] - time_index).total_seconds() / SECONDS_PER_HOUR
        valid = (delta >= 0) & is_tropical
        time_to_et_hrs[valid] = delta[valid]

    storm_had_landfall = bool(np.any(landfall_km <= land_interaction_threshold_km))

    # --- Wavenumber-1 R34 asymmetry index: A1 / mean(R34) per timestep. ---
    r34_asymmetry = np.full(num_timesteps, np.nan)
    for t in range(num_timesteps):
        quads = r34_quad_nm[t, :]  # [NE, SE, SW, NW]
        if np.any(np.isnan(quads)):
            continue
        mean_r34 = np.mean(quads)
        if mean_r34 <= 0:
            continue
        a1_mag = 0.5 * np.sqrt((quads[0] - quads[2]) ** 2 + (quads[1] - quads[3]) ** 2)
        r34_asymmetry[t] = a1_mag / mean_r34

    # --- Strict tropical masking of base variables. ---
    vmax_kts_arr = storm_data["usa_wind"].values.astype(float).copy()
    pmin_arr = mpres_mb.astype(float).copy()
    r34_mean_arr = storm_data["usa_r34"].values.mean(axis=1).astype(float).copy()
    r34_km_arr = r34_mean_arr * 1.852

    non_tropical = ~is_tropical
    vmax_kts_arr[non_tropical] = np.nan
    pmin_arr[non_tropical] = np.nan
    r34_mean_arr[non_tropical] = np.nan
    r34_km_arr[non_tropical] = np.nan
    r34_asymmetry[non_tropical] = np.nan

    return pd.DataFrame(
        {
            "total_ike_tj": total_ike_list,
            "vmax_kts": vmax_kts_arr,
            "pmin_mb": pmin_arr,
            "r34_nm": r34_mean_arr,
            "r34_km": r34_km_arr,
            "r34_asymmetry": r34_asymmetry,
            "storm_dir_deg": storm_dir_deg,
            "storm_heading_shifted": storm_heading_shifted,
            "storm_speed_kts": storm_speed_kts,
            "landfall_dist_km": landfall_km,
            "storm_lat_deg": storm_lat_deg,
            "storm_lon_deg": storm_lon_deg,
            "time_to_et_hrs": time_to_et_hrs,
            "had_landfall": storm_had_landfall,
        },
        index=time_index,
    )


def load_master_dataframe_decile(
    basins,
    ibtracs_file_map,
    ships_file_map,
    basin_codes,
    ships_base_path,
    start_year,
    end_year,
    land_interaction_threshold_km=0,
    verbose=True,
):
    """
    Build the master DataFrame for the decile-composite figure: per-basin
    IBTrACS + SHIPS loading (basin skipped entirely if SHIPS is missing,
    same as asymmetry_tendency.py and shear_rate_of_change.py), genesis-
    basin filtering, asymmetry storm metrics
    (calculate_asymmetry_storm_metrics), SHIPS shear merge -- both
    magnitude and direction this time (direction is needed nowhere in
    this figure directly, but is retained for parity with the source
    script in case a future notebook wants it).

    Returns
    -------
    pandas.DataFrame
    """
    all_storms = []

    if verbose:
        print(f"--- Loading Data ({start_year}-{end_year}) ---")

    for basin_name in basins:
        if verbose:
            print(f"\n--- Loading Data for {basin_name} ---")

        ibtracs_path = ibtracs_file_map.get(basin_name)
        ships_filename = ships_file_map.get(basin_name)
        expected_basin_code = basin_codes.get(basin_name)

        ships_df = None
        if ships_filename:
            ships_df = load_specific_ships_file(os.path.join(ships_base_path, ships_filename))
        if ships_df is None:
            if verbose:
                print(f"Skipping {basin_name} due to missing SHIPS data.")
            continue

        if not ibtracs_path or not os.path.exists(ibtracs_path):
            if verbose:
                print(f"Skipping {basin_name}: IBTrACS missing.")
            continue

        with xr.open_dataset(ibtracs_path) as ds:
            season_mask = (ds["season"] >= start_year) & (ds["season"] <= end_year)
            data_filtered = ds.where(season_mask, drop=True)
            names_decoded = robust_decode(data_filtered["name"])
            if "usa_atcf_id" in data_filtered:
                atcf_ids_decoded = robust_decode(data_filtered["usa_atcf_id"])
            else:
                if verbose:
                    print("      WARNING: 'usa_atcf_id' not found; SHIPS lookup will rely on storm name only.")
                atcf_ids_decoded = [""] * len(data_filtered["storm"])
            num_storms = len(data_filtered["storm"])

            if verbose:
                print(f"Processing {num_storms} storms in {basin_name}...")
            genesis_filtered_count = 0

            for i in range(num_storms):
                storm_data = data_filtered.isel(storm=i)
                raw_time_values = storm_data["time"].values
                if not np.any(~np.isnan(raw_time_values)):
                    continue

                storm_data = storm_data.isel(date_time=~np.isnan(raw_time_values))
                raw_time_values = storm_data["time"].values

                if np.issubdtype(raw_time_values.dtype, np.datetime64):
                    storm_datetimes = pd.to_datetime(raw_time_values)
                else:
                    reference_date = pd.Timestamp("1858-11-17")
                    storm_datetimes = pd.to_timedelta(raw_time_values, unit="D") + reference_date

                synoptic_mask = np.isin(storm_datetimes.hour, [0, 6, 12, 18])
                storm_data = storm_data.isel(date_time=synoptic_mask)
                storm_datetimes = storm_datetimes[synoptic_mask]
                if len(storm_datetimes) == 0:
                    continue

                valid_indices = np.where(~np.isnan(storm_data.lat) & ~np.isnan(storm_data.lon))[0]
                if len(valid_indices) == 0:
                    continue
                first_point = valid_indices[0]

                raw_basin = storm_data["basin"].values[first_point]
                genesis_basin = (
                    raw_basin.decode("utf-8").strip()
                    if isinstance(raw_basin, (bytes, np.bytes_))
                    else str(raw_basin).strip()
                )
                if genesis_basin != expected_basin_code:
                    genesis_filtered_count += 1
                    continue

                storm_name = names_decoded[i]
                atcf_id = atcf_ids_decoded[i]
                storm_year = int(storm_data.season.item())

                storm_df = calculate_asymmetry_storm_metrics(
                    storm_data, land_interaction_threshold_km, storm_datetimes
                )
                if storm_df.empty or storm_df["vmax_kts"].isnull().all():
                    continue

                ships_subset = None
                for candidate in (str(storm_name)[:4].upper(), str(atcf_id)[:4].upper()):
                    try:
                        ships_subset = ships_df.xs(candidate, level="name_short")
                        break
                    except KeyError:
                        continue

                if ships_subset is not None:
                    try:
                        reindexed = ships_subset.reindex(
                            storm_datetimes, method="nearest", tolerance=pd.Timedelta("12 hours")
                        )
                        storm_df["shear_mag_ms"] = reindexed["shrmag_vr"].values * KT_TO_MS
                        storm_df["shear_dir_deg"] = reindexed["shrdir_vr"].values
                    except Exception:
                        storm_df["shear_mag_ms"] = np.nan
                        storm_df["shear_dir_deg"] = np.nan
                else:
                    storm_df["shear_mag_ms"] = np.nan
                    storm_df["shear_dir_deg"] = np.nan

                storm_df["storm_name"] = storm_name
                storm_df["storm_year"] = storm_year
                storm_df["basin"] = basin_name
                all_storms.append(storm_df)

            if verbose:
                print(f"        - Filtered out {genesis_filtered_count} crossover storms.")

    if not all_storms:
        return pd.DataFrame()

    master_df = pd.concat(all_storms)
    master_df.dropna(subset=["vmax_kts", "shear_mag_ms"], inplace=True)
    master_df = master_df[master_df["vmax_kts"] > 0]

    if verbose:
        print(f"\n--- Master DataFrame: {len(master_df):,} tropical-phase observations ---")

    return master_df


def add_lagged_predictors(df, lag_hours=24, predictors=None):
    """
    For each (basin, storm_year, storm_name) group, add a `{col}_lag{lag_hours}`
    column holding each predictor's value lag_hours before the current
    timestep, plus `dVmax_prior{lag_hours}` (current Vmax minus Vmax
    lag_hours ago). Synoptic data is 6-hourly, so lag_hours=24 shifts 4 rows.

    Parameters
    ----------
    df : pandas.DataFrame
    lag_hours : int, default 24
    predictors : list of str, optional
        Columns to lag. Defaults to DEFAULT_LAG_PREDICTORS.

    Returns
    -------
    pandas.DataFrame
    """
    predictors = predictors if predictors is not None else DEFAULT_LAG_PREDICTORS
    lag_periods = lag_hours // 6

    df = df.sort_values(["basin", "storm_year", "storm_name"]).copy()
    grp = df.groupby(["basin", "storm_year", "storm_name"], sort=False)

    for col in predictors:
        df[f"{col}_lag{lag_hours}"] = grp[col].shift(lag_periods)

    df[f"dVmax_prior{lag_hours}"] = df["vmax_kts"] - df[f"vmax_kts_lag{lag_hours}"]

    return df


def _default_predictor_labels(lag_hours):
    """label/unit pairs for the standard predictor set, keyed by their lagged column name."""
    return {
        f"vmax_kts_lag{lag_hours}": ("Vmax (kts)", "kts"),
        f"pmin_mb_lag{lag_hours}": ("Pmin (mb)", "mb"),
        f"r34_nm_lag{lag_hours}": ("Mean R34 (nm)", "nm"),
        f"r34_asymmetry_lag{lag_hours}": ("Wavenumber-1 R34 asymmetry", "\u2013"),
        f"storm_lat_deg_lag{lag_hours}": ("Latitude (deg)", "\u00b0N"),
        f"storm_speed_kts_lag{lag_hours}": ("Translation speed (kts)", "kts"),
        f"storm_heading_shifted_lag{lag_hours}": ("Translation direction (deg)", "\u00b0"),
        f"total_ike_tj_lag{lag_hours}": ("Total IKE (TJ)", "TJ"),
        f"shear_mag_ms_lag{lag_hours}": ("Shear magnitude (m/s)", "m/s"),
        f"dVmax_prior{lag_hours}": (f"Prior {lag_hours}-hr \u0394Vmax", "kts"),
    }


def compute_decile_composite(
    master_df,
    basin,
    upper_pct=0.75,
    lower_pct=0.25,
    lag_hours=24,
    pre_et_window_hrs=48,
    shear_threshold_ms=10,
):
    """
    Identify the upper- and lower-quartile (by 6-hour IKE change) Pre-ET
    high-shear cases for one basin, then compare them across lagged
    predictors via Mann-Whitney U + rank-biserial effect size.

    Despite the "decile" name (inherited from the source script), the
    default upper_pct/lower_pct of 0.75/0.25 are quartile cuts, not
    deciles -- kept as-is since it's a naming choice, not a computational
    bug.

    Parameters
    ----------
    master_df : pandas.DataFrame
        Output of load_master_dataframe_decile.
    basin : str
    upper_pct, lower_pct : float
        Quantile cutoffs on the 6-hour IKE-change distribution.
    lag_hours : int, default 24
    pre_et_window_hrs : float, default 48
    shear_threshold_ms : float, default 10
        Only timesteps with shear_mag_ms strictly greater than this are
        included ("high shear").

    Returns
    -------
    summary_df : pandas.DataFrame
        One row per predictor with medians, IQRs, Mann-Whitney p, and
        rank-biserial effect size.
    composite : pandas.DataFrame
        Underlying upper+lower observations, tagged with a 'group' column
        ('Upper Quartile' / 'Lower Quartile').
    predictors : dict
        column name -> (label, unit), for use by the caller's plotting code.
    thresholds : tuple of (float, float)
        (upper_thresh, lower_thresh) on dIKE_6hr, TJ per 6hr.
    """
    dfc = master_df.copy()
    dfc = add_lagged_predictors(dfc, lag_hours=lag_hours)
    dfc["dIKE_6hr"] = dfc.groupby(["basin", "storm_year", "storm_name"], sort=False)["total_ike_tj"].diff(periods=1)

    dfc = dfc[dfc["basin"] == basin].copy()
    dfc = dfc[dfc["time_to_et_hrs"].notna() & (dfc["time_to_et_hrs"] <= pre_et_window_hrs)]
    dfc = dfc[dfc["shear_mag_ms"] > shear_threshold_ms]
    dfc.dropna(subset=["dIKE_6hr"], inplace=True)
    dfc = dfc[np.isfinite(dfc["dIKE_6hr"])]

    upper_thresh = dfc["dIKE_6hr"].quantile(upper_pct)
    lower_thresh = dfc["dIKE_6hr"].quantile(lower_pct)

    upper = dfc[dfc["dIKE_6hr"] >= upper_thresh].copy()
    upper["group"] = "Upper Quartile"
    lower = dfc[dfc["dIKE_6hr"] <= lower_thresh].copy()
    lower["group"] = "Lower Quartile"

    composite = pd.concat([upper, lower], ignore_index=True)
    predictors = _default_predictor_labels(lag_hours)

    summary_rows = []
    for col, (label, unit) in predictors.items():
        u_vals = upper[col].dropna()
        l_vals = lower[col].dropna()
        if len(u_vals) < 5 or len(l_vals) < 5:
            continue
        u_med, l_med = u_vals.median(), l_vals.median()
        u_iqr = (u_vals.quantile(0.25), u_vals.quantile(0.75))
        l_iqr = (l_vals.quantile(0.25), l_vals.quantile(0.75))
        try:
            _, pval = mannwhitneyu(u_vals, l_vals, alternative="two-sided")
            r_eff = rank_biserial_mwu(u_vals, l_vals)
        except ValueError:
            pval, r_eff = np.nan, np.nan
        summary_rows.append({
            "Predictor": label, "Unit": unit,
            "Upper median": u_med, "Upper IQR": f"[{u_iqr[0]:.2f}, {u_iqr[1]:.2f}]",
            "Lower median": l_med, "Lower IQR": f"[{l_iqr[0]:.2f}, {l_iqr[1]:.2f}]",
            "Difference (U-L)": u_med - l_med,
            "Mann-Whitney p": pval, "Effect size (r)": r_eff,
            "Significant (p<0.05)": "YES" if (pd.notna(pval) and pval < 0.05) else "no",
        })

    summary_df = pd.DataFrame(summary_rows)
    return summary_df, composite, predictors, (upper_thresh, lower_thresh)


def compare_basins_within_group(
    composite_df, predictors, group_col="group", basin_col="basin",
    basin_a="North Atlantic", basin_b="West Pacific",
):
    """
    For each predictor and each quartile group (Upper/Lower), test
    whether the two basins differ significantly (Mann-Whitney U +
    rank-biserial effect size). This is the direct test for claims like
    "the pattern is reversed between the two basins" -- a within-basin
    comparison alone can't show that.

    Parameters
    ----------
    composite_df : pandas.DataFrame
        Concatenation of both basins' compute_decile_composite composite
        outputs, with a 'basin' column added.
    predictors : list of str
        Lagged column names to test.

    Returns
    -------
    pandas.DataFrame
    """
    rows = []
    for col in predictors:
        for group in composite_df[group_col].unique():
            a_vals = composite_df.loc[
                (composite_df[basin_col] == basin_a) & (composite_df[group_col] == group), col
            ].dropna()
            b_vals = composite_df.loc[
                (composite_df[basin_col] == basin_b) & (composite_df[group_col] == group), col
            ].dropna()
            if len(a_vals) < 5 or len(b_vals) < 5:
                continue
            _, p = scipy_stats.mannwhitneyu(a_vals, b_vals, alternative="two-sided")
            r_eff = rank_biserial_mwu(a_vals, b_vals)
            rows.append({
                "predictor": col, "group": group,
                f"{basin_a}_median": a_vals.median(),
                f"{basin_b}_median": b_vals.median(),
                "p": p, "r": r_eff,
            })
    return pd.DataFrame(rows)


def levene_upper_vs_lower(composite_df, col, group_col="group",
                           upper_label="Upper Quartile", lower_label="Lower Quartile"):
    """Levene's test comparing variance of `col` between the upper and lower quartile groups."""
    upper = composite_df.loc[composite_df[group_col] == upper_label, col].dropna()
    lower = composite_df.loc[composite_df[group_col] == lower_label, col].dropna()
    stat, p = scipy_stats.levene(upper, lower)
    return {"stat": stat, "p": p, "upper_n": len(upper), "lower_n": len(lower)}


def build_decile_composite_long(
    master_df, basins, lag_hours=24, upper_pct=0.75, lower_pct=0.25,
    pre_et_window_hrs=48, shear_threshold_ms=10, predictors=None,
):
    """
    Long-format, multi-basin version of the quartile-composite split (see
    compute_decile_composite for the per-basin version): applies the same
    Pre-ET/high-shear filter and upper/lower quartile split independently
    within each basin, then concatenates the results into one DataFrame
    tagged with 'basin' and 'group' -- suitable for a combined figure that
    puts basin on one axis and quartile group as the hue, rather than a
    separate figure per basin.

    Parameters
    ----------
    master_df : pandas.DataFrame
        Output of load_master_dataframe_decile.
    basins : list of str
    lag_hours : int, default 24
    upper_pct, lower_pct : float
        Quantile cutoffs on the 6-hour IKE-change distribution, computed
        independently per basin.
    pre_et_window_hrs : float, default 48
    shear_threshold_ms : float, default 10
    predictors : list of str, optional
        Columns to lag (passed to add_lagged_predictors). Defaults to a
        reduced set (vmax_kts, pmin_mb, r34_km, r34_asymmetry,
        storm_lat_deg, storm_speed_kts, total_ike_tj) -- notably without
        storm_heading_shifted/shear_mag_ms, since this combined figure's
        panels don't use them; pass DEFAULT_LAG_PREDICTORS explicitly for
        the fuller set instead.

    Returns
    -------
    pandas.DataFrame
        Columns include 'basin', 'group' ('Upper Quartile'/'Lower
        Quartile'), 'dIKE_6hr', and every `{col}_lag{lag_hours}` column.
    """
    if predictors is None:
        predictors = ["vmax_kts", "pmin_mb", "r34_km", "r34_asymmetry",
                      "storm_lat_deg", "storm_speed_kts", "total_ike_tj"]

    pieces = []
    for basin in basins:
        dfc = master_df.copy()
        dfc = add_lagged_predictors(dfc, lag_hours=lag_hours, predictors=predictors)
        dfc["dIKE_6hr"] = dfc.groupby(["basin", "storm_year", "storm_name"], sort=False)["total_ike_tj"].diff(periods=1)

        dfc = dfc[dfc["basin"] == basin].copy()
        dfc = dfc[dfc["time_to_et_hrs"].notna() & (dfc["time_to_et_hrs"] <= pre_et_window_hrs)]
        dfc = dfc[dfc["shear_mag_ms"] > shear_threshold_ms]
        dfc.dropna(subset=["dIKE_6hr"], inplace=True)
        dfc = dfc[np.isfinite(dfc["dIKE_6hr"])]

        upper_thresh = dfc["dIKE_6hr"].quantile(upper_pct)
        lower_thresh = dfc["dIKE_6hr"].quantile(lower_pct)

        upper = dfc[dfc["dIKE_6hr"] >= upper_thresh].copy()
        upper["group"] = "Upper Quartile"
        lower = dfc[dfc["dIKE_6hr"] <= lower_thresh].copy()
        lower["group"] = "Lower Quartile"

        composite = pd.concat([upper, lower], ignore_index=True)
        composite["basin"] = basin
        pieces.append(composite)

    return pd.concat(pieces, ignore_index=True)
