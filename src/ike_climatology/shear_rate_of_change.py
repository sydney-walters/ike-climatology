"""
shear_rate_of_change.py

Per-storm IKE/Vmax/Pmin/R34 with strict tropical-phase masking,
extratropical-transition (ET) anchoring, merged with SHIPS shear
magnitude, feeding the shear-split rate-of-change box plot figure
(North Atlantic vs. West Pacific).
"""

import os

import numpy as np
import pandas as pd
import xarray as xr

from .storm_metrics import M_PER_NM, KT_TO_MS, NM_TO_KM
from .wind_asymmetry import robust_decode, load_specific_ships_file

TROPICAL_PHASES = ["TS", "HU", "TY", "ST", "TC"]
ET_PHASES = ["EX", "ET"]
SECONDS_PER_HOUR = 3600


def calculate_masked_storm_metrics(storm_data, land_interaction_threshold_km, time_index):
    """
    Compute per-timestep IKE plus tropical-phase-masked Vmax/Pmin/R34/R50/R64
    and robustly-anchored time_to_et_hrs, for a single storm.

    Returns
    -------
    pandas.DataFrame
        Indexed by time_index, with total_ike_tj (+ band components),
        vmax_kts/pmin_mb/r34_nm/r50_nm/r64_nm (NaN outside tropical
        phases), storm_dir_deg, storm_speed_kts, landfall_dist_km,
        had_landfall, storm_lat_deg/storm_lon_deg, and time_to_et_hrs
        (defined only for tropical-phase timesteps counting down to the
        last tropical observation before an EX/ET transition, NaN if the
        storm never transitions or its very first observation is already
        post-transition).
    """
    rmw_m = storm_data["usa_rmw"].fillna(0).values * M_PER_NM
    vmax_ms = storm_data["usa_wind"].fillna(0).values * KT_TO_MS
    mpres_mb = storm_data["usa_pres"].fillna(np.nan).values

    r34_m = storm_data["usa_r34"].fillna(0).values * M_PER_NM
    r50_m = storm_data["usa_r50"].fillna(0).values * M_PER_NM
    r64_m = storm_data["usa_r64"].fillna(0).values * M_PER_NM

    storm_dir_deg = storm_data["storm_dir"].fillna(np.nan).values
    storm_speed_kts = storm_data["storm_speed"].fillna(0).values
    landfall_km = storm_data["landfall"].fillna(9999).values
    storm_lat_deg = storm_data["lat"].fillna(np.nan).values
    storm_lon_deg = storm_data["lon"].fillna(np.nan).values

    try:
        status_np = np.vectorize(lambda x: x.decode("utf-8").strip())(storm_data["usa_status"].values)
    except Exception:
        status_np = np.array([str(s).strip() for s in storm_data["usa_status"].values])

    is_tropical = np.isin(status_np, TROPICAL_PHASES)
    num_timesteps = len(time_index)

    ike1826_list, ike2633_list, ikehur_list, total_ike_list = [], [], [], []

    for t in range(num_timesteps):
        rmw, mwind, status = rmw_m[t], vmax_ms[t], status_np[t]
        ike_in_quads_1826, ike_in_quads_2633, ike_in_quads_hur = [], [], []

        if status in TROPICAL_PHASES:
            if np.any(r34_m[t, :] > 0):
                for q in range(r34_m.shape[1]):
                    r34, r50 = r34_m[t, q], r50_m[t, q]
                    if r50 > 0:
                        # R50 reported: band is bounded by R34 and R50.
                        ike_m, ike_a = 20, 0.25 * np.pi * (r34**2 - r50**2)
                    elif r50 == 0 and mwind > 26 and r34 > rmw:
                        # R50 unreported, intensity above threshold:
                        # band closed at 0.75*RMW.
                        ike_m, ike_a = 20, 0.25 * np.pi * (r34**2 - (0.75 * rmw)**2)
                    elif r50 == 0 and mwind < 26 and r34 > rmw:
                        # R50 unreported, intensity below threshold: 
                        # modified mean wind, band closed at 0.75*RMW
                        ike_m, ike_a = 0.25 * mwind + 0.75 * 18, 0.25 * np.pi * (r34**2 - (0.75 * rmw)**2)
                    elif r50 == 0 and r34 <= rmw:
                        # RMW and R34 coincide/RMW outside R34
                        ike_m, ike_a = 18, 0.25 * np.pi * (r34**2 - (0.5 * r34)**2)
                    else:
                        ike_m, ike_a = 0, 0
                    ike_in_quads_1826.append((0.5 * ike_a * ike_m**2) / 1E12)

            # --- 50-64 kt band ------------------------------------------
            if np.any(r50_m[t, :] >= 0):
                for q in range(r50_m.shape[1]):
                    r50, r64 = r50_m[t, q], r64_m[t, q]
                    if r64 > 0:
                        # R64 reported: band is bounded by R50 and R64.
                        ike_m, ike_a = 27.75, 0.25 * np.pi * (r50**2 - r64**2)
                    elif r64 == 0 and mwind > 33 and r50 > rmw:
                        # R64 unreported, intensity above threshold:
                        # band closed at 0.75*RMW.
                        ike_m, ike_a = 27.75, 0.25 * np.pi * (r50**2 - (0.75 * rmw)**2)
                    elif r64 == 0 and mwind < 33 and r50 > rmw:
                        # R64 unreported, intensity below threshold: 
                        # modified mean wind, band closed at 0.75*RMW
                        ike_m, ike_a = 0.25 * mwind + 0.75 * 26, 0.25 * np.pi * (r50**2 - (0.75 * rmw)**2)
                    elif r64 == 0 and r50 <= rmw:
                        # RMW and R34 coincide/RMW outside R34
                        ike_m, ike_a = 26, 0.25 * np.pi * (r50**2 - (0.5 * r50)**2)
                    else:
                        ike_m, ike_a = 0, 0
                    ike_in_quads_2633.append((0.5 * ike_a * ike_m**2) / 1E12)

            # --- >64 kt band ----------------------------------------------
            if np.any(r64_m[t, :] >= 0):
                for q in range(r64_m.shape[1]):
                    r64 = r64_m[t, q]
                    if r64 > rmw:
                    # R64 outside RMW: band is bounded by R64 and 0.75*RMW    
                        ike_m, ike_a = 0.25 * mwind + 0.75 * 33, 0.25 * np.pi * (r64**2 - (0.75 * rmw)**2)
                    elif r64 == rmw:
                    # R64 equals RMW: band is bounded by R64 and 0.75*RMW/R64        
                        ike_m, ike_a = 0.25 * mwind + 0.75 * 33, 0.25 * np.pi * (r64**2 - (0.75 * r64)**2)
                    elif r64 < rmw:
                    # R64 inside RMW: band is bounded by R64 and 0.75*RMW/R64 
                    # higher mean wind   
                        ike_m, ike_a = 0.1 * mwind + 0.9 * 33, 0.25 * np.pi * (r64**2 - (0.75 * r64)**2)
                    else:
                        ike_m, ike_a = 0, 0
                    ike_in_quads_hur.append((0.5 * ike_a * ike_m**2) / 1E12)

        if status in TROPICAL_PHASES and np.nansum(ike_in_quads_1826) > 0:
            ike1826_list.append(np.nansum(ike_in_quads_1826))
            ike2633_list.append(np.nansum(ike_in_quads_2633))
            ikehur_list.append(np.nansum(ike_in_quads_hur))
            total_ike_list.append(
                np.nansum(ike_in_quads_1826) + np.nansum(ike_in_quads_2633) + np.nansum(ike_in_quads_hur)
            )
        else:
            ike1826_list.append(np.nan)
            ike2633_list.append(np.nan)
            ikehur_list.append(np.nan)
            total_ike_list.append(np.nan)

    # --- Robust ET anchoring: actual datetime delta, not assumed 6-hour spacing ---
    time_to_et_hrs = np.full(num_timesteps, np.nan)
    et_indices = np.where(np.isin(status_np, ET_PHASES))[0]

    if len(et_indices) > 0 and et_indices[0] > 0:
        first_et_index = et_indices[0]
        last_tropical = first_et_index - 1
        delta = (time_index[last_tropical] - time_index).total_seconds() / SECONDS_PER_HOUR
        valid = (delta >= 0) & is_tropical
        time_to_et_hrs[valid] = delta[valid]

    storm_had_landfall = bool(np.any(landfall_km <= land_interaction_threshold_km))

    # --- Strict tropical masking of base variables, so diff() math never
    # spans a tropical/extratropical boundary silently. ---
    vmax_kts_arr = storm_data["usa_wind"].values.astype(float).copy()
    pmin_arr = mpres_mb.astype(float).copy()
    r34_mean_arr = storm_data["usa_r34"].values.mean(axis=1).astype(float).copy()
    r50_mean_arr = storm_data["usa_r50"].values.mean(axis=1).astype(float).copy()
    r64_mean_arr = storm_data["usa_r64"].values.mean(axis=1).astype(float).copy()

    non_tropical = ~is_tropical
    vmax_kts_arr[non_tropical] = np.nan
    pmin_arr[non_tropical] = np.nan
    r34_mean_arr[non_tropical] = np.nan
    r50_mean_arr[non_tropical] = np.nan
    r64_mean_arr[non_tropical] = np.nan

    return pd.DataFrame(
        {
            "ike_18_26_list": ike1826_list, "ike_26_33_list": ike2633_list, "ike_hur_list": ikehur_list,
            "total_ike_tj": total_ike_list,
            "rmw_nm": storm_data["usa_rmw"].values,
            "vmax_kts": vmax_kts_arr,
            "pmin_mb": pmin_arr,
            "r34_nm": r34_mean_arr, "r50_nm": r50_mean_arr, "r64_nm": r64_mean_arr,
            "r34_km": r34_mean_arr * NM_TO_KM,
            "storm_dir_deg": storm_dir_deg,
            "storm_speed_kts": storm_speed_kts,
            "landfall_dist_km": landfall_km,
            "storm_lat_deg": storm_lat_deg, "storm_lon_deg": storm_lon_deg,
            "time_to_et_hrs": time_to_et_hrs,
            "had_landfall": storm_had_landfall,
        },
        index=time_index,
    )


def load_master_dataframe_masked(
    basins,
    ibtracs_file_map,
    basin_codes,
    start_year,
    end_year,
    land_interaction_threshold_km=0,
    verbose=True,
):
    """
    Build the master DataFrame for figures that need calculate_masked_storm_metrics'
    strict tropical-phase masking and robust ET anchoring, but do NOT need
    SHIPS shear data at all (unlike load_master_dataframe_shear_composite,
    which requires a SHIPS match and skips a basin entirely if missing).

    Genesis-basin-code filtering + calculate_masked_storm_metrics, then a
    plain dropna(subset=['vmax_kts']) + vmax_kts > 0 filter -- since
    vmax_kts is already NaN outside tropical phases (the masking), this
    implicitly restricts the frame to tropical-phase timesteps, same as
    load_master_dataframe_shear_composite.

    Returns
    -------
    pandas.DataFrame
    """
    all_storms = []

    if verbose:
        print(f"--- Loading Data ({start_year}-{end_year}) ---")

    for basin_name in basins:
        ibtracs_path = ibtracs_file_map.get(basin_name)
        expected_basin_code = basin_codes.get(basin_name)

        if not ibtracs_path or not os.path.exists(ibtracs_path) or not expected_basin_code:
            if verbose:
                print(f"    - {basin_name}: file not found, skipping.")
            continue

        genesis_filtered_count = 0

        with xr.open_dataset(ibtracs_path) as ds:
            season_mask = (ds["season"] >= start_year) & (ds["season"] <= end_year)
            data_filtered_by_year = ds.where(season_mask, drop=True)
            names_decoded = [
                name.decode("utf-8").strip() for name in data_filtered_by_year["name"].values
            ]
            num_storms = len(data_filtered_by_year["storm"])

            for i in range(num_storms):
                storm_data = data_filtered_by_year.isel(storm=i)
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

                first_point_index = valid_indices[0]
                raw_basin = storm_data["basin"].values[first_point_index]
                genesis_basin = (
                    raw_basin.decode("utf-8").strip()
                    if isinstance(raw_basin, (bytes, np.bytes_))
                    else str(raw_basin).strip()
                )
                if genesis_basin != expected_basin_code:
                    genesis_filtered_count += 1
                    continue

                storm_df = calculate_masked_storm_metrics(
                    storm_data, land_interaction_threshold_km, storm_datetimes
                )
                if storm_df.empty or storm_df["vmax_kts"].isnull().all():
                    continue

                storm_df["storm_name"] = names_decoded[i]
                storm_df["storm_year"] = int(storm_data.season.item())
                storm_df["basin"] = basin_name
                all_storms.append(storm_df)

        if verbose:
            print(f"    - {basin_name}: filtered out {genesis_filtered_count} non-genesis storms.")

    if not all_storms:
        return pd.DataFrame()

    master_df = pd.concat(all_storms)
    master_df = master_df.dropna(subset=["vmax_kts"])
    master_df = master_df[master_df["vmax_kts"] > 0]

    return master_df


def load_master_dataframe_shear_composite(
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
    Build the master DataFrame for the shear-split rate-of-change figure:
    per-basin IBTrACS + SHIPS loading (basin skipped entirely if SHIPS is
    missing, same as asymmetry_tendency.py), genesis-basin filtering,
    masked storm metrics (calculate_masked_storm_metrics), SHIPS shear-
    magnitude merge (direction not needed for this figure -- only
    magnitude, for Low/Mod/High categorization).

    The strict tropical masking in calculate_masked_storm_metrics means
    vmax_kts is already NaN on every non-tropical-phase timestep, so the
    final dropna(subset=['vmax_kts', 'shear_mag_ms']) below implicitly
    restricts the returned frame to tropical-phase timesteps with a valid
    shear match -- no separate non-tropical-phase filter is needed.

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
            data_filtered_by_year = ds.where(season_mask, drop=True)
            names_decoded = robust_decode(data_filtered_by_year["name"])
            if "usa_atcf_id" in data_filtered_by_year:
                atcf_ids_decoded = robust_decode(data_filtered_by_year["usa_atcf_id"])
            else:
                if verbose:
                    print("      WARNING: 'usa_atcf_id' not found; SHIPS lookup will rely on storm name only.")
                atcf_ids_decoded = [""] * len(data_filtered_by_year["storm"])
            num_storms = len(data_filtered_by_year["storm"])

            if verbose:
                print(f"Processing {num_storms} storms in {basin_name}...")
            genesis_filtered_count = 0

            for i in range(num_storms):
                storm_data = data_filtered_by_year.isel(storm=i)
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

                first_point_index = valid_indices[0]
                raw_basin = storm_data["basin"].values[first_point_index]
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

                storm_df = calculate_masked_storm_metrics(
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
                        reindexed_ships = ships_subset.reindex(
                            storm_datetimes, method="nearest", tolerance=pd.Timedelta("12 hours")
                        )
                        storm_df["shear_mag_ms"] = reindexed_ships["shrmag_vr"].values * KT_TO_MS
                    except Exception:
                        storm_df["shear_mag_ms"] = np.nan
                else:
                    storm_df["shear_mag_ms"] = np.nan

                storm_df["storm_name"] = storm_name
                storm_df["storm_year"] = storm_year
                storm_df["basin"] = basin_name
                all_storms.append(storm_df)

            if verbose:
                print(f"        - Filtered out {genesis_filtered_count} crossover/external storms "
                      f"based on official genesis basin label.")

    if not all_storms:
        return pd.DataFrame()

    master_df = pd.concat(all_storms)
    master_df.dropna(subset=["vmax_kts", "shear_mag_ms"], inplace=True)
    master_df = master_df[master_df["vmax_kts"] > 0]

    return master_df


def add_shear_categories(master_df):
    """
    Add a 'shear_cat' column (Low <5 m/s, Mod 5-10 m/s, High >10 m/s),
    dropping any row that doesn't fall in one of those three bands.
    """
    conditions = [
        (master_df["shear_mag_ms"] < 5),
        (master_df["shear_mag_ms"] >= 5) & (master_df["shear_mag_ms"] <= 10),
        (master_df["shear_mag_ms"] > 10),
    ]
    choices = ["Low (<5 m/s)", "Mod (5-10 m/s)", "High (>10 m/s)"]

    master_df = master_df.copy()
    master_df["shear_cat"] = np.select(conditions, choices, default="MISSING")
    return master_df[master_df["shear_cat"] != "MISSING"]


def compute_rate_of_change_plot_df(df, var_name, col_name, diff_periods=1, pre_et_window_hrs=48):
    """
    Build the per-timestep rate-of-change table consumed by the box plot:
    diffs col_name within each storm, classifies each timestep into a
    Pre-ET/Early-Tropical phase, and restricts to storms with a defined
    time_to_et_hrs -- i.e. storms that actually underwent ET later in
    their lifecycle. Non-transitioning storms are excluded entirely from
    this figure, not just left with a blank phase label, per the source
    methodology.

    Parameters
    ----------
    df : pandas.DataFrame
        Output of add_shear_categories(load_master_dataframe_shear_composite(...)).
    var_name : str
        Short label used to build the diff column's name, e.g. "IKE".
    col_name : str
        Actual column to diff, e.g. "total_ike_tj".
    diff_periods : int, default 1
        Number of synoptic (6-hourly) steps between comparison points.
    pre_et_window_hrs : float, default 48
        Timesteps with time_to_et_hrs <= this are labeled "Pre-ET";
        everything else (that still has a defined time_to_et_hrs) is
        "Early Tropical".

    Returns
    -------
    plot_df : pandas.DataFrame
        Columns: basin, phase (ordered categorical), shear_cat (ordered
        categorical), and the diff column.
    diff_col_name : str
    """
    diff_hours = diff_periods * 6
    dfc = df.copy()
    diff_col_name = f"d{var_name}_{diff_hours}hr"

    dfc[diff_col_name] = dfc.groupby(["basin", "storm_year", "storm_name"])[col_name].diff(periods=diff_periods)
    dfc.dropna(subset=[diff_col_name, "shear_cat"], inplace=True)
    dfc = dfc[np.isfinite(dfc[diff_col_name])]

    dfc = dfc[dfc["time_to_et_hrs"].notna()].copy()

    is_pre_et = dfc["time_to_et_hrs"] <= pre_et_window_hrs
    conditions = [is_pre_et]
    choices = [f"Pre-ET\n(<{pre_et_window_hrs}hr)"]
    default_choice = f"Early Tropical\n(>{pre_et_window_hrs}hr to ET)"
    dfc["phase"] = np.select(conditions, choices, default=default_choice)

    plot_df = dfc[["basin", "phase", "shear_cat", diff_col_name]].copy()
    plot_df[diff_col_name] = pd.to_numeric(plot_df[diff_col_name], errors="coerce")
    plot_df.dropna(subset=[diff_col_name], inplace=True)

    phase_order = [f"Early Tropical\n(>{pre_et_window_hrs}hr to ET)", f"Pre-ET\n(<{pre_et_window_hrs}hr)"]
    shear_order = ["Low (<5 m/s)", "Mod (5-10 m/s)", "High (>10 m/s)"]
    plot_df["phase"] = pd.Categorical(plot_df["phase"], categories=phase_order, ordered=True)
    plot_df["shear_cat"] = pd.Categorical(plot_df["shear_cat"], categories=shear_order, ordered=True)

    return plot_df, diff_col_name
