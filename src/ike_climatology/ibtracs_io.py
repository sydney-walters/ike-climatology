"""
ibtracs_io.py

Loads and concatenates per-timestep IKE/storm metrics across all basins
into a single "master" DataFrame, using the shared preprocessing pipeline
(prepare_ibtracs_storm, calculate_storm_metrics, passes_quality_filters).

Usage
-----
    from ike_climatology import config
    from ike_climatology.ibtracs_io import load_master_dataframe

    master_df = load_master_dataframe(
        basins=config.BASINS_TO_ANALYZE,
        file_map=config.IBTRACS_FILE_MAP,
        basin_codes=config.IBTRACS_BASIN_CODES,
        start_year=config.START_YEAR,
        end_year=config.END_YEAR,
        land_interaction_threshold_km=config.LAND_INTERACTION_THRESHOLD_KM,
    )
"""

import os

import numpy as np
import pandas as pd
import xarray as xr

from .ibtracs_preprocessing import prepare_ibtracs_storm, passes_quality_filters, MIN_VMAX_KTS
from .storm_metrics import calculate_storm_metrics, TROPICAL_PHASES


def load_master_dataframe(
    basins,
    file_map,
    basin_codes,
    start_year,
    end_year,
    land_interaction_threshold_km,
    apply_quality_filters=True,
    drop_non_tropical_phases=False,
    verbose=True,
):
    """
    Build a single per-timestep DataFrame across all basins and storms in
    [start_year, end_year], with basin/storm_year/storm_name/storm_id
    columns attached for downstream groupby aggregation.

    Parameters
    ----------
    basins : list of str
        Basin display names to process, e.g. config.BASINS_TO_ANALYZE.
    file_map : dict
        basin name -> IBTrACS NetCDF path, e.g. config.IBTRACS_FILE_MAP.
    basin_codes : dict
        basin name -> two-letter IBTrACS basin code, e.g.
        config.IBTRACS_BASIN_CODES.
    start_year, end_year : int
        Inclusive season range to retain (filtered on IBTrACS's own
        'season' variable).
    land_interaction_threshold_km : float
        Passed through to calculate_storm_metrics.
    apply_quality_filters : bool, default True
        If True (the interannual and SSHWS figures' behavior), each storm
        must pass passes_quality_filters (min 34 kt, must have positive
        IKE) to be retained. If False (the global spatial climatology
        figure's behavior), only the minimal check that calculate_storm_metrics
        returned a non-empty frame with at least one non-null vmax_kts is
        applied -- sub-tropical-storm-strength systems are kept, useful
        for track-density counting where every observed storm should
        contribute regardless of peak intensity.
    drop_non_tropical_phases : bool, default False
        If True, drop rows whose usa_status is not in TROPICAL_PHASES (e.g.
        extratropical 'EX', subtropical 'SD'/'SS', disturbance 'DB') from
        the returned master_df, in addition to whatever apply_quality_filters
        already does. calculate_storm_metrics already forces total_ike_tj
        to NaN for these rows regardless of this flag, so any step that
        drops NaN IKE (e.g. the global climatology notebook's
        dropna(subset=['total_ike_tj']) before plotting) excludes them from
        IKE-based results either way -- this flag only controls whether
        master_df itself carries those rows.
        Default False because get_interannual_ike_metrics detects landfall
        onset from landfall_dist_km across ALL rows in master_df, including
        extratropical-phase ones: some storms make landfall only after
        transitioning extratropical, and dropping those rows would silently
        change which timestep counts as landfall for annual_avg_landfall_ike
        / has_landfall. Only pass True for a use case (like the global
        climatology figure) that doesn't depend on post-tropical-phase rows.
    verbose : bool, default True
        Print basin-by-basin progress.

    Returns
    -------
    pandas.DataFrame
        Concatenated per-timestep storm records, with vmax_kts NaN/zero
        rows already dropped. If drop_non_tropical_phases=True, rows whose
        usa_status is not in TROPICAL_PHASES are also dropped. Columns
        include everything returned by calculate_storm_metrics plus
        'basin', 'storm_year', 'storm_name', and 'storm_id'.
    """
    all_storms = []

    if verbose:
        print("--- Starting Initial Data Loading and Processing for All Basins ---")

    for basin_name in basins:
        if verbose:
            print(f"    - Loading IBTrACS data for {basin_name} ({start_year}-{end_year})")

        ibtracs_path = file_map.get(basin_name)
        expected_basin_code = basin_codes.get(basin_name)

        if not ibtracs_path or not os.path.exists(ibtracs_path) or not expected_basin_code:
            if verbose:
                print(f"      > Skipping {basin_name}: file not found.")
            continue

        with xr.open_dataset(ibtracs_path) as ds:
            season_mask = (ds["season"] >= start_year) & (ds["season"] <= end_year)
            data_filtered_by_year = ds.where(season_mask, drop=True)
            names_decoded = [
                name.decode("utf-8").strip() for name in data_filtered_by_year["name"].values
            ]
            num_storms_in_file = len(data_filtered_by_year["storm"])
            if verbose:
                print(f"        Found {num_storms_in_file} storms in file.")

            genesis_filtered_count = 0

            for i in range(num_storms_in_file):
                storm_data = data_filtered_by_year.isel(storm=i)

                prep = prepare_ibtracs_storm(storm_data, expected_basin_code=expected_basin_code)
                if prep.rejected:
                    if prep.status == "genesis_basin_mismatch":
                        genesis_filtered_count += 1
                    continue

                storm_df = calculate_storm_metrics(
                    prep.storm_data, land_interaction_threshold_km, prep.storm_datetimes
                )

                if apply_quality_filters:
                    if not passes_quality_filters(storm_df):
                        continue
                else:
                    if storm_df.empty or storm_df["vmax_kts"].isnull().all():
                        continue

                # storm_year comes from IBTrACS's own 'season' label -- the
                # same variable season_mask filters on above -- not from the
                # observation timestamps. For Southern Hemisphere basins
                # (South Pacific, South Indian), a storm with season ==
                # start_year can have its earliest synoptic observations
                # timestamped in Nov/Dec of the prior calendar year; deriving
                # storm_year from the timestamps instead would let those
                # early rows in under the wrong year.
                storm_year = int(storm_data.season.item())
                storm_name = names_decoded[i]

                storm_df["storm_name"] = storm_name
                storm_df["storm_year"] = storm_year
                storm_df["basin"] = basin_name
                storm_df["storm_id"] = f"{storm_year}_{basin_name}_{i:03d}"
                all_storms.append(storm_df)

        if verbose:
            print(
                f"        - Filtered out {genesis_filtered_count} crossover/external "
                f"storms based on official genesis basin label."
            )

    if not all_storms:
        return pd.DataFrame()

    master_df = pd.concat(all_storms)
    master_df.dropna(subset=["vmax_kts"], inplace=True)
    master_df = master_df[master_df["vmax_kts"] > 0]

    if drop_non_tropical_phases:
        master_df = master_df[master_df["usa_status"].isin(TROPICAL_PHASES)]

    return master_df


def load_storm_tracks(
    basins,
    file_map,
    basin_codes,
    start_year,
    end_year,
    land_interaction_threshold_km,
    min_vmax_kts=MIN_VMAX_KTS,
    verbose=True,
):
    """
    Build per-storm track records (lon/lat arrays) for storms passing the
    standard quality pipeline, plus a rejection-reason breakdown per basin
    (for methods-section accounting, e.g. an "n = 1,705" total). Applies
    the same overall filtering as load_master_dataframe's default
    (apply_quality_filters=True) -- temporal/synoptic-hour filtering,
    genesis-basin-code match, Vmax >= min_vmax_kts, and at least one
    timestep of positive IKE -- but tracks WHY a storm was rejected
    (crossover / below intensity threshold / no valid IKE / other) rather
    than just pass/fail, and returns each surviving storm's full lon/lat
    track rather than a per-timestep table.

    The IKE check reuses calculate_storm_metrics directly (the same
    canonical IKE calculation used everywhere else in this package,
    including its 34-50 kt band fix -- see README, "A recurring bug across
    pipeline copies") rather than a separate hand-rolled "would this storm
    produce valid IKE" check -- avoiding a second, potentially-diverging
    copy of that logic.

    Note on rejection order: intensity (Vmax) is checked BEFORE the
    genesis-basin match, matching the original script's counting
    convention -- a storm below min_vmax_kts is counted there even if it
    would also have failed the genesis-basin check.

    Parameters
    ----------
    basins : list of str
    file_map : dict
        basin name -> IBTrACS NetCDF path.
    basin_codes : dict
        basin name -> two-letter IBTrACS basin code.
    start_year, end_year : int
        Inclusive season range (filtered on IBTrACS's 'season' variable).
    land_interaction_threshold_km : float
        Passed through to calculate_storm_metrics.
    min_vmax_kts : float, default ibtracs_preprocessing.MIN_VMAX_KTS (34)
        Minimum peak Vmax, in knots, for a storm to be retained.
    verbose : bool, default True
        Print basin-by-basin progress and the rejection breakdown.

    Returns
    -------
    tracks_df : pandas.DataFrame
        One row per accepted storm: basin, storm_year, storm_name,
        storm_id, lons (array), lats (array).
    rejection_counts : dict
        basin name -> {'crossover': int, 'below_intensity': int,
        'no_ike': int, 'other': int}.
    """
    track_records = []
    rejection_counts = {}
    total_accepted = 0

    if verbose:
        print(f"--- Plotting Global TC Tracks ({start_year}-{end_year}) ---")

    for basin_name in basins:
        ibtracs_path = file_map.get(basin_name)
        expected_basin_code = basin_codes.get(basin_name)

        if not ibtracs_path or not os.path.exists(ibtracs_path) or not expected_basin_code:
            if verbose:
                print(f"      > Skipping {basin_name}: file not found.")
            continue

        if verbose:
            print(f"  Processing {basin_name}...")

        counts = {"crossover": 0, "below_intensity": 0, "no_ike": 0, "other": 0}
        basin_accepted = 0

        with xr.open_dataset(ibtracs_path) as ds:
            season_mask = (ds["season"] >= start_year) & (ds["season"] <= end_year)
            data_filtered_by_year = ds.where(season_mask, drop=True)
            names_decoded = [
                name.decode("utf-8").strip() for name in data_filtered_by_year["name"].values
            ]
            num_storms_in_file = len(data_filtered_by_year["storm"])

            for i in range(num_storms_in_file):
                storm_data = data_filtered_by_year.isel(storm=i)

                # Temporal/synoptic-hour filtering only -- genesis-basin
                # match is done manually below, after the intensity check,
                # to match this figure's rejection-counting order.
                prep = prepare_ibtracs_storm(storm_data, expected_basin_code=None)
                if prep.rejected:
                    counts["other"] += 1
                    continue

                valid_pos = np.where(
                    ~np.isnan(prep.storm_data.lat) & ~np.isnan(prep.storm_data.lon)
                )[0]
                if len(valid_pos) == 0:
                    counts["other"] += 1
                    continue

                vmax_vals = prep.storm_data["usa_wind"].values
                vmax_max = np.nanmax(vmax_vals) if not np.all(np.isnan(vmax_vals)) else 0.0
                if vmax_max < min_vmax_kts:
                    counts["below_intensity"] += 1
                    continue

                first_idx = valid_pos[0]
                raw_basin = prep.storm_data["basin"].values[first_idx]
                genesis_basin = (
                    raw_basin.decode("utf-8").strip()
                    if isinstance(raw_basin, (bytes, np.bytes_))
                    else str(raw_basin).strip()
                )
                if genesis_basin != expected_basin_code:
                    counts["crossover"] += 1
                    continue

                storm_df = calculate_storm_metrics(
                    prep.storm_data, land_interaction_threshold_km, prep.storm_datetimes
                )
                if storm_df.empty or storm_df["total_ike_tj"].fillna(0).max() <= 0:
                    counts["no_ike"] += 1
                    continue

                storm_year = int(storm_data.season.item())
                storm_name = names_decoded[i]

                track_records.append({
                    "basin": basin_name,
                    "storm_year": storm_year,
                    "storm_name": storm_name,
                    "storm_id": f"{storm_year}_{basin_name}_{i:03d}",
                    "lons": prep.storm_data["lon"].values[valid_pos],
                    "lats": prep.storm_data["lat"].values[valid_pos],
                })
                basin_accepted += 1

        rejection_counts[basin_name] = counts
        total_accepted += basin_accepted

        if verbose:
            print(
                f"    Plotted {basin_accepted} storms  "
                f"(rejected: {counts['crossover']} crossover, "
                f"{counts['below_intensity']} below-intensity, "
                f"{counts['no_ike']} no valid IKE, "
                f"{counts['other']} other)"
            )

    tracks_df = pd.DataFrame(track_records)

    if verbose:
        print(f"\n  Total storms plotted: {total_accepted:,}")

    return tracks_df, rejection_counts
