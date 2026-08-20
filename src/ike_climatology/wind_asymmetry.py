"""
wind_asymmetry.py

Quadrant-relative decomposition of IKE and wind radii (R34/R50/R64), in
three reference frames:
  - geographic (N/E/S/W quadrants, fixed to compass direction)
  - shear-relative (front-right/rear-right/rear-left/front-left, rotated
    into the deep-layer shear vector's frame, from SHIPS)
  - motion-relative (RF/RR/LR/LF, rotated into the storm's own heading)

This is a genuinely different computation from storm_metrics.calculate_storm_metrics:
that function collapses each IKE band to a single scalar per timestep;
this one keeps all four quadrants separate (needed for the asymmetry
figures) and merges in SHIPS shear direction/magnitude, which
calculate_storm_metrics doesn't use at all. 

Usage
-----
    from ike_climatology import config
    from ike_climatology.wind_asymmetry import (
        load_master_dataframe_quadrant,
        compute_basin_asymmetry_results,
        compute_global_scale_limits,
    )

    master_df = load_master_dataframe_quadrant(
        basins=config.BASINS_TO_ANALYZE,
        ibtracs_file_map=config.IBTRACS_FILE_MAP,
        ships_file_map=config.SHIPS_FILE_MAP,
        basin_codes=config.IBTRACS_BASIN_CODES,
        ships_base_path=config.SHIPS_BASE_PATH,
        start_year=2020, end_year=2024,
        intensity_threshold_kt=64,
    )
    basin_results, sample_size_records = compute_basin_asymmetry_results(master_df)
    global_max_ike, global_max_r34 = compute_global_scale_limits(basin_results)
"""

import os

import numpy as np
import pandas as pd
import xarray as xr

from .storm_metrics import M_PER_NM, KT_TO_MS, NM_TO_KM

TROPICAL_PHASES = ["TS", "HU", "TY", "ST", "TC"]

BASIN_ABBREV_MAP = {
    "North Atlantic": "NA", "East Pacific": "EP", "West Pacific": "WP",
    "North Indian": "NI", "South Indian": "SI", "South Pacific": "SP",
}


def robust_decode(data_array):
    """Decode an IBTrACS byte-character array (e.g. usa_atcf_id) to strings."""
    decoded_list = []
    vals = data_array.values
    for row in vals:
        try:
            decoded_list.append(b"".join(row).decode("utf-8").strip())
        except (TypeError, AttributeError):
            valid_chars = [chr(c) for c in row if 32 <= c <= 126]
            decoded_list.append("".join(valid_chars).strip())
    return decoded_list


def load_specific_ships_file(filepath):
    """
    Load one SHIPS NetCDF file into a DataFrame indexed by (time, name_short),
    with deep-layer shear direction (shrdir_vr) and magnitude (shrmag_vr).
    Returns None if the file is missing or can't be parsed.
    """
    if not os.path.exists(filepath):
        print(f"    WARNING: SHIPS file not found: {filepath}")
        return None

    try:
        with xr.open_dataset(filepath) as ships_ds:
            df_vars = ["year", "month", "day", "hour", "tc_id", "shrdir_vr", "shrmag_vr"]
            ships_df = ships_ds[df_vars].to_dataframe()
            ships_df["time"] = pd.to_datetime(ships_df[["year", "month", "day", "hour"]])

            char_array = ships_ds["tc_id"].values
            name_strings = [b"".join(row).decode("utf-8", "replace").strip() for row in char_array.T]

            if len(name_strings) > 0 and len(ships_df) % len(name_strings) == 0:
                n_repeats = len(ships_df) // len(name_strings)
                ships_df["name_short"] = np.repeat(name_strings, n_repeats)
            else:
                return None

            ships_df = ships_df.dropna(subset=["shrdir_vr", "shrmag_vr"])
            ships_df = (
                ships_df.drop_duplicates(subset=["time", "name_short"])
                .set_index(["time", "name_short"])
                .sort_index()
            )
        return ships_df
    except Exception as e:
        print(f"    Error loading SHIPS file {filepath}: {e}")
        return None


def calculate_vector_relative_rotation(geo_quads, vector_direction):
    """
    Rotate geographic-quadrant array (N/E/S/W-ordered) into a
    frame relative to vector_direction (e.g. shear or storm-motion
    heading, degrees), returning [Front_Right, Rear_Right, Rear_Left,
    Front_Left] by area-weighted overlap between the geographic and
    rotated quadrant boundaries.
    """
    if np.isnan(vector_direction) or np.all(np.isnan(geo_quads)):
        return [np.nan] * 4

    geo_bounds = np.array([[0, 90], [90, 180], [180, 270], [270, 360]])
    geo_densities = np.array(geo_quads) / 90.0
    v = vector_direction % 360

    relative_bounds = {
        "Front_Right": [v, v + 90],
        "Rear_Right": [v + 90, v + 180],
        "Rear_Left": [v + 180, v + 270],
        "Front_Left": [v + 270, v + 360],
    }

    def get_overlap(interval1, interval2):
        start1, end1 = interval1[0] % 360, interval1[1] % 360
        start2, end2 = interval2[0] % 360, interval2[1] % 360

        def get_intervals(start, end):
            return [(start, end)] if start < end else [(start, 360), (0, end)]

        total = 0
        for s1, e1 in get_intervals(start1, end1):
            for s2, e2 in get_intervals(start2, end2):
                total += max(0, min(e1, e2) - max(s1, s2))
        return total

    ordered_keys = ["Front_Right", "Rear_Right", "Rear_Left", "Front_Left"]
    return [
        sum(geo_densities[i] * get_overlap(relative_bounds[key], geo_bounds[i]) for i in range(4))
        for key in ordered_keys
    ]


def safe_mean(df, col_name):
    """Column-wise nanmean of a column holding length-4 list-per-row values."""
    if df.empty:
        return [np.nan] * 4
    return np.nanmean(np.vstack(df[col_name].values).astype(float), axis=0)


def calculate_quadrant_ike_metrics(storm_data, time_index, ships_df, storm_name, atcf_id):
    """
    Per-storm, per-timestep IKE and wind-radii decomposition, kept as
    per-quadrant (length-4) values in three reference frames: geographic
    (_geo), shear-relative (_sr, requires a SHIPS shear direction match),
    and motion-relative (_mr, from the storm's own heading).

    Parameters
    ----------
    storm_data : xarray.Dataset
        Pre-filtered (synoptic-hour, genesis-basin) per-storm IBTrACS slice.
    time_index : pandas.DatetimeIndex
    ships_df : pandas.DataFrame or None
        Output of load_specific_ships_file, or None if unavailable for
        this basin -- shear-relative fields will be all-NaN in that case.
    storm_name, atcf_id : str
        Used to look up this storm in ships_df (tried in that order).

    Returns
    -------
    pandas.DataFrame
        Indexed by time_index, with total_ike_tj, s_dir, m_dir, vmax_kts,
        and 18 quadrant columns ({ike1826,ike2633,ikehur,r34_nm,r50_nm,r64_nm}
        x {geo,sr,mr}_quad), each holding a length-4 list per row.
    """
    rmw_m = storm_data["usa_rmw"].fillna(0).values * M_PER_NM
    vmax_ms = storm_data["usa_wind"].fillna(0).values * KT_TO_MS

    r34_m = storm_data["usa_r34"].fillna(0).values * M_PER_NM
    r50_m = storm_data["usa_r50"].fillna(0).values * M_PER_NM
    r64_m = storm_data["usa_r64"].fillna(0).values * M_PER_NM

    storm_dir_deg = storm_data["storm_dir"].fillna(np.nan).values

    try:
        status_np = np.vectorize(lambda x: x.decode("utf-8").strip())(storm_data["usa_status"].values)
    except Exception:
        status_np = np.array([str(s).strip() for s in storm_data["usa_status"].values])

    ships_subset = None
    if ships_df is not None:
        for candidate in (str(storm_name)[:4].upper(), str(atcf_id)[:4].upper()):
            try:
                ships_subset = ships_df.xs(candidate, level="name_short")
                break
            except KeyError:
                continue

    if ships_subset is not None:
        try:
            reindexed_ships = ships_subset.reindex(
                time_index, method="nearest", tolerance=pd.Timedelta("12 hours")
            )
            shear_dirs = reindexed_ships["shrdir_vr"].values
            shear_mags = reindexed_ships["shrmag_vr"].values
        except Exception:
            shear_dirs = np.full(len(time_index), np.nan)
            shear_mags = np.full(len(time_index), np.nan)
    else:
        shear_dirs = np.full(len(time_index), np.nan)
        shear_mags = np.full(len(time_index), np.nan)

    num_timesteps = len(time_index)

    ike1826_geo_list, ike2633_geo_list, ikehur_geo_list = [], [], []
    r34_nm_geo_list, r50_nm_geo_list, r64_nm_geo_list = [], [], []

    ike1826_sr_list, ike2633_sr_list, ikehur_sr_list = [], [], []
    r34_nm_sr_list, r50_nm_sr_list, r64_nm_sr_list = [], [], []

    ike1826_mr_list, ike2633_mr_list, ikehur_mr_list = [], [], []
    r34_nm_mr_list, r50_nm_mr_list, r64_nm_mr_list = [], [], []

    total_ike_list = []
    s_dir_valid_list = []
    m_dir_valid_list = []

    for t in range(num_timesteps):
        rmw = rmw_m[t]
        mwind = vmax_ms[t]
        status = status_np[t]
        s_dir = shear_dirs[t]
        m_dir = storm_dir_deg[t]

        ike_in_quads_1826, ike_in_quads_2633, ike_in_quads_hur = [0] * 4, [0] * 4, [0] * 4
        r34_raw = storm_data["usa_r34"].fillna(0).values[t, :] * NM_TO_KM
        r50_raw = storm_data["usa_r50"].fillna(0).values[t, :] * NM_TO_KM
        r64_raw = storm_data["usa_r64"].fillna(0).values[t, :] * NM_TO_KM

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
                    ike_in_quads_1826[q] = (0.5 * ike_a * ike_m**2) / 1e12

            if np.any(r50_m[t, :] >= 0):
                for q in range(r50_m.shape[1]):
                    r50, r64 = r50_m[t, q], r64_m[t, q]
                    if r64 > 0:
                        ike_m, ike_a = 27.75, 0.25 * np.pi * (r50**2 - r64**2)
                    elif r64 == 0 and mwind > 33 and r50 > rmw:
                        ike_m, ike_a = 27.75, 0.25 * np.pi * (r50**2 - (0.75 * rmw) ** 2)
                    elif r64 == 0 and mwind < 33 and r50 > rmw:
                        ike_m, ike_a = 0.25 * mwind + 0.75 * 26, 0.25 * np.pi * (r50**2 - (0.75 * rmw) ** 2)
                    elif r64 == 0 and r50 <= rmw:
                        ike_m, ike_a = 26, 0.25 * np.pi * (r50**2 - (0.5 * r50) ** 2)
                    else:
                        ike_m, ike_a = 0, 0
                    ike_in_quads_2633[q] = (0.5 * ike_a * ike_m**2) / 1e12

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
                    ike_in_quads_hur[q] = (0.5 * ike_a * ike_m**2) / 1e12

            total_t_ike = np.nansum(ike_in_quads_1826) + np.nansum(ike_in_quads_2633) + np.nansum(ike_in_quads_hur)

            if total_t_ike > 0:
                total_ike_list.append(total_t_ike)
                s_dir_valid_list.append(s_dir)
                m_dir_valid_list.append(m_dir)

                ike1826_geo_list.append(ike_in_quads_1826)
                ike2633_geo_list.append(ike_in_quads_2633)
                ikehur_geo_list.append(ike_in_quads_hur)
                r34_nm_geo_list.append(r34_raw)
                r50_nm_geo_list.append(r50_raw)
                r64_nm_geo_list.append(r64_raw)

                if not np.isnan(s_dir):
                    ike1826_sr_list.append(calculate_vector_relative_rotation(ike_in_quads_1826, s_dir))
                    ike2633_sr_list.append(calculate_vector_relative_rotation(ike_in_quads_2633, s_dir))
                    ikehur_sr_list.append(calculate_vector_relative_rotation(ike_in_quads_hur, s_dir))
                    r34_nm_sr_list.append(calculate_vector_relative_rotation(r34_raw, s_dir))
                    r50_nm_sr_list.append(calculate_vector_relative_rotation(r50_raw, s_dir))
                    r64_nm_sr_list.append(calculate_vector_relative_rotation(r64_raw, s_dir))
                else:
                    ike1826_sr_list.append([np.nan] * 4)
                    ike2633_sr_list.append([np.nan] * 4)
                    ikehur_sr_list.append([np.nan] * 4)
                    r34_nm_sr_list.append([np.nan] * 4)
                    r50_nm_sr_list.append([np.nan] * 4)
                    r64_nm_sr_list.append([np.nan] * 4)

                if not np.isnan(m_dir):
                    ike1826_mr_list.append(calculate_vector_relative_rotation(ike_in_quads_1826, m_dir))
                    ike2633_mr_list.append(calculate_vector_relative_rotation(ike_in_quads_2633, m_dir))
                    ikehur_mr_list.append(calculate_vector_relative_rotation(ike_in_quads_hur, m_dir))
                    r34_nm_mr_list.append(calculate_vector_relative_rotation(r34_raw, m_dir))
                    r50_nm_mr_list.append(calculate_vector_relative_rotation(r50_raw, m_dir))
                    r64_nm_mr_list.append(calculate_vector_relative_rotation(r64_raw, m_dir))
                else:
                    ike1826_mr_list.append([np.nan] * 4)
                    ike2633_mr_list.append([np.nan] * 4)
                    ikehur_mr_list.append([np.nan] * 4)
                    r34_nm_mr_list.append([np.nan] * 4)
                    r50_nm_mr_list.append([np.nan] * 4)
                    r64_nm_mr_list.append([np.nan] * 4)

                continue

        # Non-tropical, or zero total IKE this timestep.
        total_ike_list.append(0)
        s_dir_valid_list.append(np.nan)
        m_dir_valid_list.append(np.nan)

        ike1826_geo_list.append([np.nan] * 4)
        ike2633_geo_list.append([np.nan] * 4)
        ikehur_geo_list.append([np.nan] * 4)
        r34_nm_geo_list.append([np.nan] * 4)
        r50_nm_geo_list.append([np.nan] * 4)
        r64_nm_geo_list.append([np.nan] * 4)

        ike1826_sr_list.append([np.nan] * 4)
        ike2633_sr_list.append([np.nan] * 4)
        ikehur_sr_list.append([np.nan] * 4)
        r34_nm_sr_list.append([np.nan] * 4)
        r50_nm_sr_list.append([np.nan] * 4)
        r64_nm_sr_list.append([np.nan] * 4)

        ike1826_mr_list.append([np.nan] * 4)
        ike2633_mr_list.append([np.nan] * 4)
        ikehur_mr_list.append([np.nan] * 4)
        r34_nm_mr_list.append([np.nan] * 4)
        r50_nm_mr_list.append([np.nan] * 4)
        r64_nm_mr_list.append([np.nan] * 4)

    return pd.DataFrame(
        {
            "total_ike_tj": total_ike_list, "s_dir": s_dir_valid_list, "m_dir": m_dir_valid_list,
            "ike1826_geo_quad": ike1826_geo_list, "ike2633_geo_quad": ike2633_geo_list, "ikehur_geo_quad": ikehur_geo_list,
            "r34_nm_geo_quad": r34_nm_geo_list, "r50_nm_geo_quad": r50_nm_geo_list, "r64_nm_geo_quad": r64_nm_geo_list,
            "ike1826_sr_quad": ike1826_sr_list, "ike2633_sr_quad": ike2633_sr_list, "ikehur_sr_quad": ikehur_sr_list,
            "r34_nm_sr_quad": r34_nm_sr_list, "r50_nm_sr_quad": r50_nm_sr_list, "r64_nm_sr_quad": r64_nm_sr_list,
            "ike1826_mr_quad": ike1826_mr_list, "ike2633_mr_quad": ike2633_mr_list, "ikehur_mr_quad": ikehur_mr_list,
            "r34_nm_mr_quad": r34_nm_mr_list, "r50_nm_mr_quad": r50_nm_mr_list, "r64_nm_mr_quad": r64_nm_mr_list,
            "vmax_kts": storm_data["usa_wind"].values,
        },
        index=time_index,
    )


def load_master_dataframe_quadrant(
    basins,
    ibtracs_file_map,
    ships_file_map,
    basin_codes,
    ships_base_path,
    start_year,
    end_year,
    intensity_threshold_kt=64,
    verbose=True,
):
    """
    Build the per-timestep master DataFrame for the wind-asymmetry figures:
    per-basin IBTrACS + SHIPS loading, genesis-basin-code filtering,
    quadrant IKE/radii decomposition (calculate_quadrant_ike_metrics), then
    filtered to vmax_kts >= intensity_threshold_kt and total_ike_tj > 0.

    Parameters
    ----------
    basins : list of str
    ibtracs_file_map : dict
        basin name -> IBTrACS NetCDF path, e.g. config.IBTRACS_FILE_MAP.
    ships_file_map : dict
        basin name -> SHIPS NetCDF filename (relative to ships_base_path),
        e.g. config.SHIPS_FILE_MAP.
    basin_codes : dict
        basin name -> two-letter IBTrACS basin code.
    ships_base_path : str
        Directory containing the SHIPS files.
    start_year, end_year : int
    intensity_threshold_kt : float, default 64
        Minimum vmax_kts to retain a timestep (64 = hurricane force,
        96 = major/Cat 3+).
    verbose : bool, default True

    Returns
    -------
    pandas.DataFrame
    """
    all_storms = []

    if verbose:
        print("--- Starting Data Loading ---")

    for basin_name in basins:
        if verbose:
            print(f"    - Loading IBTrACS & SHIPS Data for {basin_name}...")

        ibtracs_path = ibtracs_file_map.get(basin_name)
        expected_basin_code = basin_codes.get(basin_name)
        ships_filename = ships_file_map.get(basin_name)

        if not ibtracs_path or not os.path.exists(ibtracs_path) or not expected_basin_code:
            if verbose:
                print(f"      Missing IBTrACS for {basin_name}, skipping.")
            continue

        ships_df = None
        if ships_filename:
            ships_df = load_specific_ships_file(os.path.join(ships_base_path, ships_filename))

        with xr.open_dataset(ibtracs_path) as ds:
            season_mask = (ds["season"] >= start_year) & (ds["season"] <= end_year)
            data_filtered = ds.where(season_mask, drop=True)
            names_decoded = robust_decode(data_filtered["name"])
            if "usa_atcf_id" in data_filtered:
                atcf_ids_decoded = robust_decode(data_filtered["usa_atcf_id"])
            else:
                # usa_atcf_id is expected in standard IBTrACS v04r01 files,
                # but atcf_id is only a fallback SHIPS lookup key (tried
                # after storm_name) -- don't hard-fail the whole basin if
                # it's absent in some file variant.
                if verbose:
                    print("      WARNING: 'usa_atcf_id' not found in this file; "
                          "SHIPS lookup will rely on storm name only.")
                atcf_ids_decoded = [""] * len(data_filtered["storm"])
            num_storms = len(data_filtered["storm"])

            genesis_filtered_count = 0

            for i in range(num_storms):
                storm_data = data_filtered.isel(storm=i)
                raw_time_values = storm_data["time"].values
                valid_time_mask = ~np.isnan(raw_time_values)

                if not np.any(valid_time_mask):
                    continue

                storm_data = storm_data.isel(date_time=valid_time_mask)
                raw_time_values = storm_data["time"].values

                if np.issubdtype(raw_time_values.dtype, np.datetime64):
                    storm_datetimes = pd.to_datetime(raw_time_values)
                else:
                    reference_date = pd.Timestamp("1858-11-17")
                    storm_datetimes = pd.to_timedelta(raw_time_values, unit="D") + reference_date

                synoptic_hours = storm_datetimes.hour
                synoptic_mask = np.isin(synoptic_hours, [0, 6, 12, 18])

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

                storm_name, storm_year = names_decoded[i], int(storm_data.season.item())
                atcf_id = atcf_ids_decoded[i]

                storm_df = calculate_quadrant_ike_metrics(
                    storm_data, storm_datetimes, ships_df, storm_name, atcf_id
                )

                if storm_df.empty or storm_df["vmax_kts"].isnull().all():
                    continue

                storm_df["storm_name"] = storm_name
                storm_df["storm_year"] = storm_year
                storm_df["basin"] = basin_name
                storm_df["storm_id"] = f"{storm_year}_{basin_name}_{i:03d}"
                all_storms.append(storm_df)

        if verbose:
            print(f"        - Filtered out {genesis_filtered_count} crossover/external storms "
                  f"based on official genesis basin label.")

    if not all_storms:
        return pd.DataFrame()

    master_df = pd.concat(all_storms)
    master_df.dropna(subset=["vmax_kts"], inplace=True)

    master_df = master_df[master_df["vmax_kts"] >= intensity_threshold_kt]

    master_df = master_df.dropna(subset=["total_ike_tj"])
    master_df = master_df[master_df["total_ike_tj"] > 0]

    if verbose:
        print(f"\n--- Master dataset compiled for Hurricane Phase timesteps "
              f"({len(master_df)} valid base timesteps) ---")

    return master_df


def compute_basin_asymmetry_results(master_df, basin_abbrev_map=None):
    """
    Aggregate master_df into per-basin quadrant-average results (geo/sr/mr,
    for IKE bands and wind radii) plus sample-size accounting, as consumed
    by the four polar decomposition plots.

    Parameters
    ----------
    master_df : pandas.DataFrame
        Output of load_master_dataframe_quadrant.
    basin_abbrev_map : dict, optional
        basin name -> 2-letter code. Defaults to BASIN_ABBREV_MAP.

    Returns
    -------
    basin_results : dict
        2-letter basin code -> dict of quadrant averages and sample sizes.
    sample_size_records : list of dict
        One row per basin, for a figure-caption sample-size table.
    """
    basin_abbrev_map = basin_abbrev_map or BASIN_ABBREV_MAP
    basin_results = {}
    sample_size_records = []

    for full_basin_name, basin_group in master_df.groupby("basin"):
        abbrev = basin_abbrev_map.get(full_basin_name)
        if not abbrev:
            continue

        geo_valid = basin_group.copy()

        sr_valid = basin_group.dropna(subset=["s_dir"]).copy()
        sr_valid = sr_valid[sr_valid["ike1826_sr_quad"].apply(lambda x: isinstance(x, list) and len(x) == 4)]

        mr_valid = basin_group.dropna(subset=["m_dir"]).copy()
        mr_valid = mr_valid[mr_valid["ike1826_mr_quad"].apply(lambda x: isinstance(x, list) and len(x) == 4)]

        n_timesteps_geo, n_timesteps_sr, n_timesteps_mr = len(geo_valid), len(sr_valid), len(mr_valid)
        n_storms_geo = geo_valid["storm_id"].nunique() if not geo_valid.empty else 0
        n_storms_sr = sr_valid["storm_id"].nunique() if not sr_valid.empty else 0
        n_storms_mr = mr_valid["storm_id"].nunique() if not mr_valid.empty else 0

        sample_size_records.append({
            "basin": full_basin_name, "abbrev": abbrev,
            "n_storms_geo": n_storms_geo, "n_timesteps_geo": n_timesteps_geo,
            "n_storms_sr": n_storms_sr, "n_timesteps_sr": n_timesteps_sr,
            "n_storms_mr": n_storms_mr, "n_timesteps_mr": n_timesteps_mr,
        })

        if not geo_valid.empty:
            ike1826_geo = np.vstack(geo_valid["ike1826_geo_quad"].values).astype(float)
            ike2633_geo = np.vstack(geo_valid["ike2633_geo_quad"].values).astype(float)
            ikehur_geo = np.vstack(geo_valid["ikehur_geo_quad"].values).astype(float)
            total_ike_geo = np.nan_to_num(ike1826_geo) + np.nan_to_num(ike2633_geo) + np.nan_to_num(ikehur_geo)
            total_geo_avg = np.nanmean(total_ike_geo, axis=0)
        else:
            total_geo_avg = [np.nan] * 4

        basin_results[abbrev] = {
            "n_storms_geo": n_storms_geo, "n_storms_sr": n_storms_sr, "n_storms_mr": n_storms_mr,
            "n_timesteps_geo": n_timesteps_geo, "n_timesteps_sr": n_timesteps_sr, "n_timesteps_mr": n_timesteps_mr,
            "total_geo_avg_ike": total_geo_avg,
            "ike1826_geo_avg": safe_mean(geo_valid, "ike1826_geo_quad"),
            "ike2633_geo_avg": safe_mean(geo_valid, "ike2633_geo_quad"),
            "ikehur_geo_avg": safe_mean(geo_valid, "ikehur_geo_quad"),
            "r34_geo_avg": safe_mean(geo_valid, "r34_nm_geo_quad"),
            "r50_geo_avg": safe_mean(geo_valid, "r50_nm_geo_quad"),
            "r64_geo_avg": safe_mean(geo_valid, "r64_nm_geo_quad"),
            "ike1826_sr_avg": safe_mean(sr_valid, "ike1826_sr_quad"),
            "ike2633_sr_avg": safe_mean(sr_valid, "ike2633_sr_quad"),
            "ikehur_sr_avg": safe_mean(sr_valid, "ikehur_sr_quad"),
            "r34_sr_avg": safe_mean(sr_valid, "r34_nm_sr_quad"),
            "r50_sr_avg": safe_mean(sr_valid, "r50_nm_sr_quad"),
            "r64_sr_avg": safe_mean(sr_valid, "r64_nm_sr_quad"),
            "ike1826_mr_avg": safe_mean(mr_valid, "ike1826_mr_quad"),
            "ike2633_mr_avg": safe_mean(mr_valid, "ike2633_mr_quad"),
            "ikehur_mr_avg": safe_mean(mr_valid, "ikehur_mr_quad"),
            "r34_mr_avg": safe_mean(mr_valid, "r34_nm_mr_quad"),
            "r50_mr_avg": safe_mean(mr_valid, "r50_nm_mr_quad"),
            "r64_mr_avg": safe_mean(mr_valid, "r64_nm_mr_quad"),
        }

    return basin_results, sample_size_records


def compute_global_scale_limits(basin_results):
    """
    Universal (basin-independent) axis/colorbar limits for the polar plots,
    so all six panels of a given figure share one scale.

    Returns
    -------
    (global_max_ike, global_max_r34) : tuple of float
    """
    global_max_ike = 0.1
    global_max_r34 = 10

    for data in basin_results.values():
        for prefix in ["geo", "sr", "mr"]:
            if np.isnan(data[f"ike1826_{prefix}_avg"][0]):
                continue
            max_ike_for_prefix = np.nanmax(
                [data[f"ike1826_{prefix}_avg"], data[f"ike2633_{prefix}_avg"], data[f"ikehur_{prefix}_avg"]]
            )
            if max_ike_for_prefix > global_max_ike:
                global_max_ike = max_ike_for_prefix
            if np.nanmax(data[f"r34_{prefix}_avg"]) > global_max_r34:
                global_max_r34 = np.nanmax(data[f"r34_{prefix}_avg"])

    return global_max_ike, global_max_r34
