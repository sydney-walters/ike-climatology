"""
asymmetry_tendency.py

IKE asymmetry, expressed as fractions (what share of a storm's total IKE
sits in the down-motion/down-shear/right-of-motion/right-of-shear half)
and as 12-hour forward tendencies (how fast that share is changing),
categorized by shear-motion orientation (Parallel / Perp-Right /
Anti-Parallel / Perp-Left).

Usage
-----
    from ike_climatology import config
    from ike_climatology.asymmetry_tendency import load_master_dataframe_tendency

    master_df = load_master_dataframe_tendency(
        basins=config.BASINS_TO_ANALYZE,
        ibtracs_file_map=config.IBTRACS_FILE_MAP,
        ships_file_map=config.SHIPS_FILE_MAP,
        basin_codes=config.IBTRACS_BASIN_CODES,
        ships_base_path=config.SHIPS_BASE_PATH,
        start_year=config.START_YEAR, end_year=config.END_YEAR,
        land_interaction_threshold_km=config.LAND_INTERACTION_THRESHOLD_KM,
    )
"""

import os

import numpy as np
import pandas as pd
import xarray as xr

from .storm_metrics import M_PER_NM, KT_TO_MS
from .wind_asymmetry import robust_decode, load_specific_ships_file

TROPICAL_PHASES = ["TS", "HU", "TY", "ST", "TC"]


def rotate_to_relative_quadrants(ike_ne, ike_se, ike_sw, ike_nw, reference_heading):
    """
    Distribute geographic IKE (NE/SE/SW/NW quadrant totals) into 360
    one-degree bins, rotate into the frame of reference_heading (degrees),
    and re-sum into 4 relative quadrants: Front-Right, Rear-Right,
    Rear-Left, Front-Left.
    """
    if pd.isna(reference_heading) or pd.isna(ike_ne):
        return np.nan, np.nan, np.nan, np.nan

    azimuth_ike = np.zeros(360)
    azimuth_ike[0:90] = ike_ne / 90.0
    azimuth_ike[90:180] = ike_se / 90.0
    azimuth_ike[180:270] = ike_sw / 90.0
    azimuth_ike[270:360] = ike_nw / 90.0

    heading_idx = int(np.round(reference_heading)) % 360
    rotated_ike = np.roll(azimuth_ike, -heading_idx)

    ike_front_right = np.sum(rotated_ike[0:90])
    ike_rear_right = np.sum(rotated_ike[90:180])
    ike_rear_left = np.sum(rotated_ike[180:270])
    ike_front_left = np.sum(rotated_ike[270:360])

    return ike_front_right, ike_rear_right, ike_rear_left, ike_front_left


def enforce_hemisphere_parity(quads, lat):
    """
    Mirror Left/Right quadrants for Southern Hemisphere storms, so "Right"
    always represents the kinematically additive side of the vortex.
    Input/output order: [Front-Right, Rear-Right, Rear-Left, Front-Left].
    """
    if pd.isna(lat) or lat >= 0 or pd.isna(quads[0]):
        return quads
    return [quads[3], quads[2], quads[1], quads[0]]


def calculate_tendency_asymmetry_metrics(storm_data, land_interaction_threshold_km, time_index):
    """
    Per-storm, per-timestep IKE bands, geographic-quadrant totals
    (NE/SE/SW/NW), motion-relative quadrants (with hemisphere-parity
    correction applied against storm heading), and a normalized asymmetry
    index (std/mean across the 4 geographic quadrants).

    Shear-relative quadrants are NOT computed here -- they require
    shear_dir_deg, which is only available after the SHIPS merge in
    load_master_dataframe_tendency, so that rotation happens at the
    concatenated master_df level instead.

    Returns
    -------
    pandas.DataFrame
        Indexed by time_index. See module docstring for the difference
        from calculate_storm_metrics / calculate_quadrant_ike_metrics.
    """
    rmw_m = storm_data["usa_rmw"].fillna(0).values * M_PER_NM
    vmax_ms = storm_data["usa_wind"].fillna(0).values * KT_TO_MS
    mpres_mb = storm_data["usa_pres"].fillna(np.nan).values

    r34_m = storm_data["usa_r34"].fillna(0).values * M_PER_NM
    r50_m = storm_data["usa_r50"].fillna(0).values * M_PER_NM
    r64_m = storm_data["usa_r64"].fillna(0).values * M_PER_NM

    storm_dir_deg = storm_data["storm_dir"].fillna(np.nan).values
    landfall_km = storm_data["landfall"].fillna(9999).values
    storm_lat_deg = storm_data["lat"].fillna(np.nan).values
    storm_lon_deg = storm_data["lon"].fillna(np.nan).values

    try:
        status_np = np.vectorize(lambda x: x.decode("utf-8").strip())(storm_data["usa_status"].values)
    except Exception:
        status_np = np.array([str(s).strip() for s in storm_data["usa_status"].values])

    num_timesteps = len(time_index)

    ike1826_list, ike2633_list, ikehur_list, total_ike_list = [], [], [], []
    ike_ne_list, ike_se_list, ike_sw_list, ike_nw_list = [], [], [], []
    ike_fr_list, ike_rr_list, ike_rl_list, ike_fl_list = [], [], [], []
    ike_asymmetry_list = []

    for t in range(num_timesteps):
        rmw, mwind, status = rmw_m[t], vmax_ms[t], status_np[t]
        ike_in_quads_1826, ike_in_quads_2633, ike_in_quads_hur = [], [], []

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
                    ike_in_quads_1826.append((0.5 * ike_a * ike_m**2) / 1e12)
            else:
                ike_in_quads_1826 = [0, 0, 0, 0]

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
                    ike_in_quads_2633.append((0.5 * ike_a * ike_m**2) / 1e12)
            else:
                ike_in_quads_2633 = [0, 0, 0, 0]

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
                    ike_in_quads_hur.append((0.5 * ike_a * ike_m**2) / 1e12)
            else:
                ike_in_quads_hur = [0, 0, 0, 0]

        if status in TROPICAL_PHASES and np.nansum(ike_in_quads_1826) > 0:
            ike1826_list.append(np.nansum(ike_in_quads_1826))
            ike2633_list.append(np.nansum(ike_in_quads_2633))
            ikehur_list.append(np.nansum(ike_in_quads_hur))

            total_ike = np.nansum(ike_in_quads_1826) + np.nansum(ike_in_quads_2633) + np.nansum(ike_in_quads_hur)
            total_ike_list.append(total_ike)

            quads_total = [ike_in_quads_1826[q] + ike_in_quads_2633[q] + ike_in_quads_hur[q] for q in range(4)]
            ike_ne_list.append(quads_total[0])
            ike_se_list.append(quads_total[1])
            ike_sw_list.append(quads_total[2])
            ike_nw_list.append(quads_total[3])

            mr_raw = rotate_to_relative_quadrants(
                quads_total[0], quads_total[1], quads_total[2], quads_total[3], storm_dir_deg[t]
            )
            mr_parity = enforce_hemisphere_parity(list(mr_raw), storm_lat_deg[t])

            ike_fr_list.append(mr_parity[0])
            ike_rr_list.append(mr_parity[1])
            ike_rl_list.append(mr_parity[2])
            ike_fl_list.append(mr_parity[3])

            ike_asymmetry_list.append(np.std(quads_total) / np.mean(quads_total))
        else:
            for lst in (
                ike1826_list, ike2633_list, ikehur_list, total_ike_list,
                ike_ne_list, ike_se_list, ike_sw_list, ike_nw_list,
                ike_fr_list, ike_rr_list, ike_rl_list, ike_fl_list, ike_asymmetry_list,
            ):
                lst.append(np.nan)

    time_to_et_hrs = np.full(num_timesteps, np.nan)
    et_indices = np.where((status_np == "EX") | (status_np == "ET"))[0]
    if len(et_indices) > 0:
        first_et_index = et_indices[0]
        hours_before_et = (first_et_index - np.arange(num_timesteps)) * 6
        valid_indices = hours_before_et >= 0
        time_to_et_hrs[valid_indices] = hours_before_et[valid_indices]

    storm_had_landfall = np.any(landfall_km <= land_interaction_threshold_km)

    return pd.DataFrame(
        {
            "ike_18_26_list": ike1826_list, "ike_26_33_list": ike2633_list, "ike_hur_list": ikehur_list,
            "total_ike_tj": total_ike_list,
            "ike_ne_tj": ike_ne_list, "ike_se_tj": ike_se_list, "ike_sw_tj": ike_sw_list, "ike_nw_tj": ike_nw_list,
            "ike_fr_tj": ike_fr_list, "ike_rr_tj": ike_rr_list, "ike_rl_tj": ike_rl_list, "ike_fl_tj": ike_fl_list,
            "ike_asymmetry_index": ike_asymmetry_list,
            "rmw_nm": storm_data["usa_rmw"].values, "vmax_kts": storm_data["usa_wind"].values, "pmin_mb": mpres_mb,
            "storm_dir_deg": storm_dir_deg, "storm_speed_kts": storm_data["storm_speed"].values,
            "landfall_dist_km": landfall_km, "had_landfall": storm_had_landfall,
            "storm_lat_deg": storm_lat_deg, "storm_lon_deg": storm_lon_deg, "time_to_et_hrs": time_to_et_hrs,
        },
        index=time_index,
    )


def load_master_dataframe_tendency(
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
    Build the master DataFrame for the asymmetry-fraction/tendency figures:
    per-basin IBTrACS + SHIPS loading (basin skipped entirely if SHIPS is
    missing), genesis-basin filtering, quadrant IKE computation
    (calculate_tendency_asymmetry_metrics), SHIPS shear matching and
    shear-motion-orientation categorization, quality gates (all 4
    geographic quadrants positive, valid shear match), shear-relative
    quadrant rotation, IKE fractions, and 12-hour forward tendencies.

    Returns
    -------
    pandas.DataFrame
        One row per qualifying timestep, with (among others) total_ike_tj,
        ike_{ne,se,sw,nw}_tj (geographic), ike_{fr,rr,rl,fl}_tj
        (motion-relative), ike_{dr,ur,ul,dl}_tj (shear-relative),
        {front_motion,right_motion,down_shear,right_shear}_frac,
        shear_motion_orient, and the matching *_tendency columns
        (TJ/12hr forward change, grouped by storm_id).
    """
    all_storms = []

    if verbose:
        print(f"\n--- Starting Global Processing ({start_year}-{end_year}) ---")

    for basin_name in basins:
        if verbose:
            print(f"\nProcessing Basin: {basin_name}...")

        ibtracs_path = ibtracs_file_map.get(basin_name)
        ships_filename = ships_file_map.get(basin_name)
        expected_basin_code = basin_codes.get(basin_name)

        if not ibtracs_path or not os.path.exists(ibtracs_path):
            if verbose:
                print(f"  Skipping {basin_name}: IBTrACS missing.")
            continue

        ships_df = None
        if ships_filename:
            ships_df = load_specific_ships_file(os.path.join(ships_base_path, ships_filename))
        if ships_df is None:
            if verbose:
                print(f"  Skipping {basin_name}: SHIPS missing.")
            continue

        with xr.open_dataset(ibtracs_path) as ds:
            data_filtered_by_year = ds.where((ds["season"] >= start_year) & (ds["season"] <= end_year), drop=True)
            names_decoded = robust_decode(data_filtered_by_year["name"])
            if "usa_atcf_id" in data_filtered_by_year:
                atcf_ids_decoded = robust_decode(data_filtered_by_year["usa_atcf_id"])
            else:
                if verbose:
                    print("      WARNING: 'usa_atcf_id' not found; SHIPS lookup will rely on storm name only.")
                atcf_ids_decoded = [""] * len(data_filtered_by_year["storm"])

            genesis_filtered_count = 0

            for i in range(len(data_filtered_by_year["storm"])):
                storm_data = data_filtered_by_year.isel(storm=i)
                raw_time_values = storm_data["time"].values
                if not np.any(~np.isnan(raw_time_values)):
                    continue
                storm_data = storm_data.isel(date_time=~np.isnan(raw_time_values))

                if np.issubdtype(storm_data["time"].values.dtype, np.datetime64):
                    storm_datetimes = pd.to_datetime(storm_data["time"].values)
                else:
                    storm_datetimes = pd.to_timedelta(storm_data["time"].values, unit="D") + pd.Timestamp("1858-11-17")

                synoptic_mask = np.isin(storm_datetimes.hour, [0, 6, 12, 18])
                storm_data, storm_datetimes = storm_data.isel(date_time=synoptic_mask), storm_datetimes[synoptic_mask]

                valid_idx = np.where(~np.isnan(storm_data.lat) & ~np.isnan(storm_data.lon))[0]
                if len(valid_idx) == 0:
                    continue

                first_point_index = valid_idx[0]
                raw_basin = storm_data["basin"].values[first_point_index]
                genesis_basin = (
                    raw_basin.decode("utf-8").strip()
                    if isinstance(raw_basin, (bytes, np.bytes_))
                    else str(raw_basin).strip()
                )
                if genesis_basin != expected_basin_code:
                    genesis_filtered_count += 1
                    continue

                storm_df = calculate_tendency_asymmetry_metrics(storm_data, land_interaction_threshold_km, storm_datetimes)
                if storm_df.empty or storm_df["vmax_kts"].isnull().all():
                    continue

                ships_subset = None
                for candidate in (str(names_decoded[i])[:4].upper(), str(atcf_ids_decoded[i])[:4].upper()):
                    try:
                        ships_subset = ships_df.xs(candidate, level="name_short")
                        break
                    except KeyError:
                        continue

                if ships_subset is not None:
                    reindexed = ships_subset.reindex(storm_datetimes, method="nearest", tolerance=pd.Timedelta("12 hours"))
                    storm_df["shear_mag_ms"] = reindexed["shrmag_vr"].values * KT_TO_MS
                    storm_df["shear_dir_deg"] = reindexed["shrdir_vr"].values
                    storm_df["shear_relative_angle"] = (storm_df["shear_dir_deg"] - storm_df["storm_dir_deg"]) % 360

                    conds = [
                        (storm_df["shear_relative_angle"] >= 315) | (storm_df["shear_relative_angle"] < 45),
                        (storm_df["shear_relative_angle"] >= 45) & (storm_df["shear_relative_angle"] < 135),
                        (storm_df["shear_relative_angle"] >= 135) & (storm_df["shear_relative_angle"] < 225),
                        (storm_df["shear_relative_angle"] >= 225) & (storm_df["shear_relative_angle"] < 315),
                    ]
                    storm_df["shear_motion_orient"] = np.select(
                        conds, ["Parallel", "Perp-Right", "Anti-Parallel", "Perp-Left"], default="Unknown"
                    )

                storm_year = int(storm_data.season.item())
                storm_name = names_decoded[i]
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

    if verbose:
        print("\n--- Compiling Global Master DataFrame ---")

    master_df = pd.concat(all_storms).dropna(subset=["vmax_kts", "shear_mag_ms"])
    master_df = master_df[(master_df["vmax_kts"] > 0) & (master_df["shear_motion_orient"] != "Unknown")]
    master_df = master_df[
        (master_df["ike_ne_tj"] > 0) & (master_df["ike_se_tj"] > 0)
        & (master_df["ike_sw_tj"] > 0) & (master_df["ike_nw_tj"] > 0)
    ]

    if master_df.empty:
        return master_df

    def process_sr_quads(r):
        raw_quads = rotate_to_relative_quadrants(
            r["ike_ne_tj"], r["ike_se_tj"], r["ike_sw_tj"], r["ike_nw_tj"], r["shear_dir_deg"]
        )
        return enforce_hemisphere_parity(list(raw_quads), r["storm_lat_deg"])

    sr_quads = master_df.apply(process_sr_quads, axis=1)
    master_df["ike_dr_tj"] = [x[0] for x in sr_quads]
    master_df["ike_ur_tj"] = [x[1] for x in sr_quads]
    master_df["ike_ul_tj"] = [x[2] for x in sr_quads]
    master_df["ike_dl_tj"] = [x[3] for x in sr_quads]

    master_df["front_motion_ike"] = master_df["ike_fl_tj"] + master_df["ike_fr_tj"]
    master_df["right_motion_ike"] = master_df["ike_fr_tj"] + master_df["ike_rr_tj"]
    master_df["down_shear_ike"] = master_df["ike_dl_tj"] + master_df["ike_dr_tj"]
    master_df["right_shear_ike"] = master_df["ike_dr_tj"] + master_df["ike_ur_tj"]

    master_df["front_motion_frac"] = master_df["front_motion_ike"] / master_df["total_ike_tj"]
    master_df["right_motion_frac"] = master_df["right_motion_ike"] / master_df["total_ike_tj"]
    master_df["down_shear_frac"] = master_df["down_shear_ike"] / master_df["total_ike_tj"]
    master_df["right_shear_frac"] = master_df["right_shear_ike"] / master_df["total_ike_tj"]

    # Grouped by storm_id (not the original script's ['storm_year', 'storm_name'])
    # so that two different storms sharing a name in the same season across
    # different basins can't have their tendencies computed across each
    # other -- storm_id already uniquely identifies a storm via basin +
    # season + index, storm_name/storm_year alone do not.
    group = master_df.groupby("storm_id")
    master_df["total_ike_tendency_12h"] = group["total_ike_tj"].diff(-2) * -1
    master_df["front_motion_tendency"] = group["front_motion_ike"].diff(-2) * -1
    master_df["right_motion_tendency"] = group["right_motion_ike"].diff(-2) * -1
    master_df["down_shear_tendency"] = group["down_shear_ike"].diff(-2) * -1
    master_df["right_shear_tendency"] = group["right_shear_ike"].diff(-2) * -1

    return master_df.reset_index(drop=True)
