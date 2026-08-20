"""
aggregate.py

Aggregation of per-timestep/per-storm IKE records into the summary tables

- get_interannual_ike_metrics: annual, basin-level IKE summary (genesis,
  landfall, peak, time-average) built from a pre-loaded master_df (see
  ibtracs_io.load_master_dataframe).
- extract_category_breakdown_metrics: a self-contained extraction for the
  4-panel Saffir-Simpson category-breakdown figure (wind-field size, and
  IKE at peak/instantaneous/landfall, by category).
"""

import os

import numpy as np
import pandas as pd
import xarray as xr

from .ibtracs_preprocessing import (
    prepare_ibtracs_storm,
    passes_quality_filters,
    extract_landfall_values,       # first-landfall summary -> (ike, vmax) tuple
    extract_all_landfall_values,   # still used for the multi-event landfall figure
    LANDFALL_DISTANCE_KM,
)

from .storm_metrics import calculate_storm_metrics
from .categories import get_saffir_simpson_cat


def get_interannual_ike_metrics(master_df, pre_et_window_hrs=48):
    """
    Calculate annual IKE metrics per basin from a pre-loaded master
    per-timestep DataFrame (see ibtracs_io.load_master_dataframe).

    Parameters
    ----------
    master_df : pandas.DataFrame
        Concatenated per-timestep storm records with 'basin', 'storm_year',
        'storm_name', 'total_ike_tj', and 'landfall_dist_km' columns.
    pre_et_window_hrs : int, default 48
        Retained for interface compatibility with earlier callers; not
        currently used within this function.

    Returns
    -------
    pandas.DataFrame
        One row per (basin, storm_year), with annual_total_ike,
        annual_average_ike, annual_avg_peak_ike, annual_avg_genesis_ike,
        annual_avg_of_avgs_ike, and annual_avg_landfall_ike.
    """
    print("\n--- Calculating Interannual IKE Metrics ---")

    storm_avg_ike = (
        master_df.groupby(["basin", "storm_year", "storm_name"])["total_ike_tj"]
        .mean()
        .rename("storm_time_avg_ike")
        .reset_index()
    )

    storm_metrics = master_df.groupby(["basin", "storm_year", "storm_name"]).agg(
        peak_ike=("total_ike_tj", "max"),
        # Matches extract_landfall_values' own threshold (LANDFALL_DISTANCE_KM,
        has_landfall=("landfall_dist_km", lambda s: (s <= LANDFALL_DISTANCE_KM).any()),
    ).reset_index()

    storm_metrics = storm_metrics.merge(
        storm_avg_ike[["basin", "storm_year", "storm_name", "storm_time_avg_ike"]],
        on=["basin", "storm_year", "storm_name"],
        how="left",
    )

    annual_metrics = master_df.groupby(["basin", "storm_year"]).agg(
        annual_total_ike=("total_ike_tj", "sum"),
        annual_average_ike=("total_ike_tj", "mean"),
    ).reset_index()

    annual_avg_peak_ike = (
        storm_metrics.groupby(["basin", "storm_year"])["peak_ike"]
        .mean()
        .rename("annual_avg_peak_ike")
        .reset_index()
    )
    annual_metrics = annual_metrics.merge(annual_avg_peak_ike, on=["basin", "storm_year"], how="outer")

    genesis_ike = (
        master_df.dropna(subset=["total_ike_tj"])
        .sort_index()
        .groupby(["basin", "storm_year", "storm_name"])["total_ike_tj"]
        .first()
        .rename("genesis_ike")
        .reset_index()
    )
    annual_avg_genesis_ike = (
        genesis_ike.groupby(["basin", "storm_year"])["genesis_ike"]
        .mean()
        .rename("annual_avg_genesis_ike")
        .reset_index()
    )
    annual_metrics = annual_metrics.merge(annual_avg_genesis_ike, on=["basin", "storm_year"], how="outer")

    annual_avg_of_avgs = (
        storm_metrics.groupby(["basin", "storm_year"])["storm_time_avg_ike"]
        .mean()
        .rename("annual_avg_of_avgs_ike")
        .reset_index()
    )
    annual_metrics = annual_metrics.merge(annual_avg_of_avgs, on=["basin", "storm_year"], how="outer")

    # Landfall IKE: every distinct landfall event contributes its own
    # observation here (a storm with two separate landfalls counts twice),
    # per extract_all_landfall_values / identify_landfall_events. We build
    # this with an explicit loop rather than groupby(...).apply(...) because
    # the per-storm event count varies (0, 1, or more), and apply's result
    # reassembly is unreliable across groups of differing shape.
    landfall_records = []
    for (basin, storm_year, storm_name), g in master_df.groupby(
        ["basin", "storm_year", "storm_name"]
    ):
        for event in extract_all_landfall_values(g):
            landfall_records.append({
                "basin": basin,
                "storm_year": storm_year,
                "storm_name": storm_name,
                "landfall_ike": event["landfall_ike"],
            })

    landfall_ike_per_storm = pd.DataFrame(
        landfall_records, columns=["basin", "storm_year", "storm_name", "landfall_ike"]
    )

    annual_avg_landfall_ike = (
        landfall_ike_per_storm.dropna(subset=["landfall_ike"])
        .groupby(["basin", "storm_year"])["landfall_ike"]
        .mean()
        .rename("annual_avg_landfall_ike")
        .reset_index()
    )
    annual_metrics = annual_metrics.merge(annual_avg_landfall_ike, on=["basin", "storm_year"], how="outer")
    

    print("    - Metrics calculated successfully.")
    return annual_metrics


def extract_sshws_comparison_metrics(
    basins,
    file_map,
    basin_codes,
    start_year,
    end_year,
    land_interaction_threshold_km,
    min_instantaneous_vmax_kts=34,
    verbose=True,
):
    """
    Build per-storm summary metrics, annual aggregates, and per-timestep
    instantaneous IKE/Vmax pairs for the IKE-vs-SSHWS comparison figure.

    This is self-contained (loads and processes its own IBTrACS data
    rather than taking a pre-built master_df) so it can be run
    independently for a single figure without depending on the full
    interannual pipeline having been run first.

    Landfall IKE/Vmax use ibtracs_preprocessing.extract_landfall_values
    (exact coastal contact, LANDFALL_DISTANCE_KM), the same definition
    used everywhere else in the package -- including re-emergence-aware
    handling of storms that make more than one distinct landfall.

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
    min_instantaneous_vmax_kts : float, default 34
        Minimum Vmax for a timestep to be included in instantaneous_df.
    verbose : bool, default True
        Print basin-by-basin progress.

    Returns
    -------
    storm_peak_df : pandas.DataFrame
        One row per storm: peak/genesis/landfall/time-avg/total IKE and
        peak Vmax.
    annual_metrics : pandas.DataFrame
        One row per (basin, storm_year), aggregated from storm_peak_df.
    instantaneous_df : pandas.DataFrame
        One row per synoptic timestep with vmax_kts >= min_instantaneous_vmax_kts
        and a valid (>0) IKE value.
    """
    if verbose:
        print("--- Starting Data Extraction for SSHWS Comparison ---")

    storm_summaries = []
    landfall_records = []
    instantaneous_data = []
    total_storms_processed = 0

    for basin_name in basins:
        if verbose:
            print(f"  - Loading IBTrACS data for {basin_name} ({start_year}-{end_year})")

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

                if not passes_quality_filters(storm_df):
                    continue

                storm_year = int(storm_data.season.item())
                storm_name = names_decoded[i]

                total_storms_processed += 1

                ike_history = storm_df["total_ike_tj"].fillna(0).values
                vmax_history = storm_df["vmax_kts"].fillna(0).values

                peak_idx = np.argmax(vmax_history)
                peak_vmax = vmax_history[peak_idx]

                for t, v in enumerate(vmax_history):
                    if v >= min_instantaneous_vmax_kts and ike_history[t] > 0:
                        instantaneous_data.append({
                            "storm_name": storm_name,
                            "basin": basin_name,
                            "storm_year": storm_year,
                            "vmax_kts": v,
                            "ike_tj": ike_history[t],
                            "storm_lon_deg": storm_df["storm_lon_deg"].iloc[t],
                            "storm_lat_deg": storm_df["storm_lat_deg"].iloc[t],
                        })

                landfall_events = extract_all_landfall_values(storm_df)
                for event in landfall_events:
                    landfall_records.append({
                        "basin": basin_name,
                        "storm_year": storm_year,
                        "storm_name": storm_name,
                        "landfall_ike": event["landfall_ike"],
                        "landfall_vmax_kts": event["landfall_vmax_kts"],
                    })

                valid_ike = storm_df["total_ike_tj"].dropna()
                genesis_ike = valid_ike.iloc[0] if not valid_ike.empty else np.nan

                
                storm_summaries.append({
                    "basin": basin_name,
                    "storm_year": storm_year,
                    "storm_name": storm_name,
                    "peak_ike": storm_df["total_ike_tj"].max(),
                    "storm_total_ike": storm_df["total_ike_tj"].sum(),
                    "storm_time_avg_ike": storm_df["total_ike_tj"].mean(),
                    "genesis_ike": genesis_ike,
                    "has_landfall": len(landfall_events) > 0,
                    "peak_vmax_kts": peak_vmax,
                })
        if verbose:
            print(f"      > Dropped {genesis_filtered_count} crossovers based on origin label.")

    storm_peak_df = pd.DataFrame(storm_summaries)
    landfall_df = pd.DataFrame(landfall_records)


    if storm_peak_df.empty:
        annual_metrics = pd.DataFrame()
    else:
        annual_metrics = storm_peak_df.groupby(["basin", "storm_year"]).agg(
            annual_total_ike=("storm_total_ike", "sum"),
            annual_average_ike=("storm_time_avg_ike", "mean"),
            annual_avg_peak_ike=("peak_ike", "mean"),
            annual_avg_genesis_ike=("genesis_ike", "mean"),
            annual_avg_of_avgs_ike=("storm_time_avg_ike", "mean"),
            annual_avg_landfall_ike=("landfall_ike", "mean"),
        ).reset_index()

    instantaneous_df = pd.DataFrame(instantaneous_data)

    if verbose:
        print(
            f"--- Done. Processed {total_storms_processed} storms across "
            f"{len(basins)} basins. ---"
        )

    return storm_peak_df, landfall_df, annual_metrics, instantaneous_df


def extract_category_breakdown_metrics(
    basins,
    file_map,
    basin_codes,
    start_year,
    end_year,
    land_interaction_threshold_km,
    min_instantaneous_vmax_kts=34,
    verbose=True,
):
    """
    Build per-storm peak-intensity summaries, per-landfall-event records,
    and per-timestep instantaneous records, each categorized by
    Saffir-Simpson category, for the 4-panel category-breakdown figure
    (wind-field size and IKE at peak/instantaneous/landfall, by category).

    Self-contained: loads and processes its own IBTrACS data rather than
    taking a pre-built master_df.

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
    min_instantaneous_vmax_kts : float, default 34
        Minimum Vmax for a timestep to be included in inst_df.
    verbose : bool, default True
        Print basin-by-basin progress.

    Returns
    -------
    peak_df : pandas.DataFrame
        One row per storm: peak_vmax_kts, peak_cat, coincident_ike_tj
        (IKE at the peak-Vmax timestep), mean_r34_km (at peak-Vmax
        timestep).
    landfall_df : pandas.DataFrame
        One row per distinct landfall EVENT (a storm with two separate
        landfalls contributes two rows): storm_name, basin,
        landfall_ike_tj, landfall_vmax_kts, landfall_cat.
    inst_df : pandas.DataFrame
        One row per synoptic timestep with vmax_kts >= min_instantaneous_vmax_kts
        and a valid (>0) IKE value: vmax_kts, cat, ike_tj.
    """
    if verbose:
        print("--- Starting Data Extraction for SSHWS Comparison ---")

    storm_peak_data = []
    landfall_data = []
    instantaneous_data = []
    total_storms_processed = 0

    for basin_name in basins:
        if verbose:
            print(f"  - Loading IBTrACS data for {basin_name} ({start_year}-{end_year})")

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

                if not passes_quality_filters(storm_df):
                    continue

                total_storms_processed += 1
                storm_name = names_decoded[i]

                ike_history = storm_df["total_ike_tj"].fillna(0).values
                vmax_history = storm_df["vmax_kts"].fillna(0).values

                peak_idx = np.argmax(vmax_history)
                peak_vmax = vmax_history[peak_idx]

                for t, v in enumerate(vmax_history):
                    if v >= min_instantaneous_vmax_kts and ike_history[t] > 0:
                        instantaneous_data.append({
                            "storm_name": storm_name,
                            "basin": basin_name,
                            "vmax_kts": v,
                            "cat": get_saffir_simpson_cat(v),
                            "ike_tj": ike_history[t],
                        })

                # One row per distinct landfall event, not just the first.
                for event in extract_all_landfall_values(storm_df):
                    landfall_vmax = event["landfall_vmax_kts"]
                    landfall_data.append({
                        "storm_name": storm_name,
                        "basin": basin_name,
                        "landfall_ike_tj": event["landfall_ike"],
                        "landfall_vmax_kts": landfall_vmax,
                        "landfall_cat": (
                            get_saffir_simpson_cat(landfall_vmax)
                            if not np.isnan(landfall_vmax) else np.nan
                        ),
                    })

                storm_peak_data.append({
                    "storm_name": storm_name,
                    "basin": basin_name,
                    "peak_vmax_kts": peak_vmax,
                    "peak_cat": get_saffir_simpson_cat(peak_vmax),
                    "coincident_ike_tj": ike_history[peak_idx],
                    "mean_r34_km": storm_df["r34_km"].iloc[peak_idx],
                })

        if verbose:
            print(f"      > Dropped {genesis_filtered_count} crossovers based on origin label.")

    peak_df = pd.DataFrame(storm_peak_data)
    landfall_df = pd.DataFrame(landfall_data)
    inst_df = pd.DataFrame(instantaneous_data)

    if verbose:
        print(f"Extraction complete. Generated data for {total_storms_processed} storms.")
        print(f"  peak_df:      {len(peak_df)} rows")
        print(f"  landfall_df:  {len(landfall_df)} rows")
        print(f"  inst_df:      {len(inst_df)} rows")

    return peak_df, landfall_df, inst_df