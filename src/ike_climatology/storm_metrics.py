"""
storm_metrics.py

Calculation of Integrated Kinetic Energy (IKE) and compilation of
best-track storm variables from IBTrACS data.

IKE is computed following the piecewise wind-band decomposition of
Powell and Reinhold (2007), in which the wind field is partitioned into
three intensity bands -- 34-50 kt, 50-64 kt, and >64 kt -- and integrated
over the area of each band's reported radii (R34, R50, R64). All formulas below
reproduce that decomposition exactly; no values have been altered from the
original implementation.

Usage
-----
    from storm_metrics import calculate_storm_metrics
    df = calculate_storm_metrics(storm_data, land_interaction_threshold_km, time_index)
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Unit conversion constants
# ---------------------------------------------------------------------------
M_PER_NM = 1852           # meters per nautical mile
KT_TO_MS = 1 / 1.94384     # knots to meters per second
NM_TO_KM = 1.852           # nautical miles to kilometers

# IBTrACS USA_STATUS codes corresponding to a tropical (non-extratropical,
# non-disturbance) phase. Only timesteps in one of these phases contribute
# to the IKE climatology.
TROPICAL_PHASES = ['TS', 'HU', 'TY', 'ST', 'TC']

# Variables required from the input dataset.
_REQUIRED_VARS = [
    'usa_rmw', 'usa_wind', 'usa_pres',
    'usa_r34', 'usa_r50', 'usa_r64',
    'storm_dir', 'storm_speed', 'landfall',
    'lat', 'lon', 'usa_status',
]


def _decode_status(status_raw):
    """Decode USA_STATUS from byte strings to Unicode, if necessary."""
    try:
        return np.vectorize(lambda x: x.decode('utf-8').strip())(status_raw)
    except Exception:
        return np.array([str(s).strip() for s in status_raw])


def calculate_storm_metrics(storm_data, land_interaction_threshold_km, time_index):
    """
    Compute per-timestep Integrated Kinetic Energy (IKE) and compile the
    corresponding set of best-track variables for a single storm.

    Parameters
    ----------
    storm_data : xarray.Dataset
        Per-storm subset of IBTrACS (v04r01), containing the variables
        usa_rmw, usa_wind, usa_pres, usa_r34, usa_r50, usa_r64, storm_dir,
        storm_speed, landfall, lat, lon, and usa_status. The wind-radii
        variables (usa_r34, usa_r50, usa_r64) are two-dimensional, with
        dimensions (time, quadrant).
    land_interaction_threshold_km : float
        Distance threshold, in kilometers, used by downstream land-
        interaction filtering. Not used within the IKE calculation itself;
        retained here for interface consistency across the processing
        pipeline and recorded as a DataFrame attribute on the returned
        object.
    time_index : array-like
        Timestamps to assign as the index of the returned DataFrame
        (e.g., storm_data.time).

    Returns
    -------
    pandas.DataFrame
        Indexed by time_index, with one row per best-track timestep and
        the following columns:

        IKE components (TJ):
            ike1826_tj   -- 34-50 kt band
            ike2633_tj   -- 50-64 kt band
            ikehur_tj    -- >64 kt band
            total_ike_tj -- sum of the three bands

        Intensity and size, native (reported) units:
            rmw_nm, vmax_kts, pmin_mb, r34_nm, r50_nm, r64_nm

        Intensity and size, SI units (as used in the IKE integration):
            rmw_m, vmax_ms

        Wind radii by quadrant, native units:
            r34_quadrants_nm, r50_quadrants_nm, r64_quadrants_nm

        Storm motion:
            storm_dir_deg, stormspd_kts, stormspd_ms

        Position and land interaction:
            storm_lat_deg, storm_lon_deg, landfall_dist_km

        Storm phase:
            usa_status
    """

    missing = [v for v in _REQUIRED_VARS if v not in storm_data]
    if missing:
        raise KeyError(f"storm_data is missing required variable(s): {missing}")

    if land_interaction_threshold_km is None:
        raise ValueError("land_interaction_threshold_km must be specified (e.g., 0).")

    num_timesteps = len(time_index)

    # -----------------------------------------------------------------
    # Convert reported quantities to SI units for the IKE integration.
    # -----------------------------------------------------------------
    rmw_m = storm_data['usa_rmw'].fillna(0).values * M_PER_NM
    vmax_ms = storm_data['usa_wind'].fillna(0).values * KT_TO_MS
    mpres_mb = storm_data['usa_pres'].fillna(np.nan).values

    r34_m = storm_data['usa_r34'].fillna(0).values * M_PER_NM
    r50_m = storm_data['usa_r50'].fillna(0).values * M_PER_NM
    r64_m = storm_data['usa_r64'].fillna(0).values * M_PER_NM

    storm_dir_deg = storm_data['storm_dir'].fillna(np.nan).values
    stormspd_ms = storm_data['storm_speed'].fillna(0).values * KT_TO_MS
    landfall_km = storm_data['landfall'].fillna(9999).values
    storm_lat_deg = storm_data['lat'].fillna(np.nan).values
    storm_lon_deg = storm_data['lon'].fillna(np.nan).values

    status_np = _decode_status(storm_data['usa_status'].values)

    if len(status_np) != num_timesteps:
        raise ValueError(
            f"usa_status length ({len(status_np)}) does not match "
            f"time_index length ({num_timesteps})."
        )

    ike1826_ike_list, ike2633_ike_list, ikehur_ike_list, total_ike_list = [], [], [], []

    # -----------------------------------------------------------------
    # Per-timestep IKE integration.
    #
    # For each of the four IBTrACS quadrants, the mean wind speed and
    # effective area of each intensity band are estimated, then combined
    # as IKE = 0.5 * rho * mean_wind^2 * area.
    # -----------------------------------------------------------------
    for t in range(num_timesteps):
        rmw = rmw_m[t]
        mwind = vmax_ms[t]
        status = status_np[t]

        ike_in_quads_1826, ike_in_quads_2633, ike_in_quads_hur = [], [], []

        if status in TROPICAL_PHASES:

            # --- 34-50 kt band -----------------------------------------
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

        # A timestep contributes to the climatology only if it is in a
        # tropical phase and the 34-50 kt band integrated to a positive
        # value (i.e., at least one quadrant had a valid R34).
        if status in TROPICAL_PHASES and np.nansum(ike_in_quads_1826) > 0:
            ike1826_ike_list.append(np.nansum(ike_in_quads_1826))
            ike2633_ike_list.append(np.nansum(ike_in_quads_2633))
            ikehur_ike_list.append(np.nansum(ike_in_quads_hur))
            total_ike_list.append(
                np.nansum(ike_in_quads_1826) + np.nansum(ike_in_quads_2633) + np.nansum(ike_in_quads_hur)
            )
        else:
            ike1826_ike_list.append(np.nan)
            ike2633_ike_list.append(np.nan)
            ikehur_ike_list.append(np.nan)
            total_ike_list.append(np.nan)

    # -----------------------------------------------------------------
    # Assemble the output DataFrame.
    # -----------------------------------------------------------------
    df = pd.DataFrame({
        # IKE components
        'ike1826_tj': ike1826_ike_list,
        'ike2633_tj': ike2633_ike_list,
        'ikehur_tj': ikehur_ike_list,
        'total_ike_tj': total_ike_list,

        # Intensity and size, native units
        'rmw_nm': storm_data['usa_rmw'].values,
        'vmax_kts': storm_data['usa_wind'].values,
        'pmin_mb': mpres_mb,

        # Intensity and size, SI units (as used in the IKE integration)
        'rmw_m': rmw_m,
        'vmax_ms': vmax_ms,

        # Wind radii, quadrant-mean, in km 
        'r34_km': storm_data['usa_r34'].values.mean(axis=1) * NM_TO_KM,
        'r50_km': storm_data['usa_r50'].values.mean(axis=1) * NM_TO_KM,
        'r64_km': storm_data['usa_r64'].values.mean(axis=1) * NM_TO_KM,

        # Storm motion
        'storm_dir_deg': storm_dir_deg,
        'stormspd_kts': storm_data['storm_speed'].values,
        'stormspd_ms': stormspd_ms,

        # Position and land interaction
        'storm_lat_deg': storm_lat_deg,
        'storm_lon_deg': storm_lon_deg,
        'landfall_dist_km': landfall_km,

        # Storm phase
        'usa_status': status_np,
    }, index=time_index)

    # Per-quadrant wind radii, retained in full (the columns above report
    # only the quadrant mean).
    df['r34_quadrants_nm'] = list(storm_data['usa_r34'].values)
    df['r50_quadrants_nm'] = list(storm_data['usa_r50'].values)
    df['r64_quadrants_nm'] = list(storm_data['usa_r64'].values)

    df.attrs['land_interaction_threshold_km'] = land_interaction_threshold_km

    return df
