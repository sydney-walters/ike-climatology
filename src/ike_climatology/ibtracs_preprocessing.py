"""
ibtracs_preprocessing.py

Per-storm preprocessing and quality-control filters applied ahead of, and
following, the IKE calculation in storm_metrics.calculate_storm_metrics.

Usage
-----
    from ibtracs_preprocessing import prepare_ibtracs_storm, passes_quality_filters
    from storm_metrics import calculate_storm_metrics

    prep = prepare_ibtracs_storm(storm_data, expected_basin_code='WP')
    if prep.rejected:
        continue

    storm_df = calculate_storm_metrics(
        prep.storm_data, land_interaction_threshold_km, prep.storm_datetimes
    )
    if not passes_quality_filters(storm_df):
        continue
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# IBTrACS reference epoch for numeric (non-datetime64) time encodings.
IBTRACS_REFERENCE_DATE = pd.Timestamp("1858-11-17")

# Standard synoptic reporting hours (UTC) in the best-track record.
SYNOPTIC_HOURS = (0, 6, 12, 18)

# Minimum number of synoptic-hour observations required to retain a storm
# for the climatology (below this, a per-timestep trajectory is not
# meaningfully resolved).
MIN_SYNOPTIC_POINTS = 4

# Minimum peak intensity (kt) required for a storm to be considered part of
# the tropical-storm-strength climatology.
MIN_VMAX_KTS = 34

# Distance (km) from a coastline within which a timestep is classified as
# landfall. 0 = exact coastal contact (the storm center is at or past the
# coastline), not a proximity threshold -- see README, "Notes on
# landfall-IKE definitions", for why this needs to be the single
# definition used everywhere in the package.
LANDFALL_DISTANCE_KM = 0.0

# Minimum time (hours) a storm must spend back over water, after leaving
# land, before its next landfall counts as a NEW, physically distinct
# event rather than a brief re-emergence still part of the same landfall
# (e.g. crossing a narrow peninsula or a small island chain).
MIN_REEMERGENCE_HOURS = 24


@dataclass
class StormPrepResult:
    """
    Outcome of per-storm temporal and genesis-basin preprocessing.

    Attributes
    ----------
    storm_data : xarray.Dataset or None
        The input storm_data subset to valid, synoptic-hour timesteps.
        None if the storm was rejected before this subsetting could be
        completed.
    storm_datetimes : pandas.DatetimeIndex or None
        Datetimes corresponding to the retained timesteps in storm_data.
        None if the storm was rejected.
    status : str
        One of:
            'ok'                         -- storm accepted
            'no_valid_times'             -- all reported times were NaN
            'insufficient_synoptic_points' -- fewer than MIN_SYNOPTIC_POINTS
                                             valid synoptic-hour observations
            'no_valid_position'          -- no timestep with a valid lat/lon
                                             (genesis point cannot be determined)
            'genesis_basin_mismatch'     -- storm originated outside the
                                             basin under analysis (crossover)
    """
    storm_data: Optional[object]
    storm_datetimes: Optional[pd.DatetimeIndex]
    status: str

    @property
    def rejected(self) -> bool:
        return self.status != 'ok'


def _decode_basin_code(raw_basin) -> str:
    """Decode a single IBTrACS basin code to a plain string."""
    if isinstance(raw_basin, (bytes, np.bytes_)):
        return raw_basin.decode('utf-8').strip()
    return str(raw_basin).strip()


def prepare_ibtracs_storm(
    storm_data,
    expected_basin_code: Optional[str] = None,
    require_synoptic_hours: bool = True,
    min_synoptic_points: int = MIN_SYNOPTIC_POINTS,
) -> StormPrepResult:
    """
    Subset a single storm's IBTrACS record to valid, synoptic-hour
    timesteps and, optionally, verify that the storm originated within a
    specified basin.

    This performs the temporal and genesis-basin filtering used ahead of
    the per-timestep IKE calculation in the six-basin climatology. It does
    not perform any of the quality checks that depend on the calculated
    IKE values themselves; see passes_quality_filters for those.

    Parameters
    ----------
    storm_data : xarray.Dataset
        A single storm's slice of IBTrACS (i.e., ds.isel(storm=i)), prior
        to any time or basin filtering.
    expected_basin_code : str, optional
        Two-letter IBTrACS basin code (e.g., 'WP', 'NA') the storm's
        genesis point must match. If None, genesis-basin membership is not
        checked, and only temporal filtering is applied.
    require_synoptic_hours : bool, default True
        If True, retain only the 00/06/12/18Z observations. Set to False
        for use cases -- such as SAR validation, where IBTrACS is
        interpolated to irregular acquisition times -- that require the
        full-resolution or non-synoptic time record.
    min_synoptic_points : int, default MIN_SYNOPTIC_POINTS
        Minimum number of retained timesteps required for the storm to be
        accepted. Only enforced when require_synoptic_hours is True.

    Returns
    -------
    StormPrepResult
    """
    # -- Step 1: drop timesteps with unresolvable (NaN) times -----------
    raw_time_values = storm_data['time'].values
    valid_time_mask = ~np.isnan(raw_time_values)
    if not np.any(valid_time_mask):
        return StormPrepResult(None, None, 'no_valid_times')

    storm_data = storm_data.isel(date_time=valid_time_mask)
    raw_time_values = storm_data['time'].values

    # -- Step 2: resolve to a DatetimeIndex -------------------------------
    # IBTrACS time may be encoded either as datetime64 directly or as a
    # numeric offset (days) from the archive's reference epoch, depending
    # on the file/xarray decoding path.
    if np.issubdtype(raw_time_values.dtype, np.datetime64):
        storm_datetimes = pd.to_datetime(raw_time_values)
    else:
        storm_datetimes = pd.to_timedelta(raw_time_values, unit='D') + IBTRACS_REFERENCE_DATE

    # -- Step 3: restrict to synoptic hours, if requested -----------------
    if require_synoptic_hours:
        synoptic_mask = np.isin(storm_datetimes.hour, SYNOPTIC_HOURS)
        storm_data = storm_data.isel(date_time=synoptic_mask)
        storm_datetimes = storm_datetimes[synoptic_mask]

        if len(storm_datetimes) < min_synoptic_points:
            return StormPrepResult(None, None, 'insufficient_synoptic_points')

    # -- Step 4: genesis-basin check ---------------------------------------
    # Genesis basin is determined from the first timestep with a valid
    # position, consistent with IBTrACS convention for storms that cross
    # basin boundaries during their lifetime.
    if expected_basin_code is not None:
        valid_position_indices = np.where(
            ~np.isnan(storm_data.lat) & ~np.isnan(storm_data.lon)
        )[0]
        if len(valid_position_indices) == 0:
            return StormPrepResult(None, None, 'no_valid_position')

        first_point_index = valid_position_indices[0]
        genesis_basin = _decode_basin_code(storm_data['basin'].values[first_point_index])

        if genesis_basin != expected_basin_code:
            return StormPrepResult(None, None, 'genesis_basin_mismatch')

    return StormPrepResult(storm_data, storm_datetimes, 'ok')


def passes_quality_filters(
    storm_df: pd.DataFrame,
    min_vmax_kts: float = MIN_VMAX_KTS,
) -> bool:
    """
    Apply sample-quality checks to the output of
    calculate_storm_metrics, determining whether a storm should be
    retained in the climatology.

    A storm is excluded if any of the following hold: the output frame is
    empty; maximum sustained wind is undefined at every timestep; the
    storm never reached tropical-storm strength (min_vmax_kts); or IKE
    could not be computed at any timestep (e.g., R34 never reported).

    Parameters
    ----------
    storm_df : pandas.DataFrame
        Output of calculate_storm_metrics for a single storm.
    min_vmax_kts : float, default MIN_VMAX_KTS
        Minimum peak sustained wind, in knots, required for inclusion.

    Returns
    -------
    bool
        True if the storm passes all quality checks and should be
        retained; False otherwise.
    """
    if storm_df.empty:
        return False

    if storm_df['vmax_kts'].isnull().all():
        return False

    if storm_df['vmax_kts'].max() < min_vmax_kts:
        return False

    if storm_df['total_ike_tj'].fillna(0).max() <= 0:
        return False

    return True


def identify_landfall_events(
    storm_df: pd.DataFrame,
    land_threshold_km: float = LANDFALL_DISTANCE_KM,
    min_reemergence_hours: float = MIN_REEMERGENCE_HOURS,
):
    """
    Identify distinct landfall events in a single storm's timeseries.

    A timestep is "on land" when landfall_dist_km <= land_threshold_km. A new
    landfall EVENT starts at an on-land timestep only if the storm's most
    recent prior on-land timestep was at least min_reemergence_hours
    earlier. The very first on-land timestep in the record
    always starts event 1.

    Parameters
    ----------
    storm_df : pandas.DataFrame
        Output of calculate_storm_metrics for a single storm, containing
        'landfall_dist_km', 'total_ike_tj', and 'vmax_kts', indexed by
        timestamp.
    land_threshold_km : float, default LANDFALL_DISTANCE_KM (0.0)
    min_reemergence_hours : float, default MIN_REEMERGENCE_HOURS (24)

    Returns
    -------
    list of dict
        One dict per distinct landfall event, in chronological order:
        {'onset_time', 'onset_index', 'landfall_ike', 'landfall_vmax_kts'}.
        Empty list if the storm never makes landfall. landfall_ike/
        landfall_vmax_kts may themselves be NaN if total_ike_tj/vmax_kts
        is NaN at that exact onset timestep (e.g. landfall coincides with
        an extratropical-transition timestep) -- not treated specially,
        consistent with how every other IKE lookup in this package
        handles NaN.
    """
    df = storm_df.sort_index()
    is_land = (df["landfall_dist_km"].values <= land_threshold_km)
    times = df.index

    if not np.any(is_land):
        return []

    events = []
    last_land_idx = None  # index of the most recent on-land timestep seen so far

    for i in range(len(df)):
        if is_land[i]:
            starts_new_event = (
                last_land_idx is None
                or (times[i] - times[last_land_idx]).total_seconds() / 3600.0 >= min_reemergence_hours
            )
            # Only a transition from water to land can start an event --
            # a run of consecutive on-land timesteps is one event, not one
            # per timestep. The first timestep of the storm's record being
            # on-land also counts as a transition (nothing precedes it).
            is_transition = (i == 0) or not is_land[i - 1]

            if is_transition and starts_new_event:
                events.append({
                    "onset_time": times[i],
                    "onset_index": i,
                    "landfall_ike": df["total_ike_tj"].iloc[i],
                    "landfall_vmax_kts": df["vmax_kts"].iloc[i],
                })

            last_land_idx = i

    return events


def extract_landfall_values(
    storm_df: pd.DataFrame,
    landfall_distance_km: float = LANDFALL_DISTANCE_KM,
    min_reemergence_hours: float = MIN_REEMERGENCE_HOURS,
):
    """
    Extract IKE and intensity at the onset of a storm's FIRST distinct
    landfall event (see identify_landfall_events for how "distinct" is
    determined -- a storm re-emerging over water for less than
    min_reemergence_hours before landfall again is still one event).

    Parameters
    ----------
    storm_df : pandas.DataFrame
        Output of calculate_storm_metrics for a single storm, containing
        the columns 'landfall_dist_km', 'total_ike_tj', and 'vmax_kts'.
    landfall_distance_km : float, default LANDFALL_DISTANCE_KM (0.0)
        Distance threshold, in kilometers, defining landfall. 0 = exact
        coastal contact.
    min_reemergence_hours : float, default MIN_REEMERGENCE_HOURS (24)

    Returns
    -------
    tuple of (float, float)
        (landfall_ike_tj, landfall_vmax_kts) at the first landfall
        event's onset. Both are NaN if the storm never makes landfall.
    """
    events = identify_landfall_events(storm_df, landfall_distance_km, min_reemergence_hours)
    if not events:
        return np.nan, np.nan
    return events[0]["landfall_ike"], events[0]["landfall_vmax_kts"]


def extract_all_landfall_values(
    storm_df: pd.DataFrame,
    landfall_distance_km: float = LANDFALL_DISTANCE_KM,
    min_reemergence_hours: float = MIN_REEMERGENCE_HOURS,
):
    """
    Like extract_landfall_values, but returns every distinct landfall
    event for the storm, not just the first -- for a figure that wants to
    treat multiple landfalls (e.g. a storm that hits one island, moves
    back over open water for days, then makes a second, separate
    landfall) as separate observations rather than collapsing to one.

    Returns
    -------
    list of dict
        Same shape as identify_landfall_events's return value.
    """
    return identify_landfall_events(storm_df, landfall_distance_km, min_reemergence_hours)
