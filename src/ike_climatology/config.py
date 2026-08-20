"""
config.py

Central configuration for the six-basin IKE climatology: analysis window,
basin file paths, and shared thresholds. Edit IBTRACS_PATH via the
IBTRACS_PATH environment variable rather than hardcoding a machine-specific
path here.
"""

import os

# --- Analysis window -------------------------------------------------------
START_YEAR = 2004
END_YEAR = 2024

# --- Basins ------------------------------------------------------------
BASINS_TO_ANALYZE = [
    "North Atlantic", "East Pacific", "West Pacific",
    "North Indian", "South Indian", "South Pacific"
]

IBTRACS_BASE_PATH = os.environ.get("IBTRACS_PATH", "./data/ibtracs")

IBTRACS_FILE_MAP = {
    "North Atlantic": os.path.join(IBTRACS_BASE_PATH, "IBTrACS.NA.v04r01.nc"),
    "East Pacific": os.path.join(IBTRACS_BASE_PATH, "IBTrACS.EP.v04r01.nc"),
    "West Pacific": os.path.join(IBTRACS_BASE_PATH, "IBTrACS.WP.v04r01.nc"),
    "North Indian": os.path.join(IBTRACS_BASE_PATH, "IBTrACS.NI.v04r01.nc"),
    "South Indian": os.path.join(IBTRACS_BASE_PATH, "IBTrACS.SI.v04r01.nc"),
    "South Pacific": os.path.join(IBTRACS_BASE_PATH, "IBTrACS.SP.v04r01.nc"),
}

IBTRACS_BASIN_CODES = {
    "North Atlantic": "NA",
    "East Pacific": "EP",
    "West Pacific": "WP",
    "North Indian": "NI",
    "South Indian": "SI",
    "South Pacific": "SP",
}

# --- SHIPS shear diagnostics (used only by wind_asymmetry.py) -------------
# SHIPS files aren't produced with the same per-basin coverage as IBTrACS --
# Southern Hemisphere basins (South Indian, South Pacific) share one file,
# and year ranges vary by basin -- so this is an explicit filename map
# rather than a templated path like IBTRACS_FILE_MAP.
SHIPS_BASE_PATH = os.environ.get("SHIPS_PATH", "./data/ships")

SHIPS_FILE_MAP = {
    "North Atlantic": "SHIPS_NorthAtlantic_2004_2023.nc",
    "East Pacific": "SHIPS_EastPacific_2004_2023.nc",
    "West Pacific": "SHIPS_WestPacific_2004_2021.nc",
    "North Indian": "SHIPS_NorthIndian_2004_2021.nc",
    "South Indian": "SHIPS_SouthernHemisphere_2004_2021.nc",
    "South Pacific": "SHIPS_SouthernHemisphere_2004_2021.nc",
}

# --- Shared analysis parameters ---------------------------------------
LAND_INTERACTION_THRESHOLD_KM = 0
PRE_ET_WINDOW_HRS = 48

# --- Output locations ---------------------------------------------------
BASE_OUTPUT_DIR = os.environ.get(
    "IKE_OUTPUT_DIR", "./output/figures"
)
