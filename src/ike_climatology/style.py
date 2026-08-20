"""
style.py

Shared visual style constants (basin/category order and colors) used
across the plotting notebooks in notebooks/. Kept here, rather than
duplicated per-notebook, so that basin and category colors stay
consistent across every manuscript figure.

The plotting code itself intentionally does NOT live in this package --
see notebooks/ for that. Centralizing computation but not plotting keeps
the installable package limited to logic that benefits from being
tested and version-controlled as code, while figures stay easy to
tweak (styling, panel layout, axis limits) directly in the notebook
that produces them.
"""

# Consistent basin order and colors across all basin-comparison panels.
DESIRED_BASIN_ORDER = [
    "North Atlantic", "East Pacific", "West Pacific",
    "North Indian", "South Indian", "South Pacific"
]

BASIN_COLORS = {
    "North Atlantic": "tomato",
    "East Pacific": "lightsalmon",
    "West Pacific": "maroon",
    "North Indian": "crimson",
    "South Indian": "darkblue",
    "South Pacific": "royalblue",
}

# Consistent Saffir-Simpson category colors across all category-comparison panels.
# Order follows categories.CATEGORY_ORDER.
SSHWS_CATEGORY_COLORS = {
    "TS": "#4daf4a",
    "Cat 1": "#f4ce46",
    "Cat 2": "#ff9f2c",
    "Cat 3": "#ff6319",
    "Cat 4": "#d62728",
    "Cat 5": "#800000",
}
