# IKE Climatology

This repository computes and plots Integrated Kinetic Energy (IKE) for
tropical cyclones, using IBTrACS best-track data and (for some figures)
SHIPS shear diagnostics. This repository utilizes a modified version of
the IKE calculation from Misra et al. 2013 and Klotzbach et al. 2022.

## How this repo is organized, and why

There are three folders that matter:

- **`src/ike_climatology/`** -- this is a Python *package*: code that
  does most of the work (reading IBTrACS files, computing IKE, filtering
  storms). Nothing in here makes a plot. 
- **`notebooks/`** -- one Jupyter notebook per figure. Each notebook
  imports functions from the package above, uses them to get data, and
  then makes exactly one figure (or a small set of related figures).
 

**Why split it this way?** If all the data-loading and IKE-calculation
logic lived inside each notebook, fixing a bug would mean fixing it in
multiple different places (there are 15 notebooks). Instead, the calculation
logic lives once, in the package, and every notebook calls into it. A
notebook itself is *only* plotting code -- so you can freely change how a
figure looks (colors, labels, axis limits) without any risk of changing
the underlying numbers.

## One-time setup

Do this once, from a terminal:

```bash
cd ike-climatology
pip install -e .
```

That second command installs this repo as a Python package on your
machine, in "editable" mode -- meaning if you edit a `.py` file inside
`src/ike_climatology/`, the change takes effect immediately, without
reinstalling. **You must run this from inside the `ike-climatology`
folder** (the one that contains `pyproject.toml`) -- running it from one
level up will fail with an error about not finding a Python project.

If you get an error about a missing package (e.g. `cartopy` or
`statsmodels`), see [Installing dependencies](#installing-dependencies)
below.

## Before you run any notebook: telling it where your data lives

Every notebook needs to know two things:

1. **Where your IBTrACS files are** (always required)
2. **Where your SHIPS files are** (only required for some notebooks --
   see the table below)

You tell it these locations by setting environment variables. The
**first code cell of every notebook already does this for you** -- it
looks like:

```python
import os
os.environ["IBTRACS_PATH"] = "/home/sydney.e.walters/IBTrACS"
os.environ["SHIPS_PATH"] = "/data/jacob.carstens/SHIPS"          # only in some notebooks
os.environ["IKE_OUTPUT_DIR"] = "/home/sydney.e.walters/IBTrACS_Manuscript_Figures"
```

**Just edit those three paths to match where your files actually are**,
then run the notebook from the top. 

### Why does it work this way, instead of a normal shell `export`?

You might expect to just run `export IBTRACS_PATH=...` in a terminal
before starting Jupyter, the way you would for most tools. That *does*
work in some setups -- but in VS Code specifically (and some other
editors), the Jupyter kernel that actually runs your notebook's code is
often started as its own separate process, and it doesn't inherit
whatever you typed in a nearby terminal. If you rely on a terminal
`export` and it doesn't apply, you'll get "file not found" errors with 
no obvious cause. 

**If you add a brand new notebook of your own**, copy that same first
cell in (right after the title, before any `import ike_climatology`
line), and change the three paths to wherever your data lives.

## Running a notebook

Once the first cell has your paths in it, just run the notebook
top to bottom (in Jupyter: Cell -> Run All, or step through with
Shift+Enter). Each notebook is self-contained: it loads its own data,
computes what it needs, and produces its figure(s) at the end, usually
also saving a `.png` file to whatever you set `IKE_OUTPUT_DIR` to.

### Which notebook makes which figure

| Notebook | What it makes | Needs SHIPS data? | Years covered | Basins covered |
|---|---|---|---|---|
| `interannual_ike_figure.ipynb` | 3-panel: total/landfall IKE by year and mean IKE by calendar month by basin | No | 2004-2024 | all 6 |
| `ss_breakdown_figure.ipynb` | 4-panel: wind-field size and IKE by SSHWS category | No | 2004-2024 | all 6 |
| `global_climatology_figure.ipynb` | Track density + mean IKE | No (needs `cartopy` for the map) | 2004-2024 | all 6 |
| `basin_ike_comparison_figure.ipynb` | Violin plot by basin | No | 2004-2024 | all 6 |
| `monthly_climatology_figure.ipynb` | Bar chart: mean IKE by calendar month, by basin | No | 2004-2024 | all 6 |
| `ike_intensity_relationship_figure.ipynb` | Scatter: IKE vs. wind speed or pressure | No | 2004-2024 | all 6 |
| `global_tracks_figure.ipynb` | Every storm's track | No (needs `cartopy`) | 2004-2024 | all 6 |
| `windrose_motion_figure.ipynb` | Polar plot: storm speed and IKE by storm motion direction | No | 2004-2024 | all 6 |
| `temporal_evolution_figure.ipynb` | IKE over a storm's lifetime, vs. wind/pressure/size | No | 2004-2024 | all 6 |
| `wind_asymmetry_figure.ipynb` | Wind-field asymmetry by direction, shear, and motion | **Yes** | 2004-2024, hurricane-strength only | all 6 |
| `asymmetry_tendency_figure.ipynb` | How IKE asymmetry changes over 12 hours, by shear direction | **Yes** | 2004-2024 | all 6 |
| `shear_rate_of_change_figure.ipynb` | How IKE changes, by storm phase and shear strength | **Yes** | 2004-2024 | Atlantic + West Pacific only |
| `rate_of_change_matrix_figure.ipynb` | Same idea as above, comparing basins directly | No | 2004-2024 | Atlantic + West Pacific only |
| `decile_composite_figure.ipynb` | What distinguishes storms that intensify fastest vs. slowest before transitioning | **Yes** | 2004-2024 | Atlantic + West Pacific only |
| `decile_composite_combined_figure.ipynb` | Same analysis, one combined figure with limited variables to compare basins | **Yes** | 2004-2024 | Atlantic + West Pacific only |

If a notebook needs SHIPS data and it can't find the file for a given
basin, it doesn't crash -- it just skips that basin (you'll see a
"Skipping ... due to missing SHIPS data" message printed) and continues
with whatever basins it does have.

## Installing dependencies

If `pip install -e .` didn't already pull in everything (or you're
setting this up somewhere new), here's what each package is for:

```
numpy, pandas          -- basic data handling, used everywhere
xarray, netCDF4         -- reading the IBTrACS/SHIPS NetCDF files
matplotlib, seaborn     -- plotting, used by every notebook
cartopy                 -- only needed by the two notebooks that draw
                           world maps (global_climatology_figure,
                           global_tracks_figure)
scipy, statsmodels      -- statistical tests, used by the notebooks that
                           report p-values / significance (see the table
                           above -- most of the SHIPS-dependent ones, plus
                           basin_ike_comparison_figure and ss_breakdown_figure)
```

Install everything at once with:

```bash
pip install -r requirements.txt
```

`seaborn` is intentionally pinned below version 0.14 -- a couple of
plotting calls in this repo use an older seaborn style that newer
versions removed. This is a known, harmless cosmetic warning today (see
[Known issues](#known-issues-not-yet-fixed) below), not something to fix
by ignoring the pin.

## A few concepts worth understanding before you dig into the code

These aren't bugs, they're deliberate choices that might look confusing
without context, especially if you're comparing numbers between two
figures.

### Why `storm_year` isn't just "the year of the storm's timestamps"

Tropical cyclone "seasons" don't line up neatly with calendar years
everywhere. In the Southern Hemisphere (South Pacific, South Indian
basins), the cyclone season runs roughly July through June. 
IBTrACS gives every storm a single `season` label and by
convention, a Southern Hemisphere season is labeled by the calendar year
it *ends* in. So `season = 2004` means "the 2003-2004 season" -- and a
storm in that season might have its very first observations timestamped
in November or December **2003**, before the season's year has started.

Every notebook filters storms by season (e.g. "give me everything from
season 2004 onward"). If the code then labeled each storm's year using
its *timestamps* instead of its `season` value, a storm that's correctly
included (season 2004) could get some of its rows mislabeled as "2003" --
a year that was never supposed to be in the dataset. 

**The fix, already in place:** `storm_year` is always read directly from
IBTrACS's own `season` field, never derived from timestamps. This means
the "is this storm in my date range" filter and the "what year do I file
this storm under" label always agree.

### Every figure uses the same storm-inclusion rule

`load_master_dataframe`'s `apply_quality_filters` setting controls which
storms are counted in the first place: a storm has to reach at least
34 kt (tropical storm strength) and produce a positive IKE value to be
included. **Every notebook uses this at its default (`True`)** 

There's a second, separate setting: **`drop_non_tropical_phases`**
(normally off). When on, any timestep where a storm isn't classified as
tropical (e.g. extratropical transition) gets removed entirely, rather
than just having `NaN` IKE. 

### Landfall IKE


- **Landfall** = the storm center is at or past the coastline
  (`landfall_dist_km <= 0`). 
- If a storm makes landfall, moves back out over open water, and then
  makes landfall again, that only counts as a **second, separate**
  landfall if it spent at least **24 hours** back over water first. A
  storm that briefly crosses a narrow peninsula or hops between two close
  islands is still treated as one continuous landfall, not two.
- Every landfall-related figure in this repo uses the a function (`extract_all_landfall_values`)
  that returns every distinct landfall a storm made. 

## Repository layout, for reference

```
src/ike_climatology/
    config.py                  # file paths, year ranges, basin lists -- edit here for defaults
    style.py                   # shared colors so every figure looks consistent
    categories.py               # hurricane-category classification
    ibtracs_preprocessing.py   # per-storm cleanup: time filtering, quality checks, landfall detection
    storm_metrics.py           # the core IKE calculation
    ibtracs_io.py              # the main "load all the storms" functions
    aggregate.py                 # turns per-storm data into the summary tables figures need
    stats.py                     # statistical test helper functions
    wind_asymmetry.py            # wind-field-by-quadrant calculations (needs SHIPS)
    asymmetry_tendency.py        # asymmetry-over-time calculations (needs SHIPS)
    shear_rate_of_change.py      # rate-of-change calculations, shear-split (needs SHIPS)
    rate_of_change_matrix.py     # rate-of-change calculations, basin comparison
    decile_composite.py          # fast-vs-slow intensifying storm comparison (needs SHIPS)

notebooks/    # one notebook per figure -- see the table above
scripts/      # headless, no-plotting versions that just write CSVs
```
