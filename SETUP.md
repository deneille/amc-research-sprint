# Setup Guide

If you already have Python and Git working, skip to [Loading the Data](#loading-the-data).

## Installing Python

1. Download Python 3.10+ from [python.org/downloads](https://www.python.org/downloads/)
2. **Windows**: Check "Add Python to PATH" during installation
3. **Mac**: Python 3 is often pre-installed. Check with `python3 --version` in Terminal
4. Verify: open a terminal and run `python --version` (or `python3 --version`)

## Installing packages

```bash
pip install pandas matplotlib
```

Or, from the repo directory:

```bash
pip install -r requirements.txt
```

If `pip` isn't found, try `pip3` or `python -m pip install`.

## Getting the data

### Option A: Clone this repository

If you have Git installed ([git-scm.com/downloads](https://git-scm.com/downloads)):

```bash
git clone https://github.com/WilliamHackspeare/amc-research-sprint.git
cd amc-research-sprint
```

### Option B: Download as ZIP

1. Go to the repository page on GitHub
2. Click the green **Code** button
3. Click **Download ZIP**
4. Extract the ZIP to a folder on your computer

## Setting up your own repository

Each team should create their own GitHub repository for their work. Two options:

### Fork this repo (recommended)

1. On the repository page, click **Fork** (top right)
2. This creates a copy under your own GitHub account
3. Clone your fork: `git clone https://github.com/YOUR-USERNAME/amc-research-sprint.git`
4. Work in your fork and push normally

### Create a fresh repo

1. On GitHub, click **New repository**
2. Name it something like `amc-sprint-challenge-3`
3. Clone it locally, then copy the `data/` folder from this repo into yours

Either way, your analysis code, outputs, and write-up go in your team's repo.

## Running the starter code

From the repo directory:

```bash
python examples/getting_started.py
```

On Mac/Linux you may need `python3` instead of `python`.

This loads all datasets, prints summary statistics, and saves a plot to `overview_plots.png`.

## Running the Challenge 2 analysis

From the repo directory (after `pip install -r requirements.txt`):

```bash
python data_analysis.py
```

This runs the verification-vs-participation models and saves several `verification_analysis_*.png` figures in the current directory.

## Loading the data

The CSV files need a specific encoding parameter in Python:

```python
import pandas as pd

# All files except agreement_associations use latin-1
info = pd.read_csv('data/amcdata_agreement_info_V2.csv', encoding='latin-1')
vercom = pd.read_csv('data/amcdata_vercom_V2.csv', encoding='latin-1')
demcom = pd.read_csv('data/amcdata_demcom_V2.csv', encoding='latin-1')
consultation = pd.read_csv('data/amcdata_consultation_V2.csv', encoding='latin-1')
weapons = pd.read_csv('data/amcdata_weapons_facilities_V2.csv', encoding='latin-1')

# This one is different (has a BOM header)
associations = pd.read_csv('data/amcdata_agreement_associations_V2.csv', encoding='utf-8-sig')
```

In **R**, `read.csv()` handles the encoding automatically. In **MATLAB**, `readtable()` works without issues.

## Working with Jupyter Notebooks

If you prefer notebooks over scripts:

```bash
pip install jupyter
jupyter notebook
```

Create a new notebook and paste the loading code above into the first cell.

## Saving your work with Git

```bash
# Check what you've changed
git status

# Stage your files (replace with your actual filenames)
git add your_analysis.py your_plots.png

# Commit with a message
git commit -m "Challenge 3: time-series analysis of verification design"

# Push to your repository
git push
```

MATLAB `.m` files, Jupyter `.ipynb` notebooks, R `.R` scripts, and Python `.py` files are all welcome.

## Common issues

| Problem | Fix |
|---------|-----|
| `UnicodeDecodeError` when loading CSVs | Add `encoding='latin-1'` to `read_csv()` |
| `ModuleNotFoundError: No module named 'pandas'` | Run `pip install pandas matplotlib` |
| `KeyError: 'agreement_name'` | The column is called `title_short` (or `title_full`) |
| Trailing whitespace in `item` column | Clean with `weapons['item'] = weapons['item'].str.strip()` |
