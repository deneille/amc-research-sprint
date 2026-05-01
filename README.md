# AMC Research Sprint — Challenge 2 
## Title - Verification Stringency and Treaty Participation in Arms Control Agreements

**Researchers:** Zahraa Kapasi & Deneille Guiseppi

This repository contains **Challenge 2** only: empirical work on whether verification stringency relates to treaty membership (state party counts), using the Alva Myrdal Centre Arms Control Agreement Database (V2).

- **Part A (empirical):** Required — see [`challenges/02_verification_participation.md`](challenges/02_verification_participation.md) and [`challenges/overview.md`](challenges/overview.md).
- **Part B (optional):** AI governance extension described in the challenge brief.

## Repository structure

```
├── challenges/              # Challenge 2 specification and overview
├── data/                    # AMC datasets (CSV) and codebook
├── examples/                # Starter script for loading and exploring data
├── data_analysis.py         # Main analysis: stringency vs participation
├── SETUP.md                 # Python, Git, and environment setup
└── requirements.txt         # Python dependencies
```

## Getting started

**New to Python or Git?** See [`SETUP.md`](SETUP.md).

1. Read [`challenges/02_verification_participation.md`](challenges/02_verification_participation.md).
2. Read [`data/README.md`](data/README.md) for dataset descriptions and encoding (`latin-1` for most CSVs).
3. Run `pip install -r requirements.txt`.
4. Run `python examples/getting_started.py` for a quick data tour, or `python data_analysis.py` for the full analysis (writes PNG figures in the repo root).

## Datasets

| Dataset | Rows | Description |
|---------|------|-------------|
| `agreement_info` | 128 | Metadata, dates, participation, compliance flags |
| `vercom` | 99 | Verified compliance mechanisms |
| `demcom` | 136 | Demonstrated compliance mechanisms |
| `consultation` | 159 | Consultation mechanisms |
| `weapons_facilities` | 434 | Weapons types and restrictions |
| `agreement_associations` | 137 | Links between related agreements |

Full codebook: [`data/codebook/amc_agreementdatasets_codebook.pdf`](data/codebook/amc_agreementdatasets_codebook.pdf)

## Data access

The AMC datasets are shared for sprint participants only. Do not redistribute without permission from the Alva Myrdal Centre for Nuclear Disarmament, Uppsala University.
