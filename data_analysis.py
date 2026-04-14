import matplotlib
from matplotlib.lines import Line2D
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import statsmodels.formula.api as smf
from pathlib import Path

# ── 1. LOAD ────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent / "data"

info    = pd.read_csv(DATA_DIR / "amcdata_agreement_info_V2.csv", encoding='latin-1').copy()
vercom  = pd.read_csv(DATA_DIR / "amcdata_vercom_V2.csv", encoding='latin-1').copy()
weapons = pd.read_csv(DATA_DIR / "amcdata_weapons_facilities_V2.csv", encoding='latin-1').copy()

# ── 2. BUILD VERIFICATION INDEX ────────────────────────────
score_cols = [
    'verified_compliance_mechanism_area_access',
    'verified_compliance_mechanism_facility_access',
    'verified_compliance_mechanism_item_access',
    'verified_compliance_mechanism_item_section_access',
    'verified_compliance_mechanism_development',
    'verified_compliance_mechanism_testing',
    'verified_compliance_mechanism_production',
    'verified_compliance_mechanism_possession',
    'verified_compliance_mechanism_transfer',
    'verified_compliance_mechanism_use',
    'verified_compliance_mechanism_general',
    'ntm_interferene',
    'ntm_concealment'
]

for col in score_cols:
    vercom[col] = pd.to_numeric(vercom[col], errors='coerce').fillna(0)

vercom['mechanism_stringency'] = vercom[score_cols].sum(axis=1)

vercom_agg = vercom.groupby('agreement_id').agg(
    n_mechanisms     = ('mechanism_nr', 'count'),
    total_stringency = ('mechanism_stringency', 'sum'),
    mean_stringency  = ('mechanism_stringency', 'mean')
).reset_index()

# ── 3. BUILD WEAPONS AGGREGATION ───────────────────────────
ban_cols = [
    'ban_development', 'ban_testing', 'ban_production',
    'ban_acquisition', 'ban_possession', 'ban_transfer', 'ban_use'
]

for col in ban_cols:
    weapons[col] = pd.to_numeric(weapons[col], errors='coerce').fillna(0)

# Count weapon items and total bans per treaty
weapons_agg = weapons.groupby('agreement_id').agg(
    n_weapon_items = ('item', 'count')
).reset_index()

# Add total bans across all ban columns
weapons_agg['total_bans'] = (
    weapons.groupby('agreement_id')[ban_cols].sum().sum(axis=1).values
)

print("=== weapons_agg sample ===")
print(weapons_agg.head(10).to_string())

# ── 4. WEAPON TYPE FLAGS ───────────────────────────────────
info['weapons_items'] = info['weapons_items'].astype(str).str.strip()
info['weapons_items'] = info['weapons_items'].replace('nan', pd.NA)

info['is_nuclear'] = info['weapons_items'].str.contains(
    'Nuclear|Fissile|Ballistic|ICBM|SLBM|Strategic',
    case=False, na=False).astype(int)

info['is_conventional'] = info['weapons_items'].str.contains(
    'Conventional|Small Arms|Tank|Artillery|Ship|Vessel|'
    'Mines|Mine|Cluster|Firearm|Ammunition|Helicopter|Aircraft',
    case=False, na=False).astype(int)

# Verify the 6 known treaties
check_ids = [130, 270, 310, 344, 350, 360]
print(info[info['agreement_id'].isin(check_ids)][
    ['agreement_id', 'weapons_items',
     'is_nuclear', 'is_conventional']
].to_string())


#── 5. MERGE ALL THREE — ORDER MATTERS ─────────────────────
df = info.merge(vercom_agg,  on='agreement_id', how='left')
df = df.merge(weapons_agg,   on='agreement_id', how='left')

# ── 6. BUILD ANALYSIS SAMPLE ───────────────────────────────
sample = df[
    df['total_stringency'].notna() &
    df['nr_states_parties_total'].notna()
].copy()

sample['is_bilateral'] = (sample['nr_states_parties_total'] == 2).astype(int)
multi    = sample[sample['is_bilateral'] == 0].copy()

# ── 7. COVERAGE CHECKS ─────────────────────────────────────
print(f"\nTotal sample:     {len(sample)} treaties")
print(f"Multilateral:     {len(multi)} treaties")
print(f"Bilateral:        {len(sample) - len(multi)} treaties")

print("\n=== Weapons coverage in multilateral sample ===")
print(f"Treaties WITH weapons data: {multi['n_weapon_items'].notna().sum()}")
print(f"Treaties WITHOUT:           {multi['n_weapon_items'].isna().sum()}")

print("\n=== Key variable summary ===")
print(multi[['total_stringency', 'n_weapon_items', 
             'total_bans', 'nr_states_parties_total']].describe().round(2))

# ── 8. CORRELATION ─────────────────────────────────────────
corr, pval = spearmanr(multi['total_stringency'],
                        multi['nr_states_parties_total'])
print(f"\nSpearman r = {corr:.3f}, p = {pval:.3f} (multilateral, n={len(multi)})")

# ── WEAPON TYPE CONFOUNDER ANALYSIS ───────────────────────

print("\n=== Mean stringency by weapon type ===")
print(multi.groupby('is_nuclear')[
    ['total_stringency', 'nr_states_parties_total']
].mean().round(2).rename(index={0: 'Non-nuclear', 1: 'Nuclear'}))

print()
print(multi.groupby('is_conventional')[
    ['total_stringency', 'nr_states_parties_total']
].mean().round(2).rename(index={0: 'Non-conventional', 1: 'Conventional'}))

# Correlation within nuclear treaties only
nuclear    = multi[multi['is_nuclear'] == 1]
non_nuclear = multi[multi['is_nuclear'] == 0]

if len(nuclear) > 4:
    r_nuc, p_nuc = spearmanr(nuclear['total_stringency'],
                              nuclear['nr_states_parties_total'])
    print(f"\nNuclear treaties only (n={len(nuclear)}):     "
          f"r = {r_nuc:.3f}, p = {p_nuc:.3f}")

if len(non_nuclear) > 4:
    r_non, p_non = spearmanr(non_nuclear['total_stringency'],
                              non_nuclear['nr_states_parties_total'])
    print(f"Non-nuclear treaties (n={len(non_nuclear)}):  "
          f"r = {r_non:.3f}, p = {p_non:.3f}")

# ── 9. REGRESSION ──────────────────────────────────────────
# Fill missing weapons data
multi['n_weapon_items'] = multi['n_weapon_items'].fillna(0)
multi['total_bans']     = multi['total_bans'].fillna(0)

model = smf.ols(
    'nr_states_parties_total ~ total_stringency '
    '+ is_nuclear + is_conventional '
    '+ n_weapon_items + year',
    data=multi
).fit()

print("\n=== Final Regression Results ===")
print(f"N = {int(model.nobs)},  R-squared = {model.rsquared:.3f}")
print()
print(pd.DataFrame({
    'Coefficient' : model.params.round(3),
    'P-value'     : model.pvalues.round(3)
}))
# ── 10. VISUALISATION ──────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Verification Stringency vs State Participation',
             fontsize=13, fontweight='bold')

# Plot 1 — All treaties coloured by type
axes[0].scatter(
    sample[sample['is_bilateral']==0]['total_stringency'],
    sample[sample['is_bilateral']==0]['nr_states_parties_total'],
    color='steelblue', label='Multilateral', alpha=0.7, s=70
)
axes[0].scatter(
    sample[sample['is_bilateral']==1]['total_stringency'],
    sample[sample['is_bilateral']==1]['nr_states_parties_total'],
    color='tomato', label='Bilateral', alpha=0.7, s=70
)
axes[0].set_xlabel('Total Verification Stringency')
axes[0].set_ylabel('Number of State Parties')
axes[0].set_title('All Treaties')
axes[0].legend()

# Plot 2 — Multilateral only with trend line
x = multi['total_stringency']
y = multi['nr_states_parties_total']

axes[1].scatter(x, y, color='steelblue', alpha=0.7, s=70)

m, b = np.polyfit(x, y, 1)
axes[1].plot(sorted(x), [m*xi + b for xi in sorted(x)],
             color='tomato', linewidth=1.5, linestyle='--', label='Trend')

for _, row in multi.iterrows():
    axes[1].annotate(str(int(row['agreement_id'])),
                     (row['total_stringency'], row['nr_states_parties_total']),
                     fontsize=7, alpha=0.6)

axes[1].set_xlabel('Total Verification Stringency')
axes[1].set_ylabel('Number of State Parties')
axes[1].set_title(f'Multilateral Only (n={len(multi)})\nr={corr:.2f}, p={pval:.3f}')
axes[1].legend()

plt.tight_layout()
plt.savefig('verification_analysis_scatter.png', dpi=150)
print("\nPlot saved as verification_analysis_scatter.png")

# Replace your current figure with a 3-panel version
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Verification Stringency vs State Participation',
             fontsize=13, fontweight='bold')

# Plot 1 — All treaties bilateral vs multilateral (same as before)
axes[0].scatter(
    sample[sample['is_bilateral']==0]['total_stringency'],
    sample[sample['is_bilateral']==0]['nr_states_parties_total'],
    color='steelblue', label='Multilateral', alpha=0.7, s=70
)
axes[0].scatter(
    sample[sample['is_bilateral']==1]['total_stringency'],
    sample[sample['is_bilateral']==1]['nr_states_parties_total'],
    color='tomato', label='Bilateral', alpha=0.7, s=70
)
axes[0].set_xlabel('Total Verification Stringency')
axes[0].set_ylabel('Number of State Parties')
axes[0].set_title('All Treaties\n(bilateral vs multilateral)')
axes[0].legend()

# Plot 2 — Multilateral only with trend line (same as before)
x = multi['total_stringency']
y = multi['nr_states_parties_total']
axes[1].scatter(x, y, color='steelblue', alpha=0.7, s=70)
m, b = np.polyfit(x, y, 1)
axes[1].plot(sorted(x), [m*xi + b for xi in sorted(x)],
             color='tomato', linewidth=1.5, linestyle='--', label='Trend')
for _, row in multi.iterrows():
    axes[1].annotate(str(int(row['agreement_id'])),
                     (row['total_stringency'], row['nr_states_parties_total']),
                     fontsize=7, alpha=0.6)
axes[1].set_xlabel('Total Verification Stringency')
axes[1].set_ylabel('Number of State Parties')
axes[1].set_title(f'Multilateral Only (n={len(multi)})\n'
                  f'r={corr:.2f}, p={pval:.3f}')
axes[1].legend()

# Plot 3 — NEW: coloured by weapon type to show confounder
colors = multi.apply(
    lambda r: 'gold'      if r['is_nuclear']      == 1
         else 'seagreen'  if r['is_conventional'] == 1
         else 'mediumpurple',
    axis=1
)

axes[2].scatter(
    multi['total_stringency'],
    multi['nr_states_parties_total'],
    c=colors, alpha=0.8, s=80
)

# Manual legend

legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='gold',
           markersize=9, label='Nuclear'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='seagreen',
           markersize=9, label='Conventional'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='mediumpurple',
           markersize=9, label='Other')
]
axes[2].legend(handles=legend_elements)

for _, row in multi.iterrows():
    axes[2].annotate(str(int(row['agreement_id'])),
                     (row['total_stringency'], row['nr_states_parties_total']),
                     fontsize=7, alpha=0.6)

axes[2].set_xlabel('Total Verification Stringency')
axes[2].set_ylabel('Number of State Parties')
axes[2].set_title('Multilateral — by Weapon Type\n(confounder check)')

plt.tight_layout()
plt.savefig('verification_analysis_fig.png', dpi=150)
print("Plot saved as verification_analysis_fig.png")

'''
1. Load agreement info, vercom, and weapons/facilities CSVs from data/.
2. Build per-row mechanism stringency (sum of score columns), aggregate to treaty-level
   n_mechanisms, total_stringency, mean_stringency.
3. Aggregate weapons per treaty (n_weapon_items, total_bans over ban_* columns).
4. Flag is_nuclear / is_conventional from weapons_items text; spot-check known agreement_ids.
5. Left-merge onto info; keep treaties with stringency and nr_states_parties_total;
   split bilateral vs multilateral for analysis.
6. Print sample sizes, weapons coverage, and describe(); Spearman (stringency vs parties)
   on multilateral treaties.
7. Confounder pass: group means by nuclear/conventional; Spearman within nuclear /
   non-nuclear subsamples when n > 4.
8. OLS on multilateral: parties ~ stringency + type flags + n_weapon_items + year;
   missing weapons fields filled with 0.
9. Save verification_analysis_scatter.png (2-panel) and verification_analysis_fig.png
   (3-panel, multilateral coloured by weapon-type legend).
10. Small n, regex-only type flags (e.g. misses some CBW labels), and zero-imputed
    weapons rows limit causal claims.
'''
