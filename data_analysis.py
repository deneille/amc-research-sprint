"""
Challenge 2: verification stringency vs state participation (AMC V2).
Spec: challenges/02_verification_participation.md — run from repo root.
Prints analysis to stdout; saves verification_analysis_n59_vs_n18.png and
verification_analysis_multilateral_overlay.png in the repo root.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import statsmodels.formula.api as smf
import statsmodels.api as sm
from pathlib import Path
import re

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

# ── 2B. RUBRIC-BASED STRINGENCY (five components → rubric_stringency_total) ─
# d1 intrusiveness — How invasive verification is: onsite access flags in vercom,
#   plus trigger/type/category text (challenge-like vs remote-only vs none).
# d2 reporting — Declarations / timelines in agreement info, boosted if independent
#   inspection (d4==2) or timed reporting; also text hits for report-like mechanisms.
# d3 enforcement — Escalation architecture: consultation vs demonstrated compliance vs
#   association bodies (numeric fields in agreement info).
# d4 independence — Who inspects: international/independent vs joint/committee vs none,
#   from inspector text fields in vercom (regex on lowercased strings).
# d5 scope — Treaty breadth on weapons data: counts of listed items and ban columns,
#   not derived from verification wording.
def _safe_text(value):
    if pd.isna(value):
        return ""
    return str(value).strip().lower()

def _contains_any(text, patterns):
    return any(re.search(pat, text) for pat in patterns)


def _score_intrusiveness(group):
    # d1: 0 none, 1 remote-only triggers, 2 any access, 3 challenge-like + access
    access_cols = [
        'verified_compliance_mechanism_area_access',
        'verified_compliance_mechanism_facility_access',
        'verified_compliance_mechanism_item_access',
        'verified_compliance_mechanism_item_section_access'
    ]
    has_access = (group[access_cols].sum(axis=1) > 0).any()

    trigger_text = (
        group['verified_compliance_mechanism_trigger_type'].map(_safe_text) + " " +
        group['verified_compliance_mechanism_agreement_trigger_type'].map(_safe_text) + " " +
        group['verified_compliance_mechanism_type'].map(_safe_text) + " " +
        group['verified_compliance_mechanism_category'].map(_safe_text)
    ).str.cat(sep=" ")

    challenge_like = _contains_any(
        trigger_text,
        [r'challenge', r'any time', r'upon notification', r'on request', r'short notice']
    )
    remote_only = _contains_any(
        trigger_text,
        [r'aerial', r'satellite', r'remote', r'observation']
    ) and not has_access

    if challenge_like and has_access:
        return 3
    if has_access:
        return 2
    if remote_only:
        return 1
    return 0

def _score_independence(group):
    inspector_text = (
        group['verified_compliance_mechanism_inspector_type'].map(_safe_text) + " " +
        group['verified_compliance_mechanism_inspector_type_established_body'].map(_safe_text) + " " +
        group['verified_compliance_mechanism_inspector_type_utlilized_body'].map(_safe_text)
    ).str.cat(sep=" ")

    if _contains_any(inspector_text, [r'iaea', r'opcw', r'un', r'international', r'independent']):
        return 2
    if _contains_any(inspector_text, [r'joint', r'committee', r'state part']):
        return 1
    return 0

rubric_vercom = vercom.groupby('agreement_id').apply(
    lambda g: pd.Series({
        'd1_intrusiveness': _score_intrusiveness(g),
        'd4_independence': _score_independence(g),
        'has_reporting_mechanism': int(_contains_any(
            (
                g['verified_compliance_mechanism_type'].map(_safe_text) + " " +
                g['verified_compliance_mechanism_category'].map(_safe_text)
            ).str.cat(sep=" "),
            [r'report', r'declar', r'information', r'data exchange', r'data']
        ))
    })
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
df = df.merge(rubric_vercom, on='agreement_id', how='left')

# d2 reporting — 0 none … 3 strongest: agreement info fields + has_reporting_mech + d4 for top bin
info_reporting_required = pd.to_numeric(df['general_infromation_keeping'], errors='coerce').fillna(0)
info_reporting_timed = pd.to_numeric(df['general_information_keeping_timeline'], errors='coerce').fillna(0)
has_reporting_mech = pd.to_numeric(df['has_reporting_mechanism'], errors='coerce').fillna(0)

df['d2_reporting'] = np.select(
    [
        (info_reporting_required > 0) & ((df['d4_independence'].fillna(0) == 2) | (info_reporting_timed > 0)),
        (info_reporting_required > 0),
        (has_reporting_mech > 0)
    ],
    [3, 2, 1],
    default=0
)

consultation = pd.to_numeric(df['consultation_mechanism'], errors='coerce').fillna(0)
demonstrated = pd.to_numeric(df['demonstrated_compliance_mechanism'], errors='coerce').fillna(0)
assoc_est = pd.to_numeric(df['agreement_association_established'], errors='coerce').fillna(0)
assoc_use = pd.to_numeric(df['agreement_association_utlilized'], errors='coerce').fillna(0)

df['d3_enforcement'] = np.select(
    [
        (demonstrated > 0) & ((assoc_est > 0) | (assoc_use > 0)),
        (demonstrated > 0) | (assoc_est > 0) | (assoc_use > 0),
        (consultation > 0)
    ],
    [3, 2, 1],
    default=0
)

# d5 scope — 0 no items/bans, 1 some, 2 many items or bans (thresholds on weapons table)
scope_items = pd.to_numeric(df['n_weapon_items'], errors='coerce').fillna(0)
scope_bans = pd.to_numeric(df['total_bans'], errors='coerce').fillna(0)

df['d5_scope'] = np.select(
    [
        (scope_items > 2) | (scope_bans > 3),
        (scope_items > 0) | (scope_bans > 0)
    ],
    [2, 1],
    default=0
)

for col in ['d1_intrusiveness', 'd4_independence']:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

df['rubric_stringency_total'] = (
    df['d1_intrusiveness'] +
    df['d2_reporting'] +
    df['d3_enforcement'] +
    df['d4_independence'] +
    df['d5_scope']
)

# ── 6. BUILD ANALYSIS SAMPLE ───────────────────────────────
sample = df[
    df['rubric_stringency_total'].notna() &
    df['nr_states_parties_total'].notna()
].copy()

sample['is_bilateral'] = (sample['nr_states_parties_total'] == 2).astype(int)
multi    = sample[sample['is_bilateral'] == 0].copy()

# ── 7. COVERAGE CHECKS ─────────────────────────────────────
print(f"\nTotal sample:     {len(sample)} treaties")
print(f"Multilateral:     {len(multi)} treaties")
print(f"Bilateral:        {len(sample) - len(multi)} treaties")
multi_with_verif = multi[multi['n_mechanisms'].fillna(0) > 0]
print(f"Multilateral (all):                    {len(multi)} treaties")
print(f"Multilateral (verification-coded >0):  {len(multi_with_verif)} treaties")

def run_subset_analysis(subset, label):
    subset = subset.copy()
    subset['log_parties'] = np.log1p(subset['nr_states_parties_total'])
    subset['n_weapon_items'] = subset['n_weapon_items'].fillna(0)
    subset['format_clean'] = subset['format'].fillna('Unknown').astype(str).str.strip()
    subset.loc[subset['format_clean'].eq(''), 'format_clean'] = 'Unknown'

    print(f"\n=== {label} ===")
    print(f"Treaties WITH weapons data: {subset['n_weapon_items'].notna().sum()}")
    print(f"Treaties WITHOUT:           {subset['n_weapon_items'].isna().sum()}")
    print(subset[['rubric_stringency_total', 'n_weapon_items',
                  'total_bans', 'nr_states_parties_total']].describe().round(2))

    r_all, p_all = spearmanr(subset['rubric_stringency_total'], subset['nr_states_parties_total'])
    print(f"\nSpearman r = {r_all:.3f}, p = {p_all:.3f} ({label}, n={len(subset)})")

    print("\nMean rubric stringency by weapon type")
    print(subset.groupby('is_nuclear')[
        ['rubric_stringency_total', 'nr_states_parties_total']
    ].mean().round(2).rename(index={0: 'Non-nuclear', 1: 'Nuclear'}))

    nuclear = subset[subset['is_nuclear'] == 1]
    non_nuclear = subset[subset['is_nuclear'] == 0]
    if len(nuclear) > 4:
        r_nuc, p_nuc = spearmanr(nuclear['rubric_stringency_total'], nuclear['nr_states_parties_total'])
        print(f"Nuclear only (n={len(nuclear)}):     r = {r_nuc:.3f}, p = {p_nuc:.3f}")
    if len(non_nuclear) > 4:
        r_non, p_non = spearmanr(non_nuclear['rubric_stringency_total'], non_nuclear['nr_states_parties_total'])
        print(f"Non-nuclear only (n={len(non_nuclear)}): r = {r_non:.3f}, p = {p_non:.3f}")

    if len(subset) >= 40:
        ols_specs = [
            ("Baseline (rubric only)", 'nr_states_parties_total ~ rubric_stringency_total'),
            ("+ confounders: year, is_nuclear", 'nr_states_parties_total ~ rubric_stringency_total + year + is_nuclear'),
            ("+ confounders + n_weapon_items", 'nr_states_parties_total ~ rubric_stringency_total + year + is_nuclear + n_weapon_items'),
            ("Log outcome + confounders + n_weapon_items", 'log_parties ~ rubric_stringency_total + year + is_nuclear + n_weapon_items'),
        ]
    else:
        ols_specs = [
            ("Baseline (rubric only)", 'nr_states_parties_total ~ rubric_stringency_total'),
            ("+ confounders: year, is_nuclear", 'nr_states_parties_total ~ rubric_stringency_total + year + is_nuclear'),
            ("Log outcome + confounders", 'log_parties ~ rubric_stringency_total + year + is_nuclear'),
        ]

    print("\nOLS (progressive confounders)")
    for spec_label, formula in ols_specs:
        fitted = smf.ols(formula, data=subset).fit()
        coef = fitted.params.get('rubric_stringency_total', np.nan)
        pval = fitted.pvalues.get('rubric_stringency_total', np.nan)
        print(f"{spec_label}: coef = {coef:.3f}, p = {pval:.3f}, R2 = {fitted.rsquared:.3f}, N = {int(fitted.nobs)}")

    if len(subset) < 25:
        # Small-sample fallback: permutation and bootstrap diagnostics
        print("\nSmall-sample inference (n < 25): skipping count models")
        rng = np.random.default_rng(42)
        x = subset['rubric_stringency_total'].to_numpy()
        y = subset['nr_states_parties_total'].to_numpy()

        # Permutation test for Spearman correlation
        n_perm = 5000
        perm_rs = np.empty(n_perm)
        for i in range(n_perm):
            perm_rs[i], _ = spearmanr(x, rng.permutation(y))
        perm_p = np.mean(np.abs(perm_rs) >= abs(r_all))
        print(f"Permutation Spearman p (two-sided, {n_perm} perms) = {perm_p:.3f}")

        # Bootstrap CIs for Spearman and baseline OLS coefficient
        n_boot = 3000
        boot_r = np.empty(n_boot)
        boot_beta = np.empty(n_boot)
        n_obs = len(subset)
        for i in range(n_boot):
            idx = rng.integers(0, n_obs, n_obs)
            boot_df = subset.iloc[idx]
            boot_r[i], _ = spearmanr(
                boot_df['rubric_stringency_total'],
                boot_df['nr_states_parties_total']
            )
            boot_model = smf.ols(
                'nr_states_parties_total ~ rubric_stringency_total',
                data=boot_df
            ).fit()
            boot_beta[i] = boot_model.params.get('rubric_stringency_total', np.nan)

        r_ci = np.nanpercentile(boot_r, [2.5, 97.5])
        beta_ci = np.nanpercentile(boot_beta, [2.5, 97.5])
        print(f"Bootstrap 95% CI (Spearman r): [{r_ci[0]:.3f}, {r_ci[1]:.3f}]")
        print(f"Bootstrap 95% CI (OLS beta): [{beta_ci[0]:.3f}, {beta_ci[1]:.3f}]")
    else:
        mean_parties = subset['nr_states_parties_total'].mean()
        var_parties = subset['nr_states_parties_total'].var()
        dispersion_ratio = var_parties / mean_parties if mean_parties > 0 else np.nan
        print(f"\nCount diagnostics: mean = {mean_parties:.2f}, var = {var_parties:.2f}, var/mean = {dispersion_ratio:.2f}")

        count_formula = (
            'nr_states_parties_total ~ rubric_stringency_total + year + is_nuclear + n_weapon_items'
            if len(subset) >= 40 else
            'nr_states_parties_total ~ rubric_stringency_total + year + is_nuclear'
        )
        poisson_model = smf.glm(count_formula, data=subset, family=sm.families.Poisson()).fit()
        print(f"Poisson: coef = {poisson_model.params['rubric_stringency_total']:.3f}, "
              f"p = {poisson_model.pvalues['rubric_stringency_total']:.3f}, AIC = {poisson_model.aic:.2f}")
        try:
            nb_model = smf.negativebinomial(count_formula, data=subset).fit(disp=False)
            nb_coef = nb_model.params.get('rubric_stringency_total', np.nan)
            nb_p = nb_model.pvalues.get('rubric_stringency_total', np.nan)
            alpha = nb_model.params['alpha'] if 'alpha' in nb_model.params.index else np.exp(nb_model.params['lnalpha'])
            converged = nb_model.mle_retvals.get('converged', True)
            print(f"Negative Binomial: coef = {nb_coef:.3f}, p = {nb_p:.3f}, AIC = {nb_model.aic:.2f}, "
                  f"alpha = {alpha:.3f}, converged = {converged}")
        except Exception as e:
            print(f"Negative Binomial failed: {e}")

    return r_all, p_all

# Run both sample definitions side-by-side
rubric_corr, rubric_pval = run_subset_analysis(multi, "Multilateral (all)")
_ = run_subset_analysis(multi_with_verif, "Multilateral (verification-coded >0)")
# ── 10. VISUALISATION (multilateral only; no bilateral plots) ─
subset18 = multi_with_verif.copy()
sub_r, sub_p = spearmanr(
    subset18['rubric_stringency_total'],
    subset18['nr_states_parties_total']
)

# Direct n=59 vs n=18 comparison: shared scales so panels are comparable
x59 = multi['rubric_stringency_total'].to_numpy()
y59 = multi['nr_states_parties_total'].to_numpy()
x18cmp = subset18['rubric_stringency_total'].to_numpy()
y18cmp = subset18['nr_states_parties_total'].to_numpy()
x_lo = min(x59.min(), x18cmp.min()) - 0.25
x_hi = max(x59.max(), x18cmp.max()) + 0.25
y_lo = 0.0
y_hi = max(y59.max(), y18cmp.max()) * 1.05

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
fig.suptitle(
    'Multilateral samples: all treaties vs verification-coded subset (same axis limits)',
    fontsize=13, fontweight='bold'
)

ax = axes[0]
ax.scatter(x59, y59, color='steelblue', alpha=0.7, s=65, edgecolors='white', linewidths=0.4)
m59, b59 = np.polyfit(x59, y59, 1)
xs59 = np.linspace(x_lo, x_hi, 50)
ax.plot(xs59, m59 * xs59 + b59, color='tomato', linestyle='--', linewidth=2, label='OLS trend')
ax.set_xlim(x_lo, x_hi)
ax.set_ylim(y_lo, y_hi)
ax.set_xlabel('Rubric verification stringency')
ax.set_ylabel('Number of state parties')
ax.set_title(f'All multilateral (n={len(multi)})\nSpearman r={rubric_corr:.2f}, p={rubric_pval:.3f}')
ax.legend(loc='upper left', fontsize=8)
ax.grid(True, alpha=0.25)

ax = axes[1]
rng_j = np.random.default_rng(42)
x18_plot = x18cmp + rng_j.normal(0, 0.05, size=len(x18cmp))
ax.scatter(x18_plot, y18cmp, color='darkviolet', alpha=0.85, s=85, edgecolors='white', linewidths=0.5)
m18c, b18c = np.polyfit(x18cmp, y18cmp, 1)
ax.plot(xs59, m18c * xs59 + b18c, color='black', linestyle='--', linewidth=2, label='OLS trend')
ax.set_xlim(x_lo, x_hi)
ax.set_ylim(y_lo, y_hi)
ax.set_xlabel('Rubric verification stringency (x-jitter for overlap)')
ax.set_title(
    f'Verification-coded only (n={len(subset18)})\nSpearman r={sub_r:.2f}, p={sub_p:.3f}'
)
ax.legend(loc='upper left', fontsize=8)
ax.grid(True, alpha=0.25)

plt.tight_layout()
plt.savefig('verification_analysis_n59_vs_n18.png', dpi=150)
print("Plot saved as verification_analysis_n59_vs_n18.png")

# Overlay: full multilateral (faint) + verification-coded (emphasized), one frame
fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(
    x59, y59, color='steelblue', alpha=0.35, s=42, label=f'All multilateral (n={len(multi)})', zorder=1
)
ax.scatter(
    x18cmp, y18cmp, color='darkviolet', alpha=0.95, s=110,
    edgecolors='black', linewidths=0.6, label=f'Verification-coded (n={len(subset18)})', zorder=3
)
ax.plot(xs59, m59 * xs59 + b59, color='tomato', linestyle='--', linewidth=2,
        label=f'Trend n={len(multi)} (r={rubric_corr:.2f})', zorder=2)
ax.plot(xs59, m18c * xs59 + b18c, color='black', linestyle='-', linewidth=2,
        label=f'Trend n={len(subset18)} (r={sub_r:.2f})', zorder=2)
ax.set_xlim(x_lo, x_hi)
ax.set_ylim(y_lo, y_hi)
ax.set_xlabel('Rubric verification stringency')
ax.set_ylabel('Number of state parties')
ax.set_title('Overlay: verification-coded treaties within full multilateral cloud')
ax.legend(loc='upper left', fontsize=9)
ax.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig('verification_analysis_multilateral_overlay.png', dpi=150)
print("Plot saved as verification_analysis_multilateral_overlay.png")
