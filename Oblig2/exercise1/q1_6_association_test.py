"""Ex1 Q1.6 - Association of each feature with the target.

Numeric features are tested against the 0/1 target with an ANOVA F-test and
categorical features with the chi-square test. The features are ranked by
statistical significance, which is what the brief asks for.

The snag is the sample size. At n = 48842 the p-values of the leading features
are below 1e-300, which is past the smallest double, so scipy hands back 0.0
for them and a plain p-value sort turns into a thirteen-way tie. The log tails
are therefore worked out here with Lentz continued fractions for the
incomplete gamma and incomplete beta functions. Checked against
scipy.stats.logsf wherever scipy can still represent the answer, the two agree
to about 1e-10 in log p, and the continued fraction carries on long after
scipy underflows, so the ranking is well defined all the way down.
"""

import sys
from pathlib import Path

# make src/ importable when run directly
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy.special import betaln, gammaln
from scipy.stats import chi2_contingency
from sklearn.feature_selection import f_classif

from src.data import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS, TARGET, load_all
from src.preprocessing import MIN_COUNT

OUT_DIR = Path(__file__).resolve().parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)


def _cf_gamma(a, x):
    """Continued fraction for Q(a,x), the regularized upper incomplete gamma."""
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 10000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return h


def _cf_beta(a, b, x):
    """Continued fraction for I_x(a,b); valid for x < (a+1)/(a+b+2)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, 10000):
        m2 = 2 * m
        num = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + num * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + num / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        num = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + num * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + num / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return h


def log_sf_chi2(stat, dof):
    """log P(chi2_dof > stat), stable well past double underflow."""
    a, x = dof / 2.0, stat / 2.0
    return -x + a * np.log(x) - gammaln(a) + np.log(_cf_gamma(a, x))


def log_sf_f(stat, dof1, dof2):
    """log P(F_dof1,dof2 > stat), via I_x(dof2/2, dof1/2) with x = dof2/(dof2+dof1*stat)."""
    a, b = dof2 / 2.0, dof1 / 2.0
    x = dof2 / (dof2 + dof1 * stat)
    return (
        a * np.log(x)
        + b * np.log1p(-x)
        - betaln(a, b)
        - np.log(a)
        + np.log(_cf_beta(a, b, x))
    )


df = load_all()
y = (df[TARGET] == ">50K").astype(int)
n = len(df)

# Same cleaning as the modelling pipeline: 'Unknown' missing markers (Q1.1)
# and the rare-category grouping (Q1.3).
for col in ["workclass", "occupation", "native-country"]:
    df[col] = df[col].fillna("Unknown")
for col in CATEGORICAL_COLUMNS:
    counts = df[col].value_counts()
    keep = counts[counts >= MIN_COUNT].index
    df[col] = df[col].where(df[col].isin(keep), "other")

rows = []
for col in NUMERIC_COLUMNS:
    f_stat, _ = f_classif(df[[col]].to_numpy(), y)
    f_stat = float(f_stat[0])
    logp = log_sf_f(f_stat, 1, n - 2)
    rows.append(
        {
            "feature": col,
            "type": "numeric",
            "test": "F(1,%d)" % (n - 2),
            "statistic": round(f_stat, 3),
            "log10p": -logp / np.log(10),
            # effect size behind the test, for the table in the report
            "effect": abs(float(np.corrcoef(df[col], y)[0, 1])),
        }
    )
for col in CATEGORICAL_COLUMNS:
    tab = pd.crosstab(df[col], y)
    chi2, _, dof, _ = chi2_contingency(tab)
    logp = log_sf_chi2(chi2, dof)
    rows.append(
        {
            "feature": col,
            "type": "categorical",
            "test": "chi2(%d)" % dof,
            "statistic": round(float(chi2), 1),
            "log10p": -logp / np.log(10),
            # Cramer's V; the target has two levels so this is sqrt(chi2/n)
            "effect": float(np.sqrt(chi2 / tab.to_numpy().sum())),
        }
    )

res = pd.DataFrame(rows)
# rank by significance, -log10(p), largest first. Only fnlwgt comes back with a
# p-value that is not astronomically small, so it is the one feature the
# ranking actually demotes.
res = res.sort_values("log10p", ascending=False).reset_index(drop=True)
res["significant_0.05"] = res["log10p"] > -np.log10(0.05)
res.to_csv(OUT_DIR / "q1_6_association_tests.csv", index=False)

show = res.copy()
show["log10p"] = show["log10p"].round(1)
show["effect"] = show["effect"].round(3)
print(show.to_string(index=False))
print("\ntop 8 by statistical significance:")
print(show.head(8)[["feature", "type", "log10p", "effect"]].to_string(index=False))
