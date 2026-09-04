import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from utils.helpers import load_data, save_fig, gradient_descent, predict, evaluate

warnings.filterwarnings("ignore")

df, X, y = load_data()
y_np = y.values

kf = KFold(n_splits=5, shuffle=True, random_state=42)

def cv_evaluate_gd(X, y_np, lr=0.01, n_iter=3000):
    """Run 5-fold CV using gradient descent and return mean metrics."""
    mse_list, rmse_list, r2_list = [], [], []
    for train_idx, test_idx in kf.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y_np[train_idx], y_np[test_idx]
        sc = StandardScaler()
        X_tr = sc.fit_transform(X_tr)
        X_te = sc.transform(X_te)
        w, b = gradient_descent(X_tr, y_tr, lr=lr, n_iter=n_iter)
        y_pred = predict(X_te, w, b)
        m = evaluate(y_te, y_pred)
        mse_list.append(m["MSE"])
        rmse_list.append(m["RMSE"])
        r2_list.append(m["R2"])
    return np.mean(mse_list), np.mean(rmse_list), np.mean(r2_list)

def cv_evaluate_sklearn(model, X, y_np):
    """Run 5-fold CV for an sklearn model and return mean metrics."""
    mse_list, rmse_list, r2_list = [], [], []
    for train_idx, test_idx in kf.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y_np[train_idx], y_np[test_idx]
        sc = StandardScaler()
        X_tr = sc.fit_transform(X_tr)
        X_te = sc.transform(X_te)
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        mse = mean_squared_error(y_te, y_pred)
        mse_list.append(mse)
        rmse_list.append(np.sqrt(mse))
        r2_list.append(r2_score(y_te, y_pred))
    return np.mean(mse_list), np.mean(rmse_list), np.mean(r2_list)

print("=" * 70)
print("Q2.2 - Advanced Models")
print("=" * 70)

# ══ Q2.2a Polynomial Regression (top 10 features) ═══════════════════════════
print("\n" + "-" * 70)
print("Q2.2a - Polynomial Regression (top 10 features, degree 2)")

importance = pd.read_csv("figures/feature_importance.csv")
top10 = importance.head(10)["feature"].tolist()
X_top10 = X[top10].values

# Linear baseline with top 10 features
m_mse, m_rmse, m_r2 = cv_evaluate_gd(X_top10, y_np)
print(f"  Linear (top 10) : MSE={m_mse:.2f}  RMSE={m_rmse:.2f}  R2={m_r2:.4f}")

# Polynomial degree 2 with interaction terms
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_top10)
n_poly_features = X_poly.shape[1]
print(f"  Polynomial features count: {n_poly_features} (from {X_top10.shape[1]} base)")

p_mse, p_rmse, p_r2 = cv_evaluate_gd(X_poly, y_np, lr=0.01, n_iter=3000)
print(f"  Polynomial (d=2): MSE={p_mse:.2f}  RMSE={p_rmse:.2f}  R2={p_r2:.4f}")

# ══ Q2.2b Regularization ═══════════════════════════════════════════════════
print("\n" + "-" * 70)
print("Q2.2b - Ridge and Lasso Regularization (All features)")

X_all = X.values

# Ridge with CV-selected alpha
ridge_cv = RidgeCV(alphas=np.logspace(-4, 4, 20))
ridge_cv.fit(StandardScaler().fit_transform(X_all), y_np)
print(f"  Ridge best alpha: {ridge_cv.alpha_:.4f}")

def ridge_manual(X_tr, y_tr, X_te):
    sc = StandardScaler()
    X_tr = sc.fit_transform(X_tr)
    X_te = sc.transform(X_te)
    m = Ridge(alpha=ridge_cv.alpha_)
    m.fit(X_tr, y_tr)
    return m.predict(X_te)

r_mse, r_rmse, r_r2 = cv_evaluate_sklearn(Ridge(alpha=ridge_cv.alpha_), X_all, y_np)
print(f"  Ridge     : MSE={r_mse:.2f}  RMSE={r_rmse:.2f}  R2={r_r2:.4f}")

# Lasso with CV-selected alpha
lasso_cv = LassoCV(alphas=np.logspace(-4, 2, 20), cv=3, random_state=42, max_iter=10000)
lasso_cv.fit(StandardScaler().fit_transform(X_all), y_np)
print(f"  Lasso best alpha: {lasso_cv.alpha_:.4f}")

def lasso_manual(X_tr, y_tr, X_te):
    sc = StandardScaler()
    X_tr = sc.fit_transform(X_tr)
    X_te = sc.transform(X_te)
    m = Lasso(alpha=lasso_cv.alpha_, max_iter=50000)
    m.fit(X_tr, y_tr)
    return m.predict(X_te)

l_mse, l_rmse, l_r2 = cv_evaluate_sklearn(Lasso(alpha=lasso_cv.alpha_, max_iter=10000), X_all, y_np)
print(f"  Lasso     : MSE={l_mse:.2f}  RMSE={l_rmse:.2f}  R2={l_r2:.4f}")

# Which features does Lasso eliminate?
sc = StandardScaler()
X_sc = sc.fit_transform(X_all)
lasso_final = Lasso(alpha=lasso_cv.alpha_, max_iter=10000)
lasso_final.fit(X_sc, y_np)
zero_coef = [f for f, c in zip(X.columns, lasso_final.coef_) if abs(c) < 1e-6]
n_zero = len(zero_coef)
print(f"\n  Lasso eliminates (coef ~ 0): {n_zero} of {X.shape[1]} features")
print(f"  Examples of eliminated features: {zero_coef[:15]}")

# ══ Q2.2c Model Comparison ════════════════════════════════════════════════
print("\n" + "-" * 70)
print("Q2.2c - Model Comparison: Linear vs Random Forest")

rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_mse, rf_rmse, rf_r2 = cv_evaluate_sklearn(rf, X_all, y_np)
print(f"  Random Forest : MSE={rf_mse:.2f}  RMSE={rf_rmse:.2f}  R2={rf_r2:.4f}")

# Multiple linear regression (all features) - reference
ml_mse, ml_rmse, ml_r2 = cv_evaluate_gd(X_all, y_np)
print(f"  Linear (all)  : MSE={ml_mse:.2f}  RMSE={ml_rmse:.2f}  R2={ml_r2:.4f}")

# ── Summary comparison table ────────────────────────────────────────────────
summary = pd.DataFrame({
    "Model": ["Linear (top10)", "Polynomial (top10, d=2)", "Ridge (all)",
              "Lasso (all)", "Random Forest (all)", "Linear (all)"],
    "MSE":  [m_mse,  p_mse,  r_mse,  l_mse,  rf_mse,  ml_mse],
    "RMSE": [m_rmse, p_rmse, r_rmse, l_rmse, rf_rmse, ml_rmse],
    "R2":   [m_r2,   p_r2,   r_r2,   l_r2,   rf_r2,   ml_r2],
})
print("\n" + "=" * 70)
print("FINAL MODEL COMPARISON (mean over 5 folds)")
print("=" * 70)
print(summary.to_string(index=False))

summary.to_csv("figures/model_comparison.csv", index=False)

# ── Plot coefficients for Ridge vs Lasso ────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))
x_pos = np.arange(X.shape[1])
ax.plot(x_pos, lasso_cv.coef_, "o", markersize=3, label="Lasso", color="coral")
ax.plot(x_pos, ridge_cv.coef_, ".",
        markersize=3, label="Ridge", color="steelblue")
ax.axhline(0, color="black", lw=0.5)
ax.set_title("Coefficients: Ridge vs Lasso (standardized)")
ax.set_xlabel("Feature Index")
ax.set_ylabel("Coefficient Value")
ax.legend()
plt.tight_layout()
save_fig("q2_2_ridge_vs_lasso_coefs.png")
print("\nFigure saved: q2_2_ridge_vs_lasso_coefs.png")
