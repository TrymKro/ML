import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from utils.helpers import load_data, save_fig

df, X, y = load_data()
y_np = y.values

print("=" * 60)
print("Q2.1 - Feature Importance")
print("=" * 60)

# Use Random Forest for feature importance (handles non-linearity well)
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X, y_np)

importance = pd.DataFrame({
    "feature": X.columns,
    "importance": rf.feature_importances_
}).sort_values("importance", ascending=False)

print("\nTop 20 features by Random Forest importance:")
print(importance.head(20).to_string(index=False))

top10 = importance.head(10)["feature"].tolist()
print(f"\nTop 10 features: {top10}")

importance.to_csv("figures/feature_importance.csv", index=False)

fig, ax = plt.subplots(figsize=(10, 8))
ax.barh(importance.head(20)["feature"][::-1],
        importance.head(20)["importance"][::-1], color="steelblue")
ax.set_title("Feature Importance (Random Forest)")
ax.set_xlabel("Importance")
plt.tight_layout()
save_fig("q2_1_feature_importance.png")
print("Figure saved: q2_1_feature_importance.png")

# Note on correlation caveat: plot correlated features' importance
print("\n" + "-" * 60)
print("Caveat: With correlated features, importance can be diluted/split")
print("between members of a correlated group.")
