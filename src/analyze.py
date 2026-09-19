from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports"; OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(42)
n = 12000
tenure = rng.integers(0, 73, n)
contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[.55, .25, .20])
internet = rng.choice(["Fiber optic", "DSL", "No"], n, p=[.48, .42, .10])
monthly = np.round(25 + (internet == "Fiber optic") * 45 + (internet == "DSL") * 20 + rng.normal(0, 12, n), 2)
logit = 1.1 * (contract == "Month-to-month") + .75 * (internet == "Fiber optic") - .035 * tenure + .012 * (monthly - 60) - 1.3
churn = rng.binomial(1, 1 / (1 + np.exp(-logit)))
df = pd.DataFrame({"tenure_months": tenure, "contract": contract, "internet_service": internet, "monthly_charges": monthly, "churn": churn})
df.loc[rng.choice(df.index, 80, replace=False), "monthly_charges"] = np.nan
df.to_csv(OUT / "synthetic_churn_data.csv", index=False)

sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.barplot(data=df, x="contract", y="churn", estimator="mean", ax=axes[0]); axes[0].set(title="Churn rate by contract", ylabel="Churn rate")
sns.boxplot(data=df, x="churn", y="tenure_months", ax=axes[1]); axes[1].set(title="Tenure distribution by churn", xlabel="Churn (0=no, 1=yes)")
fig.tight_layout(); fig.savefig(OUT / "eda_summary.png", dpi=160); plt.close(fig)

features = ["tenure_months", "contract", "internet_service", "monthly_charges"]
numeric = ["tenure_months", "monthly_charges"]; categorical = ["contract", "internet_service"]
prep = ColumnTransformer([("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric), ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)])
X_train, X_test, y_train, y_test = train_test_split(df[features], df.churn, test_size=.2, stratify=df.churn, random_state=42)
model = Pipeline([("prep", prep), ("model", LogisticRegression(max_iter=1000))]).fit(X_train, y_train)
pred = model.predict_proba(X_test)[:, 1]
(OUT / "model_report.txt").write_text(f"Held-out test rows: {len(y_test)}\nROC-AUC: {roc_auc_score(y_test, pred):.3f}\n\n" + classification_report(y_test, pred >= .5, zero_division=0))
df.groupby("contract").churn.agg(["count", "mean"]).to_csv(OUT / "churn_by_contract.csv")
names = model.named_steps["prep"].get_feature_names_out()
pd.DataFrame({"feature": names, "coefficient": model.named_steps["model"].coef_[0]}).sort_values("coefficient", ascending=False).to_csv(OUT / "model_coefficients.csv", index=False)
(OUT / "insights.md").write_text("# Analysis notes\n\nThis experiment uses synthetic data with deliberately embedded relationships. Segment differences are descriptive, not causal evidence.\n\nThe model uses a stratified 80/20 split. Median imputation, scaling and encoding are fit only on training rows. ROC-AUC measures ranking; the classification report uses a 0.5 threshold.\n\nCandidate retention experiments: onboarding support for early-tenure customers and contract-value offers for month-to-month customers. Validate these hypotheses on real, consented data and a randomized experiment before acting.\n")
print("Created reports/eda_summary.png and reports/model_report.txt")
