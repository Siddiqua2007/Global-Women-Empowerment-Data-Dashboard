# train_employment_model.py
# Usage: python train_employment_model.py
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import matplotlib.pyplot as plt

# ---------- File paths (edit if necessary) ----------
EMPLOY_PATH = r"C:\Users\admin\Desktop\Pyhton\Employement_rate_data.csv"
LIT_PATH = r"C:\Users\admin\Desktop\Pyhton\women_literacy_rate.csv"
POP_PATH = r"C:\Users\admin\Desktop\Pyhton\Female_population.csv"
EMPOWER_PATH = r"C:\Users\admin\Desktop\Pyhton\women_empowerement_index_data.csv"
SAFETY_PATH = r"C:\Users\admin\Desktop\Pyhton\safety_index.csv"


# ---------- small helper to reshape WB-like wide -> long ----------
def reshape_worldbank_df(df, value_name):
    # assumes year columns are digits (e.g., '2015', '2016', ...)
    year_cols = [c for c in df.columns if str(c).isdigit()]
    id_cols = [c for c in df.columns if c not in year_cols]
    df_long = df.melt(id_vars=id_cols, value_vars=year_cols, var_name="Year", value_name=value_name)
    df_long["Year"] = pd.to_numeric(df_long["Year"], errors="coerce")
    df_long = df_long.dropna(subset=["Year", value_name])
    return df_long

# ---------- Read & reshape ----------
emp_raw = pd.read_csv(EMPLOY_PATH, skiprows=4)   # many WB files have 4 metadata rows
if "Indicator Code" in emp_raw.columns:
    # try to select female employment code if present
    mask = emp_raw["Indicator Code"].astype(str).str.contains("SL.TLF.CACT.FE.ZS", na=False)
    if mask.sum() > 0:
        emp_df = emp_raw[mask].copy()
    else:
        emp_df = emp_raw.copy()
else:
    emp_df = emp_raw.copy()

emp_df = reshape_worldbank_df(emp_df, "Employment")
# normalize country column name
if "Country Name" in emp_df.columns and "Country" not in emp_df.columns:
    emp_df = emp_df.rename(columns={"Country Name":"Country"})

lit_raw = pd.read_csv(LIT_PATH, skiprows=4)
lit_df = reshape_worldbank_df(lit_raw, "Literacy")
if "Country Name" in lit_df.columns and "Country" not in lit_df.columns:
    lit_df = lit_df.rename(columns={"Country Name":"Country"})

# population percentage: detect female % column, handle both wide and snapshot formats
pop_raw = pd.read_csv(POP_PATH)
female_col = None
for c in pop_raw.columns:
    if "female" in c.lower() and "%" in c:
        female_col = c
        break

if female_col is None:
    # fallback to first numeric column after a country column
    numeric_cols = pop_raw.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        female_col = numeric_cols[0]
    else:
        raise RuntimeError("Cannot find female population % column in Female_population.csv")

# if pop file has years as columns (digits), melt it, else assume snapshot and assign Year=2024
if any(col.isdigit() for col in pop_raw.columns):
    year_cols = [c for c in pop_raw.columns if c.isdigit()]
    id_cols = [c for c in pop_raw.columns if c not in year_cols]
    pop_long = pop_raw.melt(id_vars=id_cols, value_vars=year_cols, var_name="Year", value_name=female_col)
    pop_long["Year"] = pd.to_numeric(pop_long["Year"], errors="coerce")
    pop_long = pop_long.dropna(subset=["Year", female_col])
    if "Country Name" in pop_long.columns:
        pop_long = pop_long.rename(columns={"Country Name":"Country"})
    pop_df = pop_long[["Country","Year",female_col]].rename(columns={female_col:"Population_female_pct"})
else:
    if "Country Name" in pop_raw.columns:
        pop_df = pop_raw.rename(columns={"Country Name":"Country"})[["Country", female_col]].copy()
        pop_df = pop_df.rename(columns={female_col:"Population_female_pct"})
        pop_df["Year"] = 2024
    else:
        pop_df = pd.DataFrame({ "Country": pop_raw.iloc[:,0], "Population_female_pct": pop_raw.iloc[:,1] })
        pop_df["Year"] = 2024

# WEI (empowerment) - static column
empower_raw = pd.read_csv(EMPOWER_PATH)
wei_col = next((c for c in empower_raw.columns if "wei" in c.lower() or "empower" in c.lower()), None)
if wei_col is None:
    raise RuntimeError("Cannot find WEI column in empowerment CSV")
if "Country Name" in empower_raw.columns:
    empower_raw = empower_raw.rename(columns={"Country Name":"Country"})
empower_df = empower_raw[["Country", wei_col]].rename(columns={wei_col:"WEI_2022"})

# Safety index - prefer year 2024 if present
safety_raw = pd.read_csv(SAFETY_PATH)
if set(["country","year","score"]).issubset(set(safety_raw.columns)):
    safety_2024 = safety_raw[safety_raw["year"]==2024][["country","score"]].rename(columns={"country":"Country","score":"Safety_2024"})
else:
    # fallback: if a numeric year column exists, pick latest
    years = [c for c in safety_raw.columns if str(c).isdigit()]
    if years:
        latest = max(years, key=int)
        safety_2024 = safety_raw.rename(columns={latest:"Safety_2024"})
        if "country" in safety_2024.columns:
            safety_2024 = safety_2024[["country","Safety_2024"]].rename(columns={"country":"Country"})
    else:
        safety_2024 = pd.DataFrame(columns=["Country","Safety_2024"])

# ---------- Merge everything ----------
merged = emp_df.merge(lit_df[["Country","Year","Literacy"]], on=["Country","Year"], how="left")
merged = merged.merge(pop_df, on=["Country","Year"], how="left")
merged = merged.merge(empower_df, on="Country", how="left")
if not safety_2024.empty:
    merged = merged.merge(safety_2024, on="Country", how="left")
else:
    merged["Safety_2024"] = np.nan

# numeric safety
for c in ["Employment","Literacy","WEI_2022","Safety_2024","Population_female_pct"]:
    if c in merged.columns:
        merged[c] = pd.to_numeric(merged[c], errors="coerce")

# create lag feature and drop rows without lag
merged = merged.sort_values(["Country","Year"])
merged["Employment_lag1"] = merged.groupby("Country")["Employment"].shift(1)
df_model = merged.dropna(subset=["Employment","Employment_lag1"]).copy()

# choose features - fill small missings with median
feature_cols = ["Employment_lag1","Literacy","Population_female_pct","WEI_2022","Safety_2024"]
for c in feature_cols:
    if c not in df_model.columns:
        df_model[c] = np.nan
    df_model[c] = df_model[c].fillna(df_model[c].median())

X = df_model[feature_cols]
y = df_model["Employment"]

# train/test (random split) - small projects: random split is acceptable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

lr = LinearRegression()
rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)

lr.fit(X_train, y_train)
rf.fit(X_train, y_train)

def evaluate(model, Xtr, ytr, Xte, yte):
    yptr = model.predict(Xtr)
    ypte = model.predict(Xte)
    return {
        "train_r2": r2_score(ytr, yptr),
        "test_r2": r2_score(yte, ypte),
        "train_mae": mean_absolute_error(ytr, yptr),
        "test_mae": mean_absolute_error(yte, ypte),
        "train_rmse": np.sqrt(mean_squared_error(ytr, yptr)),
        "test_rmse": np.sqrt(mean_squared_error(yte, ypte))
    }

metrics_lr = evaluate(lr, X_train, y_train, X_test, y_test)
metrics_rf = evaluate(rf, X_train, y_train, X_test, y_test)

print("LinearRegression metrics:", metrics_lr)
print("RandomForest metrics:", metrics_rf)

# feature importances (RF)
fi = pd.DataFrame({"feature":feature_cols, "importance": rf.feature_importances_}).sort_values("importance", ascending=False)
print("Feature importances:\n", fi)

# save predictions sample and model
preds = X_test.copy()
preds["true_employment"] = y_test.values
preds["pred_rf"] = rf.predict(X_test)
preds["pred_lr"] = lr.predict(X_test)
pred_csv = "employment_predictions_test_sample.csv"
preds.to_csv(pred_csv, index=False)
print("Saved test predictions ->", pred_csv)

# save RF model to file
joblib.dump(rf, "rf_employment_model.joblib")
print("Saved RandomForest model -> rf_employment_model.joblib")

# quick example: next-year prediction for India (if available)
def predict_next_for_country(country, model, merged_df):
    cdf = merged_df[merged_df["Country"].str.strip()==country].sort_values("Year")
    if cdf.empty:
        return None
    last = cdf.dropna(subset=["Employment"]).iloc[-1]
    feat = {
        "Employment_lag1": last["Employment"],
        "Literacy": last["Literacy"] if not pd.isna(last["Literacy"]) else X["Literacy"].median(),
        "Population_female_pct": last["Population_female_pct"] if not pd.isna(last.get("Population_female_pct", np.nan)) else X["Population_female_pct"].median(),
        "WEI_2022": last["WEI_2022"] if not pd.isna(last.get("WEI_2022", np.nan)) else X["WEI_2022"].median(),
        "Safety_2024": last["Safety_2024"] if not pd.isna(last.get("Safety_2024", np.nan)) else X["Safety_2024"].median()
    }
    return model.predict(pd.DataFrame([feat]))[0]

if "India" in merged["Country"].values:
    print("India next-year employment (RF):", predict_next_for_country("India", rf, merged))
else:
    print("India not in merged data to demonstrate next-year prediction.")
