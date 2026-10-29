# =====================================================================
# FOOD DELIVERY DATA ANALYSIS - COMPLETE CODE (Google Colab / local)
# Steps: 1 Load  2 Explore  3 Clean  4 Analysis  5 Charts
#        6 Preprocessing  7 Prediction models  8 Accuracy summary
# =====================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (r2_score, mean_absolute_error, mean_squared_error,
                             accuracy_score, precision_score, recall_score, f1_score)

# ---------------------------------------------------------------------
# STEP 1: LOAD DATASET
# (Colab: upload the csv when asked. Local: keep the csv in the same folder)
# ---------------------------------------------------------------------
try:
    from google.colab import files
    uploaded = files.upload()
    df = pd.read_csv(list(uploaded.keys())[0])
except ImportError:
    df = pd.read_csv("food_delivery_dataset_50000.csv")
print("Original Dataset:")
print(df.head())
print("\nColumn Names:", list(df.columns))

# ---------------------------------------------------------------------
# STEP 2: EXPLORE DATA (before cleaning)
# ---------------------------------------------------------------------
print("\nDataset Shape:", df.shape)
print("\nData Types:")
print(df.dtypes)
print("\nMissing Values:")
print(df.isnull().sum())
print("\nDuplicate Rows:", df.duplicated().sum())
print("\nStatistical Summary:")
print(df.describe())

# ---------------------------------------------------------------------
# STEP 3: CLEAN DATA (missing values + duplicates)
# ---------------------------------------------------------------------
for col in df.columns:
    if df[col].isnull().any():
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])

df = df.drop_duplicates()
print("\nShape After Cleaning:", df.shape)
print("Missing Values After Cleaning:", df.isnull().sum().sum())
print("Duplicate Rows After Cleaning:", df.duplicated().sum())

# ---------------------------------------------------------------------
# STEP 4: ANALYSIS (key numbers)
# ---------------------------------------------------------------------
print("\nOrders by Cuisine:\n", df["Cuisine"].value_counts())
print("\nPayment Methods:\n", df["Payment Method"].value_counts())
print("\nTop Cities:\n", df["City"].value_counts().head())
print("\nOrder Status:\n", df["Order Status"].value_counts())
print("\nAverage Order Value  :", round(df["Order Value"].mean(), 2))
print("Average Delivery Time:", round(df["Delivery Time"].mean(), 2))
print("Average Rating       :", round(df["Customer Rating"].mean(), 2))
print("Delivery Time vs Rating correlation:",
      round(df["Delivery Time"].corr(df["Customer Rating"]), 3))

# ---------------------------------------------------------------------
# STEP 5: VISUALIZATION (bar, pie, histogram, line, scatter, box, heatmap)
# ---------------------------------------------------------------------
def show(name):
    plt.tight_layout()
    plt.savefig(name + ".png", dpi=150)
    plt.show()
    plt.close()

# ----- Bar charts -----
df["Cuisine"].value_counts().plot(kind="bar", color="#1F3756", figsize=(8, 5))
plt.title("Orders by Cuisine"); plt.xlabel("Cuisine"); plt.ylabel("Number of orders")
plt.xticks(rotation=30, ha="right"); show("bar_cuisine")

df["Restaurant"].value_counts().plot(kind="bar", color="#4472C4", figsize=(8, 5))
plt.title("Orders by Restaurant"); plt.xlabel("Restaurant"); plt.ylabel("Number of orders")
plt.xticks(rotation=30, ha="right"); show("bar_restaurant")

df["City"].value_counts().plot(kind="bar", color="#2A7F62", figsize=(8, 5))
plt.title("Orders by City"); plt.xlabel("City"); plt.ylabel("Number of orders")
plt.xticks(rotation=30, ha="right"); show("bar_city")

df["Order Status"].value_counts().plot(kind="bar", color="#E87722", figsize=(7, 5))
plt.title("Order Status"); plt.xlabel("Status"); plt.ylabel("Number of orders")
plt.xticks(rotation=0); show("bar_status")

df.groupby("Cuisine")["Delivery Time"].mean().sort_values().plot(kind="bar", color="#A23B3B", figsize=(8, 5))
plt.title("Average Delivery Time by Cuisine"); plt.ylabel("Minutes")
plt.xticks(rotation=30, ha="right"); show("bar_delivery_by_cuisine")

# ----- Pie charts -----
df["Payment Method"].value_counts().plot(kind="pie", autopct="%1.1f%%", startangle=90, figsize=(6, 6))
plt.title("Payment Method Distribution"); plt.ylabel(""); show("pie_payment")

df["Order Status"].value_counts().plot(kind="pie", autopct="%1.1f%%", startangle=90, figsize=(6, 6))
plt.title("Order Status Share"); plt.ylabel(""); show("pie_status")

df["Cuisine"].value_counts().plot(kind="pie", autopct="%1.1f%%", startangle=90, figsize=(6, 6))
plt.title("Cuisine Share"); plt.ylabel(""); show("pie_cuisine")

# ----- Histograms -----
plt.hist(df["Order Value"], bins=30, color="#1F3756", edgecolor="white")
plt.title("Order Value Distribution"); plt.xlabel("Order value"); plt.ylabel("Orders"); show("hist_order_value")

plt.hist(df["Delivery Time"], bins=30, color="#2A7F62", edgecolor="white")
plt.title("Delivery Time Distribution"); plt.xlabel("Minutes"); plt.ylabel("Orders"); show("hist_delivery_time")

plt.hist(df["Customer Rating"], bins=20, color="#E87722", edgecolor="white")
plt.title("Customer Rating Distribution"); plt.xlabel("Rating"); plt.ylabel("Orders"); show("hist_rating")

# ----- Line chart -----
df.groupby(df["Distance"].round())["Delivery Time"].mean().plot(kind="line", marker="o", figsize=(8, 5))
plt.title("Average Delivery Time vs Distance"); plt.xlabel("Distance (km)"); plt.ylabel("Avg delivery time (min)")
show("line_distance_time")

# ----- Scatter plot -----
s = df.sample(min(2000, len(df)), random_state=42)
plt.scatter(s["Delivery Time"], s["Customer Rating"], s=8, alpha=0.4, color="#A23B3B")
plt.title("Delivery Time vs Customer Rating"); plt.xlabel("Delivery time (min)"); plt.ylabel("Rating")
show("scatter_time_rating")

# ----- Box plot -----
df.boxplot(column="Delivery Time", by="City", figsize=(8, 5))
plt.title("Delivery Time by City"); plt.suptitle(""); plt.xlabel("City"); plt.ylabel("Minutes")
plt.xticks(rotation=30, ha="right"); show("box_time_city")

# ----- Heatmap (correlation) -----
corr = df[["Customer Age", "Order Value", "Distance", "Delivery Time", "Customer Rating"]].corr()
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(corr, cmap="Blues", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=45, ha="right")
ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.columns)
for i in range(len(corr)):
    for j in range(len(corr)):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center")
plt.colorbar(im); plt.title("Correlation Heatmap"); show("heatmap_corr")

# ---------------------------------------------------------------------
# STEP 6: PREPROCESSING (scaling + encoding, done on a copy)
# ---------------------------------------------------------------------
dp = df.copy()

dp["Order Value"] = MinMaxScaler().fit_transform(dp[["Order Value"]])
print("\nAfter Feature Scaling (MinMax):")
print(dp[["Order Value"]].head())

dp["Delivery Time"] = StandardScaler().fit_transform(dp[["Delivery Time"]])
print("\nAfter Feature Standardization:")
print(dp[["Delivery Time"]].head())

encoder = LabelEncoder()
dp["Cuisine"] = encoder.fit_transform(dp["Cuisine"])
print("\nAfter Label Encoding:")
print(dp[["Cuisine"]].head())
print("\nCuisine Label Mapping:")
for i, label in enumerate(encoder.classes_):
    print(label, "=", i)

dp = pd.get_dummies(dp, columns=["Payment Method"])
print("\nAfter One Hot Encoding:")
print(dp.head())
print("\nFinal Preprocessed Dataset:")
print(dp.head())

# ---------------------------------------------------------------------
# STEP 7A: PREDICTION - DELIVERY TIME (regression)
# ---------------------------------------------------------------------
fa = ["Distance", "Order Value", "Customer Age", "City", "Cuisine", "Payment Method"]
X = pd.get_dummies(df[fa], drop_first=True)
y = df["Delivery Time"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

reg_results = []
trained = {}
for name, m in [("Linear Regression", LinearRegression()),
                ("Random Forest Regressor", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))]:
    m.fit(Xtr, ytr)
    p = m.predict(Xte)
    trained[name] = (m, p)
    reg_results.append([name, round(r2_score(yte, p), 3), round(mean_absolute_error(yte, p), 2),
                        round(float(np.sqrt(mean_squared_error(yte, p))), 2)])

rf, prf = trained["Random Forest Regressor"]
plt.scatter(yte, prf, s=4, alpha=0.3)
plt.plot([yte.min(), yte.max()], [yte.min(), yte.max()], "r")
plt.xlabel("Actual delivery time"); plt.ylabel("Predicted delivery time"); plt.title("Actual vs Predicted")
show("actual_vs_predicted")

pd.Series(rf.feature_importances_, X.columns).nlargest(8).sort_values().plot(kind="barh", figsize=(8, 5))
plt.title("Feature Importance (Delivery Time)"); show("feature_importance")

# ---------------------------------------------------------------------
# STEP 7B: PREDICTION - ORDER STATUS (classification)
# ---------------------------------------------------------------------
fb = ["Distance", "Order Value", "Delivery Time", "Customer Rating", "City", "Cuisine"]
X2 = pd.get_dummies(df[fb], drop_first=True)
y2 = df["Order Status"]
Xtr2, Xte2, ytr2, yte2 = train_test_split(X2, y2, test_size=0.2, random_state=42, stratify=y2)

clf_results = []
for name, m in [("Logistic Regression", make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))),
                ("Random Forest Classifier", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))]:
    m.fit(Xtr2, ytr2)
    p = m.predict(Xte2)
    clf_results.append([name, round(accuracy_score(yte2, p), 3),
                        round(precision_score(yte2, p, average="weighted", zero_division=0), 3),
                        round(recall_score(yte2, p, average="weighted", zero_division=0), 3),
                        round(f1_score(yte2, p, average="weighted", zero_division=0), 3)])

# ---------------------------------------------------------------------
# STEP 8: ACCURACY SUMMARY (copy these values into the PPT)
# ---------------------------------------------------------------------
print("\nTASK A - Delivery Time Prediction (Regression)")
print(pd.DataFrame(reg_results, columns=["Model", "R2 Score", "MAE", "RMSE"]).to_string(index=False))
print("\nTASK B - Order Status Prediction (Classification)")
print(pd.DataFrame(clf_results, columns=["Model", "Accuracy", "Precision", "Recall", "F1"]).to_string(index=False))
