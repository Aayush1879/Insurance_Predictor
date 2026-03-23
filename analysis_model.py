import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
import joblib


df = pd.read_csv("insurance.csv")
df.head()
# Check for missing values
print(df.isnull().sum())
pd.set_option("display.float_format", "{:.2f}".format)
df.describe()
sns.set(style="whitegrid", font_scale=1.2, palette="Set2")
df.dropna(inplace=True)
df.shape
sns.pairplot(df, hue="smoker")
df.isna().sum()

numeric_cols = ["age", "bmi", "bloodpressure", "children", "claim"]
df[numeric_cols].hist(figsize=(10, 8), bins=25, color="skyblue", edgecolor="black")
plt.suptitle("Distribution of Numeric Features", fontsize=16)
plt.show()

categorical_cols = ["gender", "diabetic", "smoker", "region"]
plt.figure(figsize=(12, 8))

for i, col in enumerate(categorical_cols, 1):
    plt.subplot(2, 2, i)
    sns.countplot(x=col, data=df)
    plt.title(f"Count of {col.capitalize()}")
    plt.xlabel(col.capitalize())
    plt.ylabel("Count")

plt.tight_layout()
plt.show()

df.groupby(["gender", "smoker"])["claim"].mean().round(2)
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x="gender", y="claim", hue="smoker", palette="Set2")
plt.title("Average Claim by Gender and Smoking Status")
plt.xlabel("Gender")
plt.ylabel("Average Claim Amount")
plt.show()

region_diabetic_claims = df.groupby(["region", "diabetic"])["claim"].mean().unstack()
region_diabetic_claims.plot(kind="bar", figsize=(10, 6), color=["skyblue", "salmon"])
plt.title("Average Claim by Region and Diabetes Status")
plt.xlabel("Region")
plt.ylabel("Average Claim Amount")
plt.xticks(rotation=25)
plt.legend(title="Diabetic", labels=["No", "Yes"])
plt.show()

pivot_table = pd.pivot_table(
    df, values="claim", index="region", columns="smoker", aggfunc="mean"
)
pivot_table_children = pd.pivot_table(
    df, values="claim", index="children", columns="smoker", aggfunc="mean"
)

plt.figure(figsize=(10, 6))
sns.heatmap(df[numeric_cols].corr(), annot=True, fmt=".2f", cmap="YlGnBu")
plt.title("Correlation Heatmap of Numeric Features")
plt.show()

sns.scatterplot(
    data=df, x="age", y="claim", hue="smoker", style="gender", palette="Set2", alpha=0.7
)
plt.title("Claim Amount vs Age by Smoking Status")
plt.xlabel("Age")
plt.ylabel("Claim Amount")
plt.show()

sns.regplot(
    data=df, x="bmi", y="claim", scatter_kws={"alpha": 0.5}, line_kws={"color": "red"}
)
plt.title("Claim Amount vs BMI")
plt.xlabel("BMI")
plt.ylabel("Claim Amount")
plt.show()

sns.boxplot(data=df, x="children", y="claim", palette="Set2")
plt.title("Claim Amount by Number of Children")
plt.xlabel("Number of Children")
plt.ylabel("Claim Amount")
plt.show()

df["age_group"] = pd.cut(
    df["age"],
    bins=[0, 18, 30, 45, 60, 100],
    labels=["<18", "18-30", "31-45", "46-60", "60+"],
)
df.value_counts("age_group")
sns.barplot(data=df, x="age_group", y="claim", palette="Set2", errorbar="sd")
plt.title("Average Claim Amount by Age Group")
plt.xlabel("Age Group")
plt.ylabel("Average Claim Amount")
plt.show()

df["bmi_category"] = pd.cut(
    df["bmi"],
    bins=[0, 18.5, 25, 30, 100],
    labels=["Underweight", "Normal", "Overweight", "Obese"],
)
df.value_counts("bmi_category")
sns.barplot(data=df, x="bmi_category", y="claim", palette="Set2", errorbar="sd")
plt.title("Average Claim Amount by BMI Category")
plt.xlabel("BMI Category")
plt.ylabel("Average Claim Amount")
plt.show()
sns.boxplot(data=df, x="bmi_category", y="claim", palette="Set2", hue="smoker")
plt.title("Claim Amount by BMI Category and Smoking Status")
plt.xlabel("BMI Category")
plt.ylabel("Claim Amount")
plt.show()

region_stats = (
    df.groupby("region")
    .agg(
        smoker_rate=("smoker", lambda x: (x == "Yes").mean() * 100),
        mean_claim=("claim", "mean"),
    )
    .reset_index()
)

fig, ax1 = plt.subplots(figsize=(10, 6))
sns.barplot(data=region_stats, x="region", y="smoker_rate", ax=ax1, color="skyblue")
ax1.set_ylabel("Smoker Rate (%)", color="blue")
ax1.tick_params(axis="y", colors="blue")

ax2 = ax1.twinx()
sns.lineplot(
    data=region_stats, x="region", y="mean_claim", ax=ax2, color="red", marker="o"
)
ax2.set_ylabel("Mean Claim Amount", color="red")
ax2.tick_params(axis="y", colors="red")

plt.title("Smoker Rate and Mean Claim by Region")
plt.show()

# ── Feature & target selection ──────────────────────────────────────────────
X = df[
    ["age", "gender", "bmi", "bloodpressure", "children", "diabetic", "smoker"]
].copy()
y = df["claim"]

# ── Label encoding ───────────────────────────────────────────────────────────
cat_cols = ["gender", "diabetic", "smoker"]
label_enc = {}
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_enc[col] = le
    joblib.dump(le, f"{col}_label_encoder.pkl")

# ── Train / test split ───────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── Scaling ──────────────────────────────────────────────────────────────────
num_cols = ["age", "bmi", "bloodpressure", "children"]
scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])
joblib.dump(scaler, "scaler.pkl")


# ── Evaluation  ────────────────────────────────────────────────────────
def evaluate_model(model, X_train, X_test, y_train, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    return {"MSE": mse, "R2": r2, "MAE": mae}


results = {}
trained_models = {}  # ✅ Store actual model objects, keyed by name

# ── Linear Regression ────────────────────────────────────────────────────────
lr = LinearRegression()
lr.fit(X_train, y_train)
results["Linear Regression"] = evaluate_model(lr, X_train, X_test, y_train, y_test)
trained_models["Linear Regression"] = lr

# ── Polynomial Regression (best degree) ──────────────────────────────────────
best_poly_model = None
best_poly_score = -np.inf
best_poly_degree = 2
for degree in range(2, 5):
    poly_features = PolynomialFeatures(degree=degree)
    X_train_poly = poly_features.fit_transform(X_train)
    X_test_poly = poly_features.transform(X_test)

    poly_lr = LinearRegression()
    poly_lr.fit(X_train_poly, y_train)
    score = poly_lr.score(X_test_poly, y_test)

    if score > best_poly_score:
        best_poly_score = score
        best_poly_model = poly_lr
        best_poly_features = poly_features
        best_poly_degree = degree

X_train_poly = best_poly_features.fit_transform(X_train)
X_test_poly = best_poly_features.transform(X_test)
poly_key = f"Polynomial Regression, degree={best_poly_degree}"
results[poly_key] = evaluate_model(
    best_poly_model, X_train, X_test_poly, y_train, y_test
)

# ── Random Forest ────────────────────────────────────────────────────────────
rf = RandomForestRegressor()
rf_params = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
}
rf_grid = GridSearchCV(rf, rf_params, cv=3, n_jobs=-1, verbose=0, scoring="r2")
rf_grid.fit(X_train, y_train)
best_rf = rf_grid.best_estimator_
results["Random Forest"] = evaluate_model(best_rf, X_train, X_test, y_train, y_test)
trained_models["Random Forest"] = best_rf
print("Best Parameters (RF):", rf_grid.best_params_)

# ── SVR ──────────────────────────────────────────────────────────────────────
svr = SVR()
svr_params = {
    "kernel": ["linear", "rbf", "poly"],
    "C": [1, 10, 50],
    "epsilon": [0.1, 0.2, 0.5],
    "degree": [2, 3, 4],
}
svr_grid = GridSearchCV(svr, svr_params, cv=3, n_jobs=-1, verbose=0, scoring="r2")
svr_grid.fit(X_train, y_train)
best_svr = svr_grid.best_estimator_
results["SVR"] = evaluate_model(best_svr, X_train, X_test, y_train, y_test)
trained_models["SVR"] = best_svr
print("Best Parameters (SVR):", svr_grid.best_params_)

# ── XGBoost ──────────────────────────────────────────────────────────────────
xgb = XGBRegressor(objective="reg:squarederror", eval_metric="rmse")
xgb_params = {
    "n_estimators": [100, 200],
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [3, 5, 7],
    "subsample": [0.8, 0.9, 1.0],
    "colsample_bytree": [0.8, 0.9, 1.0],
}
xgb_grid = GridSearchCV(xgb, xgb_params, cv=3, n_jobs=-1, verbose=0, scoring="r2")
xgb_grid.fit(X_train, y_train)
best_xgb = xgb_grid.best_estimator_
results["XGBoost"] = evaluate_model(best_xgb, X_train, X_test, y_train, y_test)
trained_models["XGBoost"] = best_xgb
print("Best Parameters (XGBoost):", xgb_grid.best_params_)

results_df = pd.DataFrame(results).T
results_df = results_df.sort_values(by="R2", ascending=False)
print("\nModel Comparison:\n", results_df)

comparable_results = results_df[results_df.index.isin(trained_models)]
best_model_name = comparable_results.index[0]
best_model_obj = trained_models[best_model_name]  # ✅ Actual model object

print(
    f"\nBest Model: {best_model_name} (R2 = {comparable_results.loc[best_model_name, 'R2']:.4f})"
)
joblib.dump(best_model_obj, "best_model.pkl")  # ✅ Saves the real trained model
print("Saved best_model.pkl successfully.")
