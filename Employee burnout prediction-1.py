# ------------------------------------------------------------
# EMPLOYEE BURNOUT PREDICTION & EDA - Machine Learning Program
# ------------------------------------------------------------

# ------------------------------------------------------------
# 1. AUTOMATIC LIBRARY INSTALL + IMPORT
# ------------------------------------------------------------
import subprocess
import sys
import importlib

REQUIRED_PACKAGES = {
    "pandas": "pandas",
    "numpy": "numpy",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "sklearn": "scikit-learn",
}

for import_name, pip_name in REQUIRED_PACKAGES.items():
    try:
        importlib.import_module(import_name)
    except ImportError:
        print(f"Installing missing package: {pip_name} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name,
                                "--break-system-packages", "--quiet"])

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

sns.set_style("whitegrid")
np.random.seed(42)


# ------------------------------------------------------------
# 2. LOAD DATASET
# ------------------------------------------------------------
# Make sure "train.csv" is in the same folder as this script.
df = pd.read_csv("train.csv")
print("Dataset loaded successfully!")
print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns\n")
print(df.head(), "\n")

TARGET = "Burn Rate"


# ------------------------------------------------------------
# 3. CLEAN DATA
# ------------------------------------------------------------
df = df.dropna(subset=[TARGET]).copy()

# Fill missing numeric values with column median
numeric_cols = ["Designation", "Resource Allocation", "Mental Fatigue Score"]
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

# Encode categorical columns for model training
label_encoders = {}
categorical_cols = ["Gender", "Company Type", "WFH Setup Available"]
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

print("Missing values handled and categorical columns encoded.\n")


# ------------------------------------------------------------
# 4. FEATURES / TARGET SPLIT
# ------------------------------------------------------------
FEATURES = [
    "Gender", "Company Type", "WFH Setup Available",
    "Designation", "Resource Allocation", "Mental Fatigue Score"
]

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ------------------------------------------------------------
# 5. TRAIN MODEL
# ------------------------------------------------------------
model = RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42, n_jobs=-1)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("=" * 50)
print("MODEL TRAINING COMPLETE (Random Forest Regressor)")
print(f"Mean Absolute Error : {mae:.4f}")
print(f"R^2 Score           : {r2:.4f}")
print("=" * 50, "\n")


# ------------------------------------------------------------
# 6. INTERPRET BURN RATE
# ------------------------------------------------------------
def interpret_burnout(score):
    if score < 0.35:
        return "LOW risk of burnout"
    elif score < 0.65:
        return "MODERATE risk of burnout"
    else:
        return "HIGH risk of burnout"


# ------------------------------------------------------------
# 7. LIVE USER INPUT -> PREDICTION
# ------------------------------------------------------------
def get_user_input_and_predict():
    print("--- ENTER EMPLOYEE DETAILS FOR BURNOUT PREDICTION ---")
    try:
        gender = input("Gender (Male/Female): ").strip().capitalize()
        company_type = input("Company Type (Service/Product): ").strip().capitalize()
        wfh = input("WFH Setup Available (Yes/No): ").strip().capitalize()
        designation = float(input("Designation level (0-5): "))
        resource_allocation = float(input("Resource Allocation (1-10): "))
        mental_fatigue = float(input("Mental Fatigue Score (0-10): "))

        user_data = pd.DataFrame([{
            "Gender": label_encoders["Gender"].transform([gender])[0],
            "Company Type": label_encoders["Company Type"].transform([company_type])[0],
            "WFH Setup Available": label_encoders["WFH Setup Available"].transform([wfh])[0],
            "Designation": designation,
            "Resource Allocation": resource_allocation,
            "Mental Fatigue Score": mental_fatigue
        }])

        user_scaled = scaler.transform(user_data)
        prediction = model.predict(user_scaled)[0]
        prediction = float(np.clip(prediction, 0, 1))

        print("\n" + "=" * 50)
        print("PREDICTION RESULT")
        print(f"Predicted Burn Rate : {prediction:.3f}  (0 = none, 1 = severe)")
        print(f"Risk Category       : {interpret_burnout(prediction)}")
        print("=" * 50)

        raw_user_row = {
            "Gender": gender, "Company Type": company_type, "WFH Setup Available": wfh,
            "Designation": designation, "Resource Allocation": resource_allocation,
            "Mental Fatigue Score": mental_fatigue
        }
        return raw_user_row, prediction

    except (ValueError, KeyError) as e:
        print(f"Invalid input ({e}). Please match the expected categories/ranges exactly.")
        return None, None


# ------------------------------------------------------------
# 8. EXPLORATORY DATA ANALYSIS (EDA) VISUALIZATIONS
# ------------------------------------------------------------
def plot_eda_and_results(user_row=None, user_prediction=None):
    # Create a large figure with multiple subplots to show all EDA requirements
    fig = plt.figure(figsize=(16, 12))

    # 1. HISTOGRAM: Distribution of Burn Rate
    ax1 = plt.subplot(3, 2, 1)
    sns.histplot(df[TARGET], kde=True, color="skyblue", bins=30, ax=ax1)
    ax1.set_title("Histogram: Distribution of Burn Rate")
    ax1.set_xlabel("Burn Rate")
    ax1.set_ylabel("Frequency")

    # 2. PIE CHART: Proportion of Gender
    ax2 = plt.subplot(3, 2, 2)
    gender_original_labels = label_encoders["Gender"].inverse_transform(df["Gender"])
    gender_counts = pd.Series(gender_original_labels).value_counts()
    ax2.pie(gender_counts, labels=gender_counts.index, autopct="%1.1f%%", colors=["#ff9999", "#66b3ff"], startangle=90)
    ax2.set_title("Pie Chart: Gender Distribution")

    # 3. BAR CHART: Employee Count by Designation Level
    ax3 = plt.subplot(3, 2, 3)
    designation_counts = df["Designation"].value_counts().sort_index()
    designation_counts.plot(kind="bar", color="teal", ax=ax3, edgecolor="black")
    ax3.set_title("Bar Chart: Employee Count by Designation")
    ax3.set_xlabel("Designation Level")
    ax3.set_ylabel("Count")

    # 4. CORRELATION MATRIX: Heatmap of Numeric Features & Target
    ax4 = plt.subplot(3, 2, 4)
    # Include numeric features and target for correlation
    corr_cols = ["Designation", "Resource Allocation", "Mental Fatigue Score", TARGET]
    corr_matrix = df[corr_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax4)
    ax4.set_title("Correlation Matrix Heatmap")

    # 5. BOX PLOT: Burn Rate Distribution by Company Type
    ax5 = plt.subplot(3, 2, 5)
    plot_df = df.copy()
    plot_df["Company Type Label"] = label_encoders["Company Type"].inverse_transform(plot_df["Company Type"])
    sns.boxplot(x="Company Type Label", y=TARGET, data=plot_df, palette="Set2", ax=ax5)
    ax5.set_title("Box Plot: Burn Rate by Company Type")
    ax5.set_xlabel("Company Type")
    ax5.set_ylabel("Burn Rate")

    # 6. SCATTER PLOT: Mental Fatigue vs Burn Rate (with User Prediction marker)
    ax6 = plt.subplot(3, 2, 6)
    sc = ax6.scatter(
        df["Mental Fatigue Score"], df[TARGET],
        c=df["Resource Allocation"], cmap="viridis", alpha=0.5, edgecolor="k", linewidth=0.2
    )
    ax6.set_title("Scatter Plot: Mental Fatigue vs Burn Rate\n(Color = Resource Allocation)")
    ax6.set_xlabel("Mental Fatigue Score")
    ax6.set_ylabel("Burn Rate")
    fig.colorbar(sc, ax=ax6, label="Resource Allocation")

    if user_row is not None and user_prediction is not None:
        ax6.scatter(
            user_row["Mental Fatigue Score"], user_prediction,
            color="red", s=250, marker="*", edgecolor="black", linewidth=1.5,
            label="Your Prediction", zorder=5
        )
        ax6.legend()

    plt.tight_layout()
    output_path = "employee_burnout_eda_dashboard.png"
    plt.savefig(output_path, dpi=150)
    print(f"\nComprehensive EDA Dashboard saved to: {output_path}")
    plt.show()


# ------------------------------------------------------------
# 9. MAIN PROGRAM EXECUTION
# ------------------------------------------------------------
if __name__ == "__main__":
    user_row, user_prediction = get_user_input_and_predict()
    plot_eda_and_results(user_row, user_prediction)