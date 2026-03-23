# 🏥 Insurance Claim Predictor

A machine learning web app that predicts insurance claim amounts based on a patient's health profile. Built with **Streamlit** and trained on multiple regression models including XGBoost, Random Forest, and SVR, automatically selecting the best performer.

---

## 🚀 Live Demo
https://insurancepredictor1879.streamlit.app/


---

## 🧠 How It Works

1. User enters health details (age, BMI, blood pressure, smoking status, etc.)
2. Inputs are preprocessed using saved label encoders and a standard scaler
3. The best-trained ML model predicts the estimated claim amount
4. The app displays the result with a risk tier (Low / Moderate / High)

---

## 📊 Models Trained

| Model | Description |
|---|---|
| Linear Regression | Baseline model |
| Polynomial Regression | Captures non-linear relationships |
| Random Forest | Ensemble of decision trees with GridSearchCV tuning |
| SVR | Support Vector Regressor with kernel search |
| XGBoost | Gradient boosting with hyperparameter tuning |

The best model by R² score is automatically saved and used in the app.

---

## 🗂️ Project Structure

```
Insurance_Predictor/
├── app.py                        # Streamlit frontend
├── train.py                      # Model training script
├── insurance.csv                 # Dataset
├── requirements.txt              # Python dependencies
├── best_model.pkl                # Saved best model
├── scaler.pkl                    # Saved StandardScaler
├── gender_label_encoder.pkl      # Label encoder for gender
├── diabetic_label_encoder.pkl    # Label encoder for diabetic
└── smoker_label_encoder.pkl      # Label encoder for smoker
```

---

## ⚙️ Features

- **Multi-model training** with automatic best-model selection
- **Hyperparameter tuning** via GridSearchCV
- **Label encoding** for categorical features
- **Standard scaling** for numerical features
- **Risk tier classification** — Low / Moderate / High based on predicted amount
- **Clean dark UI** built with custom Streamlit CSS

---

## 🛠️ Installation & Running Locally

**1. Clone the repository:**
```bash
git clone https://github.com/Aayush1879/Insurance_Predictor.git
cd Insurance_Predictor
```

**2. Install dependencies:**
```bash
pip install -r requirements.txt
```

**3. Train the model (generates all `.pkl` files):**
```bash
python train.py
```

**4. Run the app:**
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## 📦 Requirements

```
streamlit
joblib
numpy
pandas
scikit-learn
xgboost
optuna
matplotlib
seaborn
```

---

## 📁 Dataset

The dataset (`insurance.csv`) contains the following features:

| Feature | Type | Description |
|---|---|---|
| age | Numeric | Age of the patient |
| gender | Categorical | Male / Female |
| bmi | Numeric | Body Mass Index |
| bloodpressure | Numeric | Blood pressure reading |
| children | Numeric | Number of dependents |
| diabetic | Categorical | Yes / No |
| smoker | Categorical | Yes / No |
| region | Categorical | Geographic region |
| claim | Numeric | Insurance claim amount (target) |

---

## 📈 Results

| Model | R² Score |
|---|---|
| XGBoost | ~0.87 |
| Random Forest | ~0.85 |
| SVR | ~0.78 |
| Polynomial Regression | ~0.72 |
| Linear Regression | ~0.65 |


