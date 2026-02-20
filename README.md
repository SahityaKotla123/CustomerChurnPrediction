# 📡 Customer Churn Prediction & Intervention System

A complete end-to-end machine learning project that predicts customer churn for a telecom company, segments customers into behavioral groups, and recommends targeted interventions with ROI analysis.

---

## 📌 Project Overview

Customer churn is one of the biggest challenges in the telecom industry. This project builds a full pipeline that:

1. **Predicts** which customers are likely to churn using classification models
2. **Segments** customers into meaningful groups using K-Means clustering
3. **Recommends** personalized interventions (discounts, support, tutorials) per segment
4. **Calculates** the expected ROI of each intervention strategy

---

## 🗂️ Project Structure

```
CustomerChurnPrediction/
│
├── TelecomChurn.ipynb           # Main notebook (EDA → Modeling → Clustering → ROI)
├── app.py                       # Streamlit/Flask app for predictions
│
├── Telco-Customer-Churn.csv     # Raw dataset
├── cleaned_telco_data.csv       # Preprocessed data
├── segmented_telco_data.csv     # Data with customer segments
├── intervention_results.csv     # Per-customer intervention outcomes
├── intervention_summary.csv     # Aggregate intervention summary
├── segment_profiles.csv         # Segment characteristics
├── segment_roi.csv              # ROI breakdown by segment
├── model_comparison.csv         # Model metrics comparison
│
├── churn_model.pkl              # Best trained model
├── logistic_regression_model.pkl
├── random_forest_model.pkl
├── xgboost_model.pkl
├── kmeans_model.pkl             # Customer segmentation model
├── scaler.pkl                   # Feature scaler (for prediction)
├── clustering_scaler.pkl        # Scaler for clustering
├── feature_names.pkl            # Feature list for inference
└── segment_names.pkl            # Segment label mapping
```

---

## 🔬 Dataset

**Source:** [Telco Customer Churn – Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

- **Rows:** 7,043 customers
- **Features:** 21 (demographics, service subscriptions, billing info)
- **Target:** `Churn` (Yes/No)
- **Churn Rate:** ~26.5%

**Key Features:**
- `tenure` – How long the customer has been with the company
- `MonthlyCharges` – Monthly bill amount
- `Contract` – Month-to-month, One year, Two year
- `InternetService` – DSL, Fiber optic, None
- `PaymentMethod` – Electronic check, Credit card, etc.

---

## ⚙️ Workflow

### Step 1 – Data Preprocessing
- Removed `customerID` (non-informative)
- Converted `TotalCharges` from object to numeric
- Filled missing values (new customers with 0 tenure)
- Encoded `Churn` as binary (1 = Yes, 0 = No)
- Applied `LabelEncoder` to categorical columns

### Step 2 – Exploratory Data Analysis (EDA)
- Churn distribution (pie chart + bar chart)
- Numerical features (tenure, MonthlyCharges, TotalCharges) vs. Churn (boxplots)
- Categorical features vs. Churn rate (grouped bar charts)
- Key business insights extracted from patterns

### Step 3 – Model Training & Hyperparameter Tuning
Three classification models trained with `GridSearchCV` (5-fold CV, scored on ROC-AUC):

| Model | Technique |
|---|---|
| Logistic Regression | L2 regularization, balanced class weights |
| Random Forest | Balanced class weights, tuned depth/splits |
| XGBoost | `scale_pos_weight` for class imbalance |

### Step 4 – Model Evaluation
Each model evaluated on:
- Accuracy, Precision, Recall, F1 Score, ROC-AUC
- Confusion Matrix
- ROC Curve

### Step 5 – Customer Segmentation (K-Means)
- Optimal cluster count selected using **Elbow Method** + **Silhouette Score**
- 4 segments identified and named based on characteristics:

| Segment | Description |
|---|---|
| At-Risk Newcomers | New customers with high churn risk |
| Loyal Champions | Long-tenure customers with low churn |
| High-Value Users | High spenders with multiple services |
| Budget Conscious | Low spenders, fewer services |

### Step 6 – Targeted Interventions & ROI Analysis
Interventions assigned based on segment profile:

| Intervention | Cost | Churn Reduction |
|---|---|---|
| Feature Tutorial | $15/customer | 20% |
| Premium Support | $50/customer | 25% |
| 10% Discount (3 months) | ~Variable | 15% |
| 20% Discount (3 months) | ~Variable | 30% |
| Loyalty Reward | $30/customer | 18% |

ROI calculated as:
```
Net Benefit = Prevented Churn Loss - Intervention Cost
ROI (%) = Net Benefit / Intervention Cost × 100
```

---

## 🤖 Models & Results

> Results are approximate based on typical runs. Exact scores are saved in `model_comparison.csv`.

| Model | Accuracy | ROC-AUC | Recall (Churn) |
|---|---|---|---|
| Logistic Regression | ~80% | ~0.84 | ~0.76 |
| Random Forest | ~79% | ~0.83 | ~0.74 |
| XGBoost | ~80% | ~0.85 | ~0.75 |

✅ **Best Model:** XGBoost (highest ROC-AUC), saved as `churn_model.pkl`

---

## 🛠️ Tech Stack

- **Language:** Python 3.x
- **Data Processing:** `pandas`, `numpy`
- **Visualization:** `matplotlib`, `seaborn`
- **Modeling:** `scikit-learn`, `xgboost`
- **Clustering:** `KMeans`, `PCA`
- **Model Saving:** `joblib`
- **App:** `app.py` (Streamlit / Flask)

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/SahityaKotla123/CustomerChurnPrediction.git
cd CustomerChurnPrediction
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Notebook
Open `TelecomChurn.ipynb` in Jupyter Notebook or VS Code and run all cells.

### 4. Run the App
```bash
python app.py
```

---

## 📦 Requirements

Create a `requirements.txt` with:
```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
joblib
```

---

## 📊 Key Business Insights

- Customers on **month-to-month contracts** churn at ~3x the rate of those on 2-year contracts
- **Fiber optic internet** users churn more than DSL users despite paying more
- **Short tenure** (< 12 months) is the strongest early churn indicator
- High **MonthlyCharges** combined with low service engagement signals at-risk customers
- Targeted interventions can generate a **positive ROI** when applied to correctly identified high-risk segments

---

## 🙋 Author

**Sahitya Kotla**  
[GitHub](https://github.com/SahityaKotla123)

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
