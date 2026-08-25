# 🚚 Supply Chain Demand Prediction

A Machine Learning based web application that predicts product demand using historical sales, inventory, pricing, promotion, and supply-chain related features.

The project uses a Random Forest Regression model and provides predictions through a simple Flask web application.

---

## 📌 Project Overview

Accurate demand prediction can help businesses improve inventory management, reduce stockouts, and make better supply-chain decisions.

This project predicts:

**Demand Quantity**

using important supply-chain features such as:

- Average Daily Sales
- Previous 30 Days Sales
- Previous 7 Days Sales
- Previous Day Sales
- Current Inventory
- Discount Percentage
- Promotion
- Customer Return Rate
- Transportation Cost
- Supplier Lead Time
- Competitor Price
- Unit Price

---

## 🧠 Machine Learning

### Model

**Random Forest Regressor**

The model is trained using a preprocessing pipeline that handles:

- Missing numerical values
- Missing categorical values
- Numerical scaling
- Categorical encoding

The complete preprocessing + model pipeline is saved using `joblib`.

---

## 📊 Model Performance

| Metric | Score |
|---|---:|
| Train R² | ~0.92 |
| Test R² | **0.8619** |
| MAE | **4.19** |
| RMSE | **6.65** |

The model achieves an R² score of approximately **0.86 on unseen test data**.

---

## 🔍 Exploratory Data Analysis

The dataset was analyzed using:

- Missing value analysis
- Duplicate detection
- Statistical summary
- Target variable distribution
- Correlation analysis
- Numerical feature distributions
- Categorical feature analysis
- Outlier analysis

The target variable, `Demand_Quantity`, showed a right-skewed distribution.

---

## 🎯 Feature Selection

Feature importance analysis was used to identify the most useful features.

The final deployment model uses 12 features:

```text
Average_Daily_Sales
Previous_30_Days_Sales
Previous_7_Days_Sales
Current_Inventory
Previous_Day_Sales
Discount_Percentage
Promotion
Customer_Return_Rate
Transportation_Cost
Supplier_Lead_Time_Days
Competitor_Price
Unit_Price