"""
Supply Chain Demand Prediction - Flask Backend
=============================================
Simple Flask application that:
  1. Serves the home page (index.html)
  2. Handles the prediction form (prediction.html)
  3. Loads the pre-trained ML model using joblib
  4. Returns the predicted demand to the Jinja2 template
"""

from flask import Flask, render_template, request
import joblib
import numpy as np
import os
import pandas as pd

# -- App Setup ---------------------------------------------------------------
app = Flask(__name__)

# -- Load Model --------------------------------------------------------------
MODEL_PATH = os.path.join("model", "supply_chain_model.pkl")

# Load the trained Random Forest pipeline once at startup
model = joblib.load(MODEL_PATH)

# -- Routes ------------------------------------------------------------------

@app.route("/")
def index():
    """Render the home / landing page."""
    return render_template("index.html")


@app.route("/prediction", methods=["GET", "POST"])
def prediction():
    """
    GET  -> Show the empty prediction form.
    POST -> Read 12 feature inputs, run ML model, show the predicted demand.
    """
    predicted_demand = None
    error = None

    if request.method == "POST":
        try:
            # -- Read the 12 input features from the form ------------------
            # IMPORTANT: Keep this order exactly as the model was trained on.
            avg_daily_sales        = float(request.form["avg_daily_sales"])
            prev_30_days_sales     = float(request.form["prev_30_days_sales"])
            prev_7_days_sales      = float(request.form["prev_7_days_sales"])
            current_inventory      = float(request.form["current_inventory"])
            prev_day_sales         = float(request.form["prev_day_sales"])
            discount_percentage    = float(request.form["discount_percentage"])
            # Promotion: "Yes" -> 1, "No" -> 0
            promotion              = 1 if request.form["promotion"] == "Yes" else 0
            customer_return_rate   = float(request.form["customer_return_rate"])
            transportation_cost    = float(request.form["transportation_cost"])
            supplier_lead_time     = float(request.form["supplier_lead_time"])
            competitor_price       = float(request.form["competitor_price"])
            unit_price             = float(request.form["unit_price"])

            # -- Build feature array (same order as training) --------------
            features = pd.DataFrame([{
            "Average_Daily_Sales": avg_daily_sales,
            "Previous_30_Days_Sales": prev_30_days_sales,
            "Previous_7_Days_Sales": prev_7_days_sales,
            "Current_Inventory": current_inventory,
            "Previous_Day_Sales": prev_day_sales,
            "Discount_Percentage": discount_percentage,
            "Promotion": promotion,
            "Customer_Return_Rate": customer_return_rate,
            "Transportation_Cost": transportation_cost,
            "Supplier_Lead_Time_Days": supplier_lead_time,
            "Competitor_Price": competitor_price,
            "Unit_Price": unit_price
        }])

            # -- Predict ---------------------------------------------------
            prediction_value = model.predict(features)[0]
            predicted_demand = round(float(prediction_value), 2)

        except Exception as e:
            error = f"Prediction failed: {str(e)}"

    return render_template("prediction.html",
                           predicted_demand=predicted_demand,
                           error=error)


# -- Run ---------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
