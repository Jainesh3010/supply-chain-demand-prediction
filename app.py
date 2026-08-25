from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load("model/supply_chain_model.pkl")

features = [
    "Average_Daily_Sales",
    "Previous_30_Days_Sales",
    "Previous_7_Days_Sales",
    "Current_Inventory",
    "Previous_Day_Sales",
    "Discount_Percentage",
    "Promotion",
    "Customer_Return_Rate",
    "Transportation_Cost",
    "Supplier_Lead_Time_Days",
    "Competitor_Price",
    "Unit_Price",
]

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        data = {}
        for feature in features:
            value = request.form[feature]
            if feature == "Promotion":
                data[feature] = value
            else:
                data[feature] = float(value)

        input_df = pd.DataFrame([data])
        result = model.predict(input_df)[0]
        prediction = round(float(result), 2)

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)