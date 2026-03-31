from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

# ---------------------------
# Initialize App
# ---------------------------
app = FastAPI(title="Invoice Prediction API")

# ---------------------------
# Load Models (once at startup)
# ---------------------------
freight_model = joblib.load("model/best_freight_model.pkl")
flag_model = joblib.load("model/predict_flag_invoice.pkl")
# ---------------------------
# Request Schemas
# ---------------------------

class FreightInput(BaseModel):
    invoice_dollar: float

class InvoiceRiskInput(BaseModel):
    invioce_quantity: int
    invoice_dollar: float
    Freight: float
    total_item_quantity: int
    total_item_dollars: float
    avg_receiving_delay: int

# ---------------------------
# Routes
# ---------------------------

@app.get("/")
def home():
    return {"message": "🚀 Invoice Prediction API is running"}

# ---------------------------
# Freight Prediction API
# ---------------------------

@app.post("/predict_freight")
def predict_freight(data: FreightInput):

    input_df = pd.DataFrame({
        "Dollars": [data.invoice_dollar]
    })

    prediction = freight_model.predict(input_df)[0]

    return {
        "predicted_freight_cost": round(float(prediction), 2)
    }

# ---------------------------
# Invoice Risk Prediction API
# ---------------------------

@app.post("/predict_invoice_risk")
def predict_invoice_risk(data: InvoiceRiskInput):

    input_df = pd.DataFrame({
        "invioce_quantity": [data.invioce_quantity],  # keep same as training
        "invoice_dollar": [data.invoice_dollar],
        "Freight": [data.Freight],
        "total_item_quantity": [data.total_item_quantity],
        "total_item_dollars": [data.total_item_dollars],
        "avg_receiving_delay": [data.avg_receiving_delay]
    })

    prediction = flag_model.predict(input_df)[0]

    return {
        "invoice_flag": int(prediction),
        "message": "⚠️ Likely Flagged" if prediction == 1 else "✅ Safe Invoice"
    }
