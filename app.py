import streamlit as st
import requests

# ---------------------------
# API URL (FastAPI)
# ---------------------------
API_URL = "http://127.0.0.1:8000"

# ---------------------------
# Sidebar
# ---------------------------

st.sidebar.title("⚙️ Prediction Settings")

st.sidebar.write("""
Select the prediction model you want to use.

📦 **Freight Prediction** – Estimate transportation or shipping cost.

⚠️ **Invoice Risk Prediction** – Detect risky invoices.
""")

model_option = st.sidebar.selectbox(
    "Choose Prediction Model",
    ["Freight Prediction", "Invoice Risk Prediction"]
)

# ---------------------------
# App Title
# ---------------------------

st.title("📦 Invoice Risk & Freight Prediction")

st.info("""
This AI-powered app.

📦 Freight Cost = Shipping cost of goods  
⚠️ Invoice Risk = Detect suspicious invoices
""")

# ---------------------------
# Freight Prediction
# ---------------------------

if model_option == "Freight Prediction":

    st.header("📦 Freight Cost Prediction")

    dollars = st.number_input("Invoice Dollars ($)", min_value=0.0)


    if st.button("Predict Freight Cost"):

        try:
            response = requests.post(
                f"{API_URL}/predict_freight",
                json={"invoice_dollar": dollars}
            )

            result = response.json()

            st.success(f"📦 Predicted Freight Cost: ${result['predicted_freight_cost']}")

        except:
            st.error("❌ FastAPI server not running!")

# ---------------------------
# Invoice Risk Prediction
# ---------------------------

if model_option == "Invoice Risk Prediction":

    st.header("⚠️ Invoice Risk Prediction")

    quantity = st.number_input("Invoice Quantity", min_value=0)
    dollars = st.number_input("Invoice Dollars ($)", min_value=0.0)
    freight = st.number_input("Freight Cost ($)", min_value=0.0)
    
    total_item_quantity = st.number_input("Total Item Quantity", min_value=0)
    total_item_dollars = st.number_input("Total Item Dollars ($)", min_value=0.0)
    avg_receiving_delay = st.number_input("Avg Receiving Delay (Days)", min_value=0)

    if st.button("Predict Invoice Risk"):

        try:
            response = requests.post(
                f"{API_URL}/predict_invoice_risk",
                json={
                    "invioce_quantity": quantity,
                    "invoice_dollar": dollars,
                    "Freight": freight,
                    "total_item_quantity": total_item_quantity,
                    "total_item_dollars": total_item_dollars,
                    "avg_receiving_delay": avg_receiving_delay
                }
            )

            result = response.json()

            if result["invoice_flag"] == 1:
                st.error("⚠️ Invoice Likely to be Flagged")
            else:
                st.success("✅ Invoice Looks Safe")

        except:
            st.error("❌ FastAPI server not running!")