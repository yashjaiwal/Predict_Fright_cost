import mlflow
import mlflow.sklearn
import joblib

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from mlflow.models import infer_signature

from data_preprocessing import load_invoice_data, apply_label, split_data, scaled_fetures
from model_eval import train_XGB

# ✅ Corrected feature names
features = [
    'invioce_quantity',
    'invoice_dollar',
    'Freight',
    'total_item_quantity',
    'total_item_dollars',
    'avg_receiving_delay'
]

target = "flag_invoice"

# 🔹 Use tracking server
mlflow.set_tracking_uri("http://localhost:5000")

def main():

    mlflow.set_experiment("Invoice_Risk_Prediction")

    # 🔥 Auto logging (recommended)
    mlflow.sklearn.autolog()

    with mlflow.start_run():

        df = load_invoice_data()
        if df.empty:
            raise ValueError("Invoice data is empty!")

        apply_label(df)

        X_train, X_test, y_train, y_test = split_data(df, features, target)
        X_train_scaled, X_test_scaled = scaled_fetures(X_train, X_test)

        # 🔹 Train
        grid = train_XGB(X_train_scaled, y_train)
        best_model = grid.best_estimator_

        # 🔹 Predict
        y_pred = best_model.predict(X_test_scaled)

        # 🔹 Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)

        # 🔹 Signature (deployment ready)
        signature = infer_signature(X_train_scaled, best_model.predict(X_train_scaled))

        # 🔹 Log model (UPDATED API ✅)
        mlflow.sklearn.log_model(
            sk_model=best_model,
            name="invoice_model",
            signature=signature
        )

        # 🔹 Save locally
        joblib.dump(best_model, "model/predict_flag_invoice.pkl")

        print("✅ Model training complete and logged in MLflow")

if __name__ == "__main__":
    main()
