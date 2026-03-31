import joblib
import pandas as pd
import os
import mlflow
import mlflow.sklearn
from datetime import datetime
 
# Import preprocessing functions
from data_preprocessing import load_vendore_invoice, prepare_feature, split_data
 
# Import model functions
from model_eval import (
    train_linear_regression,
    train_random_forest,
    train_xgboost,
    evaluate_model
)
 
# ============ MLflow Configuration ============
MLFLOW_TRACKING_URI = "http://localhost:5000"  # Change if using remote server
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
 
# ============ TRAINING FUNCTION WITH MLflow ============
def main():
    
    # 1️⃣ Set MLflow Experiment
    experiment_name = "Freight Cost Prediction"
    mlflow.set_experiment(experiment_name)
    
    # 2️⃣ Start MLflow Run
    with mlflow.start_run(run_name=f"All_Models_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        
        # 3️⃣ Database path
        data_path = (r"C:\Users\User\Desktop\ML project\Data\inventory.db")
        
        # Log data path
        mlflow.log_param("data_path", data_path)
        
        # 4️⃣ Load data
        df = load_vendore_invoice(data_path)
        
        # Log dataset info
        mlflow.log_param("dataset_rows", len(df))
        mlflow.log_param("dataset_columns", len(df.columns))
        
        # 5️⃣ Prepare features
        X, y = prepare_feature(df)
        
        mlflow.log_param("feature_count", X.shape[1])
        mlflow.log_param("target_variable", "cost")  # Change if needed
        
        # 6️⃣ Split data
        X_train, X_test, y_train, y_test = split_data(X, y)
        
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        mlflow.log_param("test_split_ratio", 0.2)  # Adjust if different
        
        print("\n" + "="*60)
        print("🚀 STARTING MODEL TRAINING WITH MLflow TRACKING")
        print("="*60)
        
        # ============ TRAIN ALL MODELS ============
        
        # Train Linear Regression
        print("\n📈 Training Linear Regression...")
        lr = train_linear_regression(X_train, y_train)
        
        # Train Random Forest
        print("🌲 Training Random Forest...")
        rf = train_random_forest(X_train, y_train)
        
        # Train XGBoost
        print("⚡ Training XGBoost...")
        xgb = train_xgboost(X_train, y_train)
        
        # ============ EVALUATE ALL MODELS ============
        
        print("\n" + "="*60)
        print("📊 EVALUATING MODELS")
        print("="*60)
        
        lr_name, lr_mse, lr_mae, lr_r2 = evaluate_model(lr, X_test, y_test)
        rf_name, rf_mse, rf_mae, rf_r2 = evaluate_model(rf, X_test, y_test)
        xgb_name, xgb_mse, xgb_mae, xgb_r2 = evaluate_model(xgb, X_test, y_test)
        
        # Store results
        results = [
            {
                "Model": lr_name, 
                "MSE": lr_mse, 
                "MAE": lr_mae, 
                "R2": lr_r2, 
                "Model_Obj": lr
            },
            {
                "Model": rf_name, 
                "MSE": rf_mse, 
                "MAE": rf_mae, 
                "R2": rf_r2, 
                "Model_Obj": rf
            },
            {
                "Model": xgb_name, 
                "MSE": xgb_mse, 
                "MAE": xgb_mae, 
                "R2": xgb_r2, 
                "Model_Obj": xgb
            },
        ]
        
        results_df = pd.DataFrame(results)
        
        print("\n" + results_df[['Model', 'MSE', 'MAE', 'R2']].to_string(index=False))
        
        # ============ LOG METRICS FOR EACH MODEL ============
        
        # Log LR metrics
        with mlflow.start_run(run_name="LinearRegression", nested=True):
            mlflow.log_metric("MSE", lr_mse)
            mlflow.log_metric("MAE", lr_mae)
            mlflow.log_metric("R2_Score", lr_r2)
            mlflow.set_tag("model_type", "LinearRegression")
            mlflow.sklearn.log_model(lr, "linear_regression_model")
        
        # Log RF metrics
        with mlflow.start_run(run_name="RandomForest", nested=True):
            mlflow.log_metric("MSE", rf_mse)
            mlflow.log_metric("MAE", rf_mae)
            mlflow.log_metric("R2_Score", rf_r2)
            mlflow.set_tag("model_type", "RandomForest")
            mlflow.sklearn.log_model(rf, "random_forest_model")
        
        # Log XGB metrics
        with mlflow.start_run(run_name="XGBoost", nested=True):
            mlflow.log_metric("MSE", xgb_mse)
            mlflow.log_metric("MAE", xgb_mae)
            mlflow.log_metric("R2_Score", xgb_r2)
            mlflow.set_tag("model_type", "XGBoost")
            mlflow.sklearn.log_model(xgb, "xgboost_model")
        
        # ============ SELECT BEST MODEL ============
        
        best_model_row = results_df.loc[results_df["MAE"].idxmin()]
        best_model_name = best_model_row["Model"]
        best_model = best_model_row["Model_Obj"]
        best_mae = best_model_row["MAE"]
        best_r2 = best_model_row["R2"]
        
        print("\n" + "="*60)
        print("🏆 BEST MODEL SELECTED")
        print("="*60)
        print(f"Model: {best_model_name}")
        print(f"MAE: {best_mae:.4f}")
        print(f"R2 Score: {best_r2:.4f}")
        print("="*60)
        
        # ============ LOG BEST MODEL TO MLflow ============
        
        mlflow.log_metric("best_model_MAE", best_mae)
        mlflow.log_metric("best_model_R2", best_r2)
        mlflow.set_tag("best_model", best_model_name)
        
        # Log best model with registration
        mlflow.sklearn.log_model(
            best_model,
            "best_freight_cost_model",
            registered_model_name="FreightCostModel"
        )
        
        # ============ SAVE BEST MODEL TO DISK ============
        
        os.makedirs("model", exist_ok=True)
        model_path = "model/best_freight_model.pkl"
        joblib.dump(best_model, model_path)
        
        # Log saved model path
        mlflow.log_artifact(model_path)
        mlflow.log_param("saved_model_path", model_path)
        
        print(f"\n✅ Best model saved to: {model_path}")
        
        # ============ LOG COMPARISON RESULTS ============
        
        # Save results to CSV and log it
        results_csv = "model/model_comparison.csv"
        results_df[['Model', 'MSE', 'MAE', 'R2']].to_csv(results_csv, index=False)
        mlflow.log_artifact(results_csv)
        
        print(f"✅ Results saved to: {results_csv}")
        
        # ============ SET TAGS ============
        
        mlflow.set_tag("experiment_type", "model_comparison")
        mlflow.set_tag("dataset", "inventory")
        mlflow.set_tag("task", "regression")
        mlflow.set_tag("models_trained", 3)
        
        print("\n✅ MLflow tracking completed!")
        print(f"📊 View results: {MLFLOW_TRACKING_URI}")
        
        return best_model
 
 
if __name__ == "__main__":
    best_model = main()