import pickle
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
# Define the expected schema based on your model input
FEATURES = [
    "checking_account_status", "duration", "credit_history", "purpose", "credit_amount",
    "savings_account", "employment_status", "installment_rate", "personal_status_sex",
    "other_debtors", "present_residence", "property", "age", "other_installment_plans",
    "housing", "number_of_existing_credits", "job", "dependents", "telephone", "foreign_worker"
]
TARGET = "credit_risk"  # Assuming the target column name

# Load the trained model
with open("D:\Rahul\jio-files\AIDS\Quarter 4\ML-ops\ML_ops_project\Gr-06_MLOPS_Project\Project Code\Gr-06_Notebooks\model.pkl", "rb") as file:
    model = pickle.load(file)

# Load production data from Parquet
prod_data = pd.read_parquet("Gr-06_MLOPS_Project/Project Code/Datasets/Processed/credit_data_test.parquet")

# Validate if required features exist in the dataset
missing_features = [col for col in FEATURES if col not in prod_data.columns]
if missing_features:
    raise ValueError(f"Missing required features in production data: {missing_features}")

# Extract features and target
X_prod = prod_data[FEATURES]
y_true = prod_data[TARGET] if TARGET in prod_data.columns else None  # Handle case where target is absent

# Make predictions
y_pred = model.predict(X_prod)
y_prob = model.predict_proba(X_prod)[:, 1]  # Probability for positive class

# Calculate evaluation metrics only if target labels exist
if y_true is not None:
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="binary")
    recall = recall_score(y_true, y_pred, average="binary")
    f1 = f1_score(y_true, y_pred, average="binary")
    roc_auc = roc_auc_score(y_true, y_prob)

    # Log metrics to MLflow
    mlflow.set_experiment("Credit Risk Classification")

    with mlflow.start_run():
        mlflow.log_param("Model Type", "Random Forest")  # Change if using another model
        mlflow.log_metric("Accuracy", accuracy)
        mlflow.log_metric("Precision", precision)
        mlflow.log_metric("Recall", recall)
        mlflow.log_metric("F1-Score", f1)
        mlflow.log_metric("AUC-ROC", roc_auc)
        
        # Log the trained model
        mlflow.sklearn.log_model(model, "credit_risk_model")

    print("Metrics logged successfully to MLflow!")
else:
    print("Production data does not contain target labels. Predictions generated, but no metrics logged.")

# Save predictions if needed
prod_data["predicted_credit_risk"] = y_pred
prod_data.to_parquet("predictions.parquet", index=False)
print("Predictions saved to 'predictions.parquet'")
