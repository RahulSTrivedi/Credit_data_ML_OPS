# MLOPS Project: Credit Risk Classification  

This project implements an end-to-end MLOps pipeline for classifying customer credit ratings as good or bad. It covers data processing, model training with MLflow, deployment via FastAPI, and monitoring for model performance.  

---

## Project Overview  

- **Data Processing & Validation:**  
  The dataset is cleaned, validated using Pydantic, and split into training and testing sets for model training.  

- **Model Training & Experiment Tracking:**  
  A classification model is trained using scikit-learn, and multiple experiments are tracked using MLflow. The best model is selected based on evaluation metrics.  

- **Model Deployment:**  
  The best model is deployed using FastAPI, providing an API endpoint for credit risk predictions.  

- **Model Monitoring:**  
  The performance of the deployed model is monitored by tracking incoming data distributions and model predictions to detect potential data drift.  

---

## Setup Instructions  

### Clone the Repository  

```bash
git clone https://github.com/<username>/Credit_Risk_MLOps.git
cd Credit_Risk_MLOps
```

## Running the Notebooks and Components  

### Notebook1: Data Processing & Experiment Tracking  

- **Content:**  
  Contains data preprocessing steps, feature engineering, and model training with MLflow experiment tracking.  
- **Usage:**  
  Open and run `Notebook1.ipynb` in Jupyter Notebook to reproduce experiments.  

### Notebook2: Model Deployment using FastAPI  

- **Content:**  
  Contains FastAPI code to deploy the trained model as an API service.  
- **Converted to Python File:**  
  The code is available as `Notebook2.py` for execution.  
- **Run the FastAPI App:**  
  ```bash
  uvicorn Notebook2:app --port 8000 --reload
  ```  

### Notebook3: Model Monitoring  

- **Content:**  
  Implements monitoring techniques to track data drift and prediction distributions over time.  
- **Usage:**  
  Run `Notebook3.ipynb` to analyze model performance on incoming data.  

### Notebook4: User Interface with Streamlit  

- **Content:**  
  Contains the Streamlit UI code for collecting user inputs and displaying predictions.  
- **Converted to Python File:**  
  The code has been converted to `Notebook4.py`.  
- **Run the Streamlit UI:**  
  ```bash
  cd Credit_Risk_MLOps
  streamlit run Notebook4.py
  ```  

### Notebook5: Model Monitoring (Data Drift Detection)  

- **Content:**  
  Uses alibi-detect and chi-square tests to monitor numeric and categorical drift.  
- **Usage:**  
  Run `Notebook5.ipynb` in your Jupyter environment to view drift detection results in tabular format.

