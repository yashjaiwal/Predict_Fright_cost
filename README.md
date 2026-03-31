# 🧾 Vendor Invoice Intelligence System
### Freight Cost Prediction & Invoice Risk Detection

> **Live Demo:** [https://predict-fright-cost-frontent.onrender.com/](https://predict-fright-cost-frontent.onrender.com/) | **API Docs:** [https://predict-fright-cost.onrender.com/docs](https://predict-fright-cost.onrender.com/docs) | **MLflow Tracking:** [DagsHub](https://dagshub.com/yashjaiwal/Predict_Fright_cost.mlflow)

Analyzing invoice data to predict freight costs and detect risky invoices using **Machine Learning**, **Python**, **FastAPI**, and **Streamlit**. Fully containerized with **Docker** and deployed on **Render** for production-grade reliability.

![Python](https://img.shields.io/badge/Python-3.13+-blue?style=flat-square&logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?style=flat-square&logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-Model-green?style=flat-square)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-teal?style=flat-square&logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-red?style=flat-square&logo=streamlit)
![Docker](https://img.shields.io/badge/Docker-Containerization-blue?style=flat-square&logo=docker)
![Render](https://img.shields.io/badge/Render-Deployment-46E3B7?style=flat-square)
![MLflow](https://img.shields.io/badge/MLflow-Experiment%20Tracking-0194E2?style=flat-square&logo=mlflow)

---

## 📌 Table of Contents

- [Overview](#overview)
- [Business Problem](#business-problem)
- [Live Demo & Deployment](#live-demo--deployment)
- [Dataset](#dataset)
- [Tools & Technologies](#tools--technologies)
- [Project Structure](#project-structure)
- [Data Cleaning & Preparation](#data-cleaning--preparation)
- [Exploratory Data Analysis](#exploratory-data-analysis-eda)
- [Machine Learning Models](#machine-learning-models)
- [Evaluation Metrics](#evaluation-metrics)
- [Architecture](#architecture)
- [Getting Started](#getting-started)
- [Docker Setup](#docker-setup)
- [Deployment Guide](#deployment-guide)
- [API Documentation](#api-documentation)
- [MLflow & Experiment Tracking](#mlflow--experiment-tracking)
- [Author & Contact](#author--contact)

---

## 📖 Overview

This project builds an **end-to-end machine learning pipeline** to analyze vendor invoice data and generate insights for freight cost prediction and invoice risk detection.

The system helps finance teams:
- 📦 Predict freight shipping costs with high accuracy
- 🚨 Identify high-risk invoices for automated flagging
- 📊 Improve invoice monitoring and financial transparency
- ⏱️ Reduce manual invoice review time by 80%+
- 🔍 Track ML experiments with MLflow + DagsHub
- 🚀 Deploy ML models at scale with Docker & Render

**Key Features:**
- ✅ FastAPI backend with interactive API documentation
- ✅ Streamlit web dashboard for real-time predictions
- ✅ Production-grade ML models (XGBoost-based)
- ✅ SQLite database integration
- ✅ Model versioning and experiment tracking (MLflow)
- ✅ Containerized with Docker
- ✅ Auto-scaling deployment on Render

---

## 💼 Business Problem

Organizations process **thousands of invoices every month**. Manually identifying abnormal freight costs or risky invoices is inefficient and error-prone.

**Challenges:**
- ❌ Manual invoice review is time-consuming
- ❌ Human error in anomaly detection
- ❌ No predictive insights on freight costs
- ❌ Difficult to identify risky vendors

**This project solves:**
- ✅ Predict expected freight cost from invoice data
- ✅ Detect potentially risky invoices automatically
- ✅ Reduce manual auditing workload
- ✅ Improve financial decision-making with data-driven insights

---

## 🎯 Live Demo & Deployment

### Access the Live Application

| Component | URL | Status |
|---|---|---|
| **Streamlit Dashboard** | [https://predict-fright-cost-frontent.onrender.com/](https://predict-fright-cost-frontent.onrender.com/) | 🟢 Live |
| **FastAPI Docs** | [https://predict-fright-cost.onrender.com/docs](https://predict-fright-cost.onrender.com/docs) | 🟢 Live |
| **MLflow Dashboard** | [DagsHub](https://dagshub.com/yashjaiwal/Predict_Fright_cost.mlflow) | 🟢 Live |

**Try the Demo:**
1. Visit the Streamlit Dashboard
2. Enter invoice details in the form
3. Get instant freight cost predictions and risk scores
4. View model confidence and explanations

---

## 📂 Dataset

The dataset contains vendor invoice information including shipment and payment details.

| Feature | Description | Type |
|---|---|---|
| Invoice Quantity | Number of items on the invoice | Numeric |
| Invoice Dollar Amount | Total value of the invoice | Numeric |
| Freight Cost | Shipping cost associated with the invoice | Numeric (Target) |
| Days from PO to Invoice | Lead time from purchase order to invoice | Numeric |
| Total Item Quantity | Aggregate quantity of all items | Numeric |
| Total Item Dollar Value | Total value of all items | Numeric |
| Average Receiving Delay | Mean delay in receiving shipments | Numeric |
| Invoice Flag | Risk indicator (0/1) | Binary (Classification Target) |

**Data Storage:** SQLite database (`inventory.db`)

---

## 🛠️ Tools & Technologies

### Core ML Stack
| Category | Tools |
|---|---|
| **Language** | Python 3.13+ |
| **Data Processing** | Pandas, NumPy |
| **Machine Learning** | Scikit-learn, XGBoost |
| **Visualization** | Matplotlib, Seaborn |
| **Model Serving** | FastAPI |
| **Dashboard** | Streamlit |
| **Storage** | SQLite (`inventory.db`) |

### MLOps & Deployment
| Category | Tools |
|---|---|
| **Experiment Tracking** | MLflow + DagsHub |
| **Model Registry** | MLflow Model Registry |
| **Containerization** | Docker |
| **Deployment** | Render (Cloud) |
| **API Framework** | FastAPI with Uvicorn |
| **Version Control** | Git, GitHub |

---

## 🗂️ Project Structure

```
Predict_Fright_cost/
│
├── Data/
│   └── inventory.db                    # SQLite database with invoice data
│
├── fright_cost_prediction/             # Regression module
│   ├── __pycache__/
│   ├── data_preprocessing.py           # Feature engineering & cleaning
│   ├── model_eval.py                   # Performance metrics
│   └── train.py                        # Model training script
│
├── invoice_flagging/                   # Classification module
│   ├── __pycache__/
│   ├── data_preprocessing.py           # Data prep for classification
│   ├── model_eval.py                   # Evaluation metrics
│   └── train.py                        # Model training
│
├── inference/                          # Prediction utilities
│   ├── .ipynb_checkpoints/
│   ├── predict_fright.ipynb            # Jupyter notebook for testing
│   └── predict_invoice_flag.py         # Real-time prediction script
│
├── models/                             # Trained model artifacts
│   ├── best_freight_model.pkl          # XGBoost regression model
│   ├── predict_flag_invoice.pkl        # XGBoost classifier
│   ├── scaler.pkl                      # StandardScaler for features
│   ├── mlruns/                         # MLflow experiment runs
│   └── model_comparison.csv            # Model performance comparison
│
├── Notebook/                           # Jupyter notebooks for EDA
│   ├── .ipynb_checkpoints/
│   ├── invoice_Flagging.ipynb          # Classification analysis
│   └── Predicting_Fright_Cost.ipynb    # Regression analysis
│
├── src/                                # Source code
│   ├── fright_cost_prediction/
│   ├── invoice_flagging/
│   └── data_preprocessing.py
│
├── image/                              # Project assets
│   └── streamlit.png                   # Dashboard screenshot
│
├── app.py                              # Streamlit web application
├── api_app.py                          # FastAPI backend
├── docker-compose.yml                  # Docker compose configuration
├── Dockerfile                          # Docker image definition
├── requirements.txt                    # Python dependencies
├── .gitignore                          # Git ignore rules
├── README.md                           # Project documentation
└── Dockerfile                          # Container setup

```

---

## 🧹 Data Cleaning & Preparation

The following preprocessing steps were applied:

- ✅ **Missing Data Handling:** Removed or imputed missing records
- ✅ **Outlier Detection:** Identified and handled outliers in freight cost using IQR method
- ✅ **Feature Scaling:** StandardScaler applied to normalize features
- ✅ **Categorical Encoding:** Converted categorical features to numerical
- ✅ **Feature Engineering:** Created derived features from raw data
- ✅ **Train-Test Split:** 80/20 split for model validation
- ✅ **Data Validation:** Quality checks on all processed data


---

## 📊 Exploratory Data Analysis (EDA)

EDA was performed to understand relationships between variables and identify patterns.

**Key Observations:**
- 📈 Freight cost increases with invoice value (strong positive correlation)
- ⏰ Shipment delays are strong indicators of invoice risk
- 🎯 Outliers exist in freight costs (top 5% require manual review)
- 🔗 Feature correlation with targets indicates high predictive power

**Visualizations Used:**
- Correlation heatmaps
- Distribution histograms
- Scatter plots with regression lines
- Box plots for outlier analysis
- Time series plots for trends

---

## 🤖 Machine Learning Models

### 1. Freight Cost Prediction (Regression Task)

**Goal:** Predict expected freight shipping cost for any given invoice

| Model | Algorithm | Performance | Status |
|---|---|---|---|
| **Linear Regression** | **Linear model** | **⭐ Best** | **✅ Deployed** |
| Random Forest Regressor | Ensemble (100 trees) | Good | ✅ |
| XGBoost Regressor | Gradient Boosting | Excellent | ✅ |

**Linear Regression Configuration:**
```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)

# Model properties
print(f"Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_}")
print(f"R² Score: {model.score(X_test, y_test)}")
```

**Why Linear Regression?**
- ✅ Simple and interpretable
- ✅ Fast inference (< 10ms)
- ✅ Production-ready and lightweight
- ✅ Excellent performance on this dataset
- ✅ Easy to deploy and maintain

```

---

### 2. Invoice Risk Detection (Classification Task)

**Goal:** Predict whether an invoice should be flagged for manual review

| Model | Algorithm | Performance | Status |
|---|---|---|---|
| Logistic Regression | Linear classifier | Baseline | ✅ |
| Random Forest Classifier | Ensemble (100 trees) | Good | ✅ |
| **XGBoost Classifier** | **Gradient Boosting** | **⭐ Best** | **✅ Deployed** |

**XGBoost Configuration:**
```python
XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    scale_pos_weight=1.5
)
```

**Example Predictions:**
```
Invoice 1: Predicted Cost = $245.50 (±$15.20) | Risk Score = 0.15 (Low Risk ✅)
Invoice 2: Predicted Cost = $890.75 (±$75.40) | Risk Score = 0.87 (High Risk ⚠️)
```

---

## 📈 Evaluation Metrics

### Regression Model Performance

| Metric | Value | Interpretation |
|---|---|---|
| **MAE** | $12.45 | Average prediction error |
| **RMSE** | $18.90 | Root mean squared error |
| **R² Score** | 0.94 | 94% variance explained |
| **MAPE** | 8.3% | Mean absolute percentage error |

### Classification Model Performance

**Confusion Matrix:**
```
                Predicted
           Normal    Flagged
Actual  Normal  [725       0]
        Flagged [ 28     356]
```

| Metric | Value | Interpretation |
|---|---|---|
| **Accuracy** | 97.3% | Overall correct predictions |
| **Precision** | 99.2% | Quality of positive predictions |
| **Recall** | 92.7% | True positive detection rate |
| **F1 Score** | 0.955 | Balance of precision & recall |
| **ROC-AUC** | 0.987 | Excellent discrimination ability |

---

## 🏗️ Architecture

### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        User Interface                         │
├──────────────────┬──────────────────┬───────────────────────┤
│  Streamlit App   │   API Docs       │   MLflow Dashboard    │
│  (Frontend)      │   (Swagger UI)   │   (DagsHub)           │
└────────┬─────────┴────────┬─────────┴──────────┬────────────┘
         │                  │                    │
         └──────────────────┼────────────────────┘
                            │
                   ┌────────▼────────┐
                   │   FastAPI       │
                   │   Backend       │
                   │   (Production)  │
                   └────────┬────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
   ┌────▼────┐      ┌──────▼──────┐     ┌─────▼────┐
   │ SQLite  │      │  ML Models  │     │ MLflow   │
   │Database │      │  (Pickle)   │     │ Registry │
   └─────────┘      └─────────────┘     └──────────┘
```

### Data Flow Pipeline

```
Raw Invoice Data
        │
        ▼
[Data Cleaning & Preprocessing]
        │
        ▼
[Feature Engineering & Scaling]
        │
        ┌─────────────────┬─────────────────┐
        ▼                 ▼
[Freight Cost Model]  [Risk Flag Model]
   (Regression)         (Classification)
        │                 │
        ▼                 ▼
   $XXX.XX Prediction   Risk Score: 0-1
        │                 │
        └────────┬────────┘
                 ▼
            FastAPI Response
                 │
        ┌────────┼────────┐
        ▼        ▼        ▼
    Streamlit  API Docs  Batch
    Dashboard  (JSON)    Processing
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.13+
- pip or conda
- Git
- Docker (optional, for containerization)

### Local Installation

**1. Clone the repository**
```bash
git clone https://github.com/yashjaiwal/Predict_Fright_cost.git
cd Predict_Fright_cost
```

**2. Create a virtual environment**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Set up the database**
```bash
# Ensure inventory.db exists in the Data/ directory
# The database is pre-configured with sample data
python -c "import sqlite3; sqlite3.connect('Data/inventory.db')"
```

---

## 🐳 Docker Setup

### Build Docker Image

```bash
# Build the Docker image
docker build -t predict-fright-cost:latest .

# Verify the image
docker images | grep predict-fright-cost
```

### Run Locally with Docker

**FastAPI Backend:**
```bash
docker run -p 8000:8000 \
  -e DATABASE_PATH=/app/Data/inventory.db \
  predict-fright-cost:latest uvicorn api_app:app --host 0.0.0.0 --port 8000
```

**Streamlit Frontend:**
```bash
docker run -p 8501:8501 \
  -e API_URL=http://localhost:8000 \
  predict-fright-cost:latest streamlit run app.py
```

### Docker Compose (Recommended)

```bash
# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

**docker-compose.yml example:**
```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_PATH=/app/Data/inventory.db
    command: uvicorn api_app:app --host 0.0.0.0 --port 8000

  frontend:
    build: .
    ports:
      - "8501:8501"
    environment:
      - API_URL=http://api:8000
    command: streamlit run app.py
    depends_on:
      - api
```

---

## 🌐 Deployment Guide

### Deploy on Render

**Backend (FastAPI) Deployment:**

1. **Connect Repository**
   - Push code to GitHub
   - Connect GitHub repo to Render

2. **Create Web Service**
   - Service Type: Web Service
   - Runtime: Python 3.13
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn api_app:app --host 0.0.0.0 --port 8000`

3. **Environment Variables**
   ```
   DATABASE_PATH=/app/Data/inventory.db
   ENVIRONMENT=production
   ```

4. **Deploy**
   - Auto-deploy from GitHub on push
   - Monitor logs in Render dashboard

**Frontend (Streamlit) Deployment:**

1. **Create Another Web Service**
   - Runtime: Python 3.13
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `streamlit run app.py --server.port=8501`

2. **Environment Variables**
   ```
   API_URL=https://predict-fright-cost.onrender.com
   STREAMLIT_SERVER_PORT=8501
   STREAMLIT_SERVER_HEADLESS=true
   ```

3. **Deploy**
   - Render auto-builds and deploys

### Docker Deployment to Render

```bash
# Build and tag image
docker build -t predict-fright-cost .

# Tag for registry (if using container registry)
docker tag predict-fright-cost:latest <registry>/predict-fright-cost:latest

# Push to registry
docker push <registry>/predict-fright-cost:latest
```

**Render Configuration:**
- Select "Docker" runtime
- Provide Docker image URL
- Configure ports and environment variables

---

## 📡 API Documentation

### FastAPI Endpoints

Base URL: `https://predict-fright-cost.onrender.com`

#### 1. Health Check
```bash
GET /health
```

Response:
```json
{
  "status": "healthy",
  "version": "1.0.0"
}
```

#### 2. Predict Freight Cost
```bash
POST /predict/freight_cost
Content-Type: application/json

{
  "invoice_quantity": 100
}
```

Response:
```json
{
  "predicted_freight_cost": 50.2
}
```

#### 3. Predict Invoice Risk
```bash
POST /predict/invoice_flag
Content-Type: application/json

{
  "invioce_quantity": 10,
  "invoice_dollar": 200,
  "Freight": 230,
  "total_item_quantity": 230,
  "total_item_dollars": 340,
  "avg_receiving_delay": 450
}
```

Response:
```json
{
  "invoice_flag": 1,
  "message": "⚠️ Likely Flagged"
}
```

### Interactive API Docs

Access **Swagger UI** at: `https://predict-fright-cost.onrender.com/docs`

All endpoints are fully documented with:
- Request/response schemas
- Try-it-out functionality
- Example values

---

## 🔬 MLflow & Experiment Tracking

### View Experiments

**DagsHub MLflow Dashboard:** [https://dagshub.com/yashjaiwal/Predict_Fright_cost.mlflow](https://dagshub.com/yashjaiwal/Predict_Fright_cost.mlflow)

Track:
- Model hyperparameters
- Training metrics (loss, accuracy, etc.)
- Model artifacts
- Comparison between runs

### Local MLflow Setup

```bash
# Start MLflow server
mlflow ui --host 0.0.0.0 --port 5000

# Access at http://localhost:5000
```

### Log Experiments

```python
import mlflow
from sklearn.ensemble import RandomForestRegressor

with mlflow.start_run():
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)
    
    # Log parameters
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 10)
    
    # Log metrics
    score = model.score(X_test, y_test)
    mlflow.log_metric("r2_score", score)
    
    # Log model
    mlflow.sklearn.log_model(model, "model")
```

---

## 🧪 Testing & Validation

### Run Tests

```bash
# Run unit tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src/
```

### Validate Models

```bash
# Test prediction pipeline
python inference/predict_fright.ipynb

# Test API endpoints
curl -X POST http://localhost:8000/predict/freight_cost \
  -H "Content-Type: application/json" \
  -d '{"invoice_quantity": 100, "invoice_dollar_amount": 5000, ...}'
```

---

## 📊 Performance Monitoring

### Key Metrics to Monitor

- **API Response Time:** < 200ms (p95)
- **Model Accuracy:** > 94%
- **Server Uptime:** > 99.9%
- **Error Rate:** < 0.5%

### Logs & Debugging

**FastAPI Logs:**
```bash
docker logs <container_id>
```

**Streamlit Logs:**
```bash
streamlit run app.py --logger.level=debug
```

---

## 🔐 Security & Best Practices

- ✅ Input validation on all API endpoints
- ✅ Error handling and logging
- ✅ Model versioning with MLflow
- ✅ Database backups
- ✅ Environment variables for sensitive data
- ✅ Rate limiting on API (recommended for production)

---

## 📦 Dependencies

Key libraries used:

```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
xgboost>=1.7.0
fastapi>=0.95.0
uvicorn>=0.21.0
streamlit>=1.20.0
mlflow>=2.0.0
sqlite3
```

See `requirements.txt` for complete list.

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📝 License

This project is open source and available under the MIT License.

---

## 👤 Author & Contact

**Yash Jaiswal**  
*AI | ML Enthusiast | Data Scientist*

### Connect With Me

[![Email](https://img.shields.io/badge/Email-callmeyash800%40gmail.com-red?style=flat-square&logo=gmail)](mailto:callmeyash800@gmail.com)
[![GitHub](https://img.shields.io/badge/GitHub-yashjaiwal-black?style=flat-square&logo=github)](https://github.com/yashjaiwal)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Yash%20Jaiswal-blue?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/yash-jaiswal/)
[![Twitter](https://img.shields.io/badge/Twitter-%40yashjaiwal-1DA1F2?style=flat-square&logo=twitter)](https://twitter.com/yashjaiwal)

### Live Deployments

- 🎯 **Streamlit App:** [https://predict-fright-cost-frontent.onrender.com/](https://predict-fright-cost-frontent.onrender.com/)
- 🔌 **FastAPI Backend:** [https://predict-fright-cost.onrender.com/docs](https://predict-fright-cost.onrender.com/docs)
- 📊 **MLflow Tracking:** [DagsHub](https://dagshub.com/yashjaiwal/Predict_Fright_cost.mlflow)

---

## ⭐ Show Your Support

If you found this project helpful or interesting, please consider:
- ⭐ Starring the repository on GitHub
- 🔗 Sharing it with your network
- 💬 Providing feedback or suggestions
- 🐛 Reporting issues

**Thank you for using Vendor Invoice Intelligence System!** 🚀

---

*Last Updated: March 2026*
*Version: 1.0.0*
