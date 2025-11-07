# 🧠 Credit Risk Prediction API (FastAPI + Docker)

This project predicts loan default risk using a trained XGBoost model.  
It’s built with FastAPI, Dockerized for deployment, and visualized with Power BI.

---

## 🚀 Features
- Machine Learning model (XGBoost) trained on customer credit data  
- REST API for real-time risk predictions  
- Dockerized for easy deployment anywhere  
- Power BI dashboard for visual insights  

---

## 🛠️ Setup (Without Docker)
```bash
pip install -r requirement.txt
uvicorn scripts.api:app --reload
