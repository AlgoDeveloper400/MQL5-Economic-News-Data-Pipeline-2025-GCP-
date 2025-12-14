# 📦 MQL5 Economic News Data Pipeline 2025 (GCP)

A **production-ready, hybrid data & ML pipeline** designed to ingest **monthly economic release data from MQL5**, store it on Google Cloud, train and validate models locally, track experiments, and surface results on dashboards — all while **keeping cloud costs minimal**.

This project demonstrates a real-world ML system covering:
- Automated ingestion
- Cloud-backed storage
- Secure data access
- Model training, validation, and testing
- Experiment tracking
- SQL-backed metrics
- Dashboard visualization

---

## ⚠️ Disclaimer (Repository Scope)

> **This repository does not contain the complete production codebase.**  
>  
> Only **selected files, configuration samples, and architectural references** are provided for demonstration and documentation purposes.
>
> Sensitive components such as:
> - Proprietary data processing logic  
> - Full ML training implementations  
> - Production credentials and secrets  
> - Private automation scripts  
>
> have been intentionally **excluded**.
>
> This repository is intended to showcase **system design, architecture, and workflow structure**, not to function as a fully runnable production system out of the box.

---

## 🎯 What This Pipeline Automates

- Monthly batch ingestion
- Google Cloud Storage persistence
- Cloud SQL data management
- Feature preparation
- Model training lifecycle
- Validation & testing
- Metrics storage
- MLflow experiment tracking
- Dashboard-ready analytics

---

## 🗺️ System Architecture

![Pipeline Diagram](./GCP%20Pipeline%20Decmeber%20Final%20Version%202025.png)

> *(See `GCP Pipeline Decmeber Final Version 2025.png` in the repository for full resolution)*

---

## 🧩 Architecture Overview

This pipeline is intentionally designed as a **hybrid system**:

- **GCP** is used for durable storage and shared data access
- **Local execution** handles orchestration, ML training, and experimentation
- Avoids expensive managed services while remaining production-structured

❌ Cloud Composer  
❌ Kubernetes / GKE  

✔ Local Docker Airflow  
✔ Cloud SQL  
✔ Cloud Storage  

---

## 🚀 Core Pipeline Stages

---

## 1️⃣ Data Ingestion & Storage

**Trigger:** Monthly (Airflow schedule)

### Flow
1. Monthly trigger initiates the pipeline
2. Economic data is arranged into batch folders
3. Batch folders are uploaded to **Google Cloud Storage**
4. Local Airflow DAG detects new uploads
5. Data is processed and written into **Cloud SQL**
6. Data becomes available for downstream ML workflows

### Tech Used
- Dockerized Airflow (local)
- Google Cloud Storage
- Cloud SQL (MySQL / PostgreSQL)

---

## 2️⃣ Model Training, Validation & Testing

**Execution:** Local (Airflow + FastAPI)

### Flow
1. Secure connection to Cloud SQL (Auth Proxy / SSH tunnel)
2. Training data pulled from the database
3. ML automation executed via **FastAPI scripts**
4. Model training step
5. Model validation step
6. Model testing step
7. Evaluation metrics stored back into **Cloud SQL**

### Tech Used
- FastAPI (ML automation layer)
- Dockerized Airflow
- Cloud SQL
- Secure DB tunneling

---

## 3️⃣ Experiment Tracking & Results Display

### Flow
1. Each training run is logged to **MLflow**
2. Metrics and metadata stored in SQL
3. Dashboard pulls metrics from Cloud SQL
4. Results visualized over time for comparison and monitoring

### Tech Used
- MLflow (local)
- Cloud SQL
- Dashboard tools:
  - Power BI
  - Looker Studio
  - Grafana
  - Custom dashboards

---

## 🛠️ Infrastructure Overview

| Component | Environment |
|---------|-------------|
| Google Cloud Storage | GCP |
| Cloud SQL | GCP |
| Airflow (all DAGs) | Local Docker |
| Model Training | Local |
| Experiment Tracking | Local |
| Dashboards | Cloud / Local Hybrid |

---

## 🌩️ Deployment Strategy (Hybrid)

This pipeline intentionally splits responsibilities to **optimize cost and control**.

### Cloud Components
- Cloud Storage bucket
- Cloud SQL database

### Local Components
- Docker-based Airflow
- ML training & validation
- Experiment tracking
- ETL orchestration

### Cost Strategy
- **Cloud Composer is NOT used** (high cost)
- **Kubernetes is NOT deployed**
- Local compute handles all heavy ML workloads

---

## 📅 Run Frequency

**Monthly — fully automated via Airflow**

---

## 🔥 Key Features

- Cloud-backed persistent storage
- Fully automated DAG-based workflows
- Secure DB access patterns
- Local-first ML experimentation
- SQL-based metrics history
- Built-in experiment tracking loop
- Dashboard-ready reporting layer

---

## 📈 Future Enhancements

- MLflow Model Registry integration
- CI/CD pipeline validation
- Optional cloud-based training mode
- Live model performance monitoring
- Automated alerting on metric degradation
