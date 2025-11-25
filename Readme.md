# 📦 MQL5 Economic News Data Pipeline 2025

A production-ready, automated data ingestion, processing, model training, experiment tracking, and dashboard pipeline designed to work with **monthly economic release data** sourced from MQL5.

This pipeline automates:
- Batch ingestion
- Storage
- Feature generation
- Model training
- Validation + testing
- Metrics storage
- Experiment logging
- Dashboard display

> **Note:** Only select components of this pipeline will run in the cloud.  
> Google Cloud Storage and Cloud SQL will run on GCP, while all Airflow DAGs will execute locally in a Docker-based Airflow environment.  
> Cloud Composer and Kubernetes will not be used due to high cost consumption, especially under a free-credit account.

---

## 🗺️ System Architecture

![Pipeline Diagram](./GCP-Pipeline-December-2025.png)

> *(See `GCP Pipeline December 2025.png` in the repo for full resolution)*

---

## 🚀 Core Pipeline Stages

---

### 1) Data Ingestion + Storage

**Trigger:** Monthly

**Flow**
1. Monthly ingestion trigger fires
2. Arranged batch folder uploaded
3. Files pushed to *Google Cloud Storage Bucket*
4. Local Airflow DAG detects & processes new uploads
5. Data is inserted into **Cloud SQL**
6. Data becomes available for downstream stages

**Tech Used**
- Docker-Airflow
- Cloud Storage
- Cloud SQL

---

### 2) Model Training, Validation, and Testing

**Flow**
1. Secure DB tunnel
2. Pull data from Cloud SQL
3. Execute ML routines via FastAPI scripts
4. Train the model
5. Monthly DAG runs Training, Validation, Testing
6. Validate model
7. Test model
8. Store model metrics in SQL

**Tech Used**
- FastAPI
- Cloud SQL
- Docker-AIrfow scheduling

---

### 3) Experiment Tracking + Dashboard Display

**Flow**
1. MLflow logs experiments and metadata
2. Dashboard queries Cloud SQL
3. Dashboard visualizes metrics & history

**Tech Used**
- MLflow
- Cloud SQL
- Dashboard tool (PowerBI / Looker Studio / Grafana / etc.)

---

## 🛠️ Infrastructure

| Component | Environment |
|----------|-------------|
| Google Cloud Storage | Cloud |
| Cloud SQL | Cloud |
| Airflow (all DAGs) | Local Docker |
| Model scripts | Local |
| Experiment tracking | Local |
| Dashboard | Cloud/Local hybrid |

---

## 🌩 Deployment Strategy (Hybrid)

This pipeline intentionally splits compute between cloud and local systems to reduce cost and maximize control.

**Cloud components**
- Cloud Storage bucket
- Cloud SQL server

**Local components**
- Docker-based Airflow
- ML experimentation
- Model tuning cycles
- ETL orchestration

**Key cost strategy decisions**
- **Cloud Composer will NOT be used** due to cost
- **Kubernetes will NOT be deployed**, as it would consume free credits extremely fast
    - *If required later, a very small K8s cluster may be tested on a local machine instead*

---

## 📅 Run Frequency

**Monthly — fully automated via Airflow**

---

## 🔥 Key Features

- Cloud-backed storage & SQL layer
- DAG-driven automation
- Local orchestration to avoid costs
- Automated retraining cycle
- SQL metric persistence
- Experiment tracking loop built-in
- Dashboard loop for results visibility

---

## 📈 Future Enhancements

- MLflow Model Registry integration
- CI/CD pipeline validation
- Optional cloud-based training mode
- Expanded live dashboards
