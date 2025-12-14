from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import os
from google.cloud import storage

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
LOCAL_FOLDER = "/opt/airflow/data/Arranged Batch"   # <-- Correct folder with space
BUCKET_NAME = "your storage bucket name"
SERVICE_ACCOUNT_FILE = "/keys/service account.json"
# -------------------------------------------------

def replace_bucket_with_local():

    # Authenticate using service account JSON
    client = storage.Client.from_service_account_json(SERVICE_ACCOUNT_FILE)
    bucket = client.bucket(BUCKET_NAME)

    # -------------------------------------------------
    # 1) DELETE ALL FILES IN BUCKET
    # -------------------------------------------------
    blobs = list(bucket.list_blobs())

    if blobs:
        print(f"Removing {len(blobs)} existing objects in the bucket...")
        for blob in blobs:
            blob.delete()
        print("Bucket cleared.")
    else:
        print("Bucket already empty.")

    # -------------------------------------------------
    # 2) UPLOAD ENTIRE FOLDER RECURSIVELY
    # -------------------------------------------------
    if not os.path.exists(LOCAL_FOLDER):
        raise ValueError(f"Local folder not found: {LOCAL_FOLDER}")

    base_folder_name = os.path.basename(LOCAL_FOLDER)
    uploaded = []

    for root, dirs, files in os.walk(LOCAL_FOLDER):
        for file in files:
            local_file_path = os.path.join(root, file)
            # preserve folder structure in GCS
            relative_path = os.path.relpath(local_file_path, LOCAL_FOLDER)
            gcs_path = f"{base_folder_name}/{relative_path}"

            blob = bucket.blob(gcs_path)
            blob.upload_from_filename(local_file_path)

            uploaded.append(gcs_path)

    if uploaded:
        print(f"Uploaded files: {uploaded}")
    else:
        print("No files found in Arranged Batch folder.")


default_args = {
    "start_date": datetime(2025, 1, 1),
}

with DAG(
    dag_id="monthly_replace_folder_in_gcs",
    default_args=default_args,
    schedule_interval="0 0 1 * *",
    catchup=False,
) as dag:

    replace_task = PythonOperator(
        task_id="replace_bucket_data",
        python_callable=replace_bucket_with_local,
    )
