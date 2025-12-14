import os
import time
import math
import re
from io import BytesIO
from datetime import datetime
import pandas as pd
import mysql.connector
from dotenv import load_dotenv
from google.cloud import storage

# ---------------- CONFIG ----------------
env_path = r"/path/to/project/.env"
print(f"📁 Loading environment from: {env_path}")
print(f"📄 File exists: {os.path.exists(env_path)}")
load_dotenv(env_path)

# ---------------- CLOUD SQL AUTH PROXY CONFIG ----------------
# Make sure the proxy is running on 127.0.0.1:3307, or set your own port number if you like, it really doesn't matter.
MYSQL_HOST = "127.0.0.1"
MYSQL_PORT = 3307
MYSQL_USER = os.getenv("MYSQL_USER", "root").strip()
MYSQL_ROOT_PASSWORD = os.getenv("MYSQL_ROOT_PASSWORD", "").strip()
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "").strip()

# GCS config
GCS_SERVICE_ACCOUNT_FILE = os.getenv("GCS_SERVICE_ACCOUNT_FILE")
if GCS_SERVICE_ACCOUNT_FILE:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GCS_SERVICE_ACCOUNT_FILE
    print(f"🔑 GCS Service Account: {GCS_SERVICE_ACCOUNT_FILE}")
else:
    print("⚠️ GCS_SERVICE_ACCOUNT_FILE not set")

GCS_BUCKET = os.getenv("GCS_BUCKET")
GCS_PREFIX = os.getenv("GCS_PREFIX", "Arranged Batch/")

# ---------------- HELPERS ----------------
def wait_for_mysql():
    missing = []
    if not MYSQL_HOST:
        missing.append("MYSQL_HOST")
    if not MYSQL_USER:
        missing.append("MYSQL_USER")
    if not MYSQL_ROOT_PASSWORD:
        missing.append("MYSQL_ROOT_PASSWORD")
    if missing:
        raise ValueError(f"Missing environment variables: {', '.join(missing)}")

    print("⏳ Waiting for MySQL server to be ready...")
    start = time.time()
    attempt = 1
    while True:
        try:
            conn = mysql.connector.connect(
                host=MYSQL_HOST,
                port=MYSQL_PORT,
                user=MYSQL_USER,
                password=MYSQL_ROOT_PASSWORD,
                connection_timeout=5
            )
            conn.close()
            print("✅ MySQL server is ready!")
            return
        except mysql.connector.Error as e:
            elapsed = time.time() - start
            if elapsed > 120:
                raise TimeoutError(f"MySQL server did not start in 120 seconds. Last error: {e}")
            print(f"⏳ Waiting 5s (Attempt {attempt}): {e}")
            time.sleep(5)
            attempt += 1

def connect_to_database():
    print(f"🔗 Connecting to database '{MYSQL_DATABASE}'...")
    start = time.time()
    attempt = 1
    while True:
        try:
            conn = mysql.connector.connect(
                host=MYSQL_HOST,
                port=MYSQL_PORT,
                user=MYSQL_USER,
                password=MYSQL_ROOT_PASSWORD,
                database=MYSQL_DATABASE,
                connection_timeout=10
            )
            print(f"✅ Connected to '{MYSQL_DATABASE}'")
            return conn
        except mysql.connector.Error as e:
            elapsed = time.time() - start
            if elapsed > 60:
                raise ConnectionError(f"Cannot connect to '{MYSQL_DATABASE}'. Error: {e}")
            print(f"⏳ Connection failed, retrying in 5s (Attempt {attempt}): {e}")
            time.sleep(5)
            attempt += 1

def verify_table_exists(conn):
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT COUNT(*) 
            FROM information_schema.tables 
            WHERE table_schema = %s 
            AND table_name = 'events'
        """, (MYSQL_DATABASE,))
        exists = cursor.fetchone()[0] > 0
        if exists:
            cursor.execute("DESCRIBE events")
            columns = [col[0] for col in cursor.fetchall()]
            print(f"✅ Events table exists with columns: {columns}")
            return True
        print("❌ Events table does not exist")
        return False
    finally:
        cursor.close()

def find_csv_gcs(bucket_name, prefix):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=prefix))
    csv_files = [b for b in blobs if b.name.endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError(f"No CSVs found in {bucket_name}/{prefix}")
    print(f"📄 Selected CSV: {csv_files[0].name}")
    return csv_files[0]

def parse_date(date_str):
    if pd.isna(date_str) or str(date_str).strip() == "":
        return None
    for fmt in ("%Y-%m-%d", "%d %B %Y", "%m/%d/%Y", "%d/%m/%Y", "%Y/%m/%d",
                "%m-%d-%Y", "%d-%m-%Y", "%b %d, %Y", "%B %d, %Y"):
        try:
            return datetime.strptime(str(date_str).strip(), fmt).date()
        except:
            continue
    return None

def parse_time(time_str):
    if pd.isna(time_str) or str(time_str).strip() == "":
        return None
    time_str = str(time_str).strip()
    for fmt in ("%H:%M", "%I:%M %p"):
        try:
            return datetime.strptime(time_str, fmt).time()
        except:
            continue
    return None

def clean_text(value):
    if pd.isna(value) or str(value).strip().lower() in ["", "nan", "none", "null"]:
        return "N/A"
    return str(value).strip()

def import_data_to_database(conn, df):
    cursor = conn.cursor()
    try:
        df["DateTime"] = pd.to_datetime(df["Date"].astype(str) + " " + df["Time"].astype(str))
        df.sort_values("DateTime", inplace=True)
        df.reset_index(drop=True, inplace=True)

        # Check last timestamp
        try:
            cursor.execute("SELECT MAX(CONCAT(Date,' ',Time)) FROM events")
            last_ts = cursor.fetchone()[0]
            if last_ts:
                last_dt = pd.to_datetime(last_ts)
                df = df[df["DateTime"] > last_dt]
        except mysql.connector.Error:
            pass

        if df.empty:
            print("✅ No new rows to import")
            return

        df.drop(columns=["DateTime"], inplace=True)
        insert_query = """
            INSERT INTO events (Date, Time, Currency, Event, Impact, Actual, Forecast, Previous)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
            ON DUPLICATE KEY UPDATE
                Impact=VALUES(Impact),
                Actual=VALUES(Actual),
                Forecast=VALUES(Forecast),
                Previous=VALUES(Previous)
        """
        data_to_insert = [tuple(row) for row in df[["Date","Time","Currency","Event","Impact","Actual","Forecast","Previous"]].values]
        for i in range(0, len(data_to_insert), 50):
            chunk = data_to_insert[i:i+50]
            cursor.executemany(insert_query, chunk)
            conn.commit()
        print(f"✅ Imported {len(data_to_insert)} rows")
    finally:
        cursor.close()

# ---------------- MAIN ----------------
def main():
    print("\n🚀 Starting CSV Import Process\n")
    wait_for_mysql()
    conn = connect_to_database()
    try:
        if not verify_table_exists(conn):
            print(f"❌ Table 'events' not found. Run init.sql first.")
            return

        blob = find_csv_gcs(GCS_BUCKET, GCS_PREFIX)
        data_bytes = blob.download_as_bytes()
        df = pd.read_csv(BytesIO(data_bytes), header=None,
                         names=["Date","Time","Currency","Event","Impact","Actual","Forecast","Previous","IsHoliday","WeekRange"],
                         quotechar='"', skipinitialspace=True, na_filter=False)

        # Clean CSV
        columns_to_drop = [c for c in ["IsHoliday","WeekRange"] if c in df.columns]
        if columns_to_drop:
            df.drop(columns=columns_to_drop, inplace=True)

        df["Date"] = df["Date"].apply(parse_date)
        df["Time"] = df["Time"].apply(parse_time)
        for col in ["Currency","Event","Impact","Actual","Forecast","Previous"]:
            if col in df.columns:
                df[col] = df[col].apply(clean_text)

        df.dropna(subset=["Date","Time"], inplace=True)
        if df.empty:
            print("❌ No valid data to import after cleaning")
            return

        import_data_to_database(conn, df)
    finally:
        conn.close()
        print("🔒 Database connection closed")

if __name__ == "__main__":
    main()
