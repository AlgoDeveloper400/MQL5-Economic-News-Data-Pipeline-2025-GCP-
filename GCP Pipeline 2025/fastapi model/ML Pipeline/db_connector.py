import os
import mysql.connector
import pandas as pd
from dotenv import load_dotenv
import mlflow

# -------------------------------------------------------------------
# Load environment variables
# -------------------------------------------------------------------
dotenv_path = r"/path/to/env/.env"
load_dotenv(dotenv_path)

# -------------------------------------------------------------------
# MLFlow tracking
# -------------------------------------------------------------------
mlflow_url = os.getenv("MLFLOW_URL", "http://localhost:5000")
mlflow.set_tracking_uri(mlflow_url)

# -------------------------------------------------------------------
# UNIVERSAL CONNECTION - TRIES EVERYTHING
# -------------------------------------------------------------------
def get_connection():
    """Tries multiple connection methods until one works"""
    
    # Get all possible configs from environment
    configs_to_try = []
    #methods 1 and 2 will only work if you have special configurations, for this project we will use auth, so the first 2 will not be used. Just added them to show you all possible solutions.
    # Method 1: Direct GCP IP 
    if os.getenv("MYSQL_HOST"):
        configs_to_try.append({
            "name": "Direct GCP IP",
            "host": os.getenv("MYSQL_HOST"),
            "port": int(os.getenv("MYSQL_PORT", 3306)),
            "user": os.getenv("MYSQL_USER", "root"),
            "password": os.getenv("MYSQL_PASSWORD"),
            "database": os.getenv("MYSQL_DATABASE"),
        })
    
    # Method 2: Direct GCP IP with root password
    if os.getenv("MYSQL_HOST") and os.getenv("MYSQL_ROOT_PASSWORD"):
        configs_to_try.append({
            "name": "Direct GCP IP (Root)",
            "host": os.getenv("MYSQL_HOST"),
            "port": int(os.getenv("MYSQL_PORT", 3306)),
            "user": os.getenv("MYSQL_USER", "root"),
            "password": os.getenv("MYSQL_ROOT_PASSWORD"),
            "database": os.getenv("MYSQL_DATABASE"),
        })
    
    # Method 3: Auth Proxy (Port 3307)
    configs_to_try.append({
        "name": "Auth Proxy (3307)",
        "host": "127.0.0.1",
        "port": 3307,
        "user": os.getenv("MYSQL_USER", "root"),
        "password": os.getenv("MYSQL_PASSWORD") or os.getenv("MYSQL_ROOT_PASSWORD"),
        "database": os.getenv("MYSQL_DATABASE"),
    })
    
    # Method 4: Auth Proxy (Port 3306)
    configs_to_try.append({
        "name": "Auth Proxy (3306)",
        "host": "127.0.0.1",
        "port": 3306,
        "user": os.getenv("MYSQL_USER", "root"),
        "password": os.getenv("MYSQL_PASSWORD") or os.getenv("MYSQL_ROOT_PASSWORD"),
        "database": os.getenv("MYSQL_DATABASE"),
    })
    
    print("🔄 Testing MySQL connections...")
    
    # Try each config
    for config in configs_to_try:
        print(f"  → Trying {config['name']} at {config['host']}:{config['port']}")
        try:
            # Remove name before connecting
            conn_config = config.copy()
            conn_config.pop('name')
            
            # Add connection timeout
            conn_config["connection_timeout"] = 5
            
            conn = mysql.connector.connect(**conn_config)
            print(f"✅ SUCCESS with {config['name']}!")
            return conn
        except mysql.connector.Error as e:
            if e.errno == 1045:  # Access denied
                print(f"    ✗ Wrong credentials for {config['user']}")
            elif e.errno == 2003:  # Can't connect
                print(f"    ✗ Cannot reach {config['host']}:{config['port']}")
            else:
                print(f"    ✗ Error: {e}")
            continue
        except Exception as e:
            print(f"    ✗ Unexpected error: {e}")
            continue
    
    # If all failed
    error_msg = "\n❌ All connection methods failed!\n\nSOLUTIONS:\n"
    error_msg += "1. For Auth Proxy: Run 'cloud_sql_proxy -instances=YOUR_INSTANCE=tcp:3307'\n"
    error_msg += "2. For Direct GCP: Make sure your IP is whitelisted in GCP Console\n"
    error_msg += "3. Check .env file has correct credentials\n"
    error_msg += f"4. Check database exists: {os.getenv('MYSQL_DATABASE')}"
    raise ConnectionError(error_msg)


# -------------------------------------------------------------------
# Database functions
# -------------------------------------------------------------------
def query_to_df(query: str) -> pd.DataFrame:
    """Execute query and return pandas DataFrame"""
    conn = get_connection()
    try:
        df = pd.read_sql(query, conn)
        return df
    finally:
        conn.close()


def df_to_table(df: pd.DataFrame, table_name: str, if_exists='append', batch_size=1000):
    """Insert DataFrame into table"""
    conn = get_connection()
    cursor = conn.cursor()

    try:
        if if_exists == 'replace':
            cursor.execute(f"TRUNCATE TABLE {table_name}")
            conn.commit()

        columns = ", ".join(df.columns)
        placeholders = ", ".join(["%s"] * len(df.columns))
        sql = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"

        rows = [tuple(row) for row in df.to_numpy()]
        total = len(rows)

        for i in range(0, total, batch_size):
            batch = rows[i:i + batch_size]
            cursor.executemany(sql, batch)
            conn.commit()

        print(f"✅ Inserted {len(rows)} rows into {table_name}")

    except Exception as e:
        conn.rollback()
        print(f"❌ Insert error in {table_name}: {e}")
        raise
    finally:
        cursor.close()
        conn.close()


def get_table_row_count(table_name: str) -> int:
    """Get row count of a table"""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        return cursor.fetchone()[0]
    finally:
        cursor.close()
        conn.close()


# -------------------------------------------------------------------
# Table-specific helpers
# -------------------------------------------------------------------
def load_events_data():
    query = """
        SELECT Date, Time, Currency, Event, Impact, Actual, Forecast, Previous
        FROM events
        ORDER BY Date, Time
    """
    return query_to_df(query)


def save_train_metrics(df):
    df_to_table(df, "train_metrics", "replace", batch_size=500)


def save_validation_metrics(df):
    df_to_table(df, "validate_metrics", "replace", batch_size=500)


def save_test_forecasts(df):
    df_to_table(df, "test_forecasts", "replace", batch_size=500)


def is_first_live_forecasts_run():
    try:
        return get_table_row_count("live_forecasts") == 0
    except:
        return True


def save_live_forecasts(df, is_first_run=False):
    if_exists = "append" if is_first_run else "replace"
    df_to_table(df, "live_forecasts", if_exists, batch_size=500)


# -------------------------------------------------------------------
# Test connection
# -------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Testing MySQL Connection")
    print("=" * 60)
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Get server info
        cursor.execute("SELECT VERSION()")
        version = cursor.fetchone()[0]
        print(f"\n✅ Connected! MySQL Version: {version}")
        
        cursor.execute("SELECT NOW()")
        server_time = cursor.fetchone()[0]
        print(f"📅 Server Time: {server_time}")
        
        # List tables
        cursor.execute("SHOW TABLES")
        tables = cursor.fetchall()
        print(f"\n📊 Tables in database: {len(tables)}")
        
        for i, table in enumerate(tables, 1):
            table_name = table[0]
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cursor.fetchone()[0]
            print(f"  {i}. {table_name} - {count:,} rows")
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")