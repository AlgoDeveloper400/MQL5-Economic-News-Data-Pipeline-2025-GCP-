import os
import importlib
import traceback
import json
import pandas as pd
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.openapi.docs import get_swagger_ui_html
from pydantic import BaseModel
from typing import Optional, Dict, Any
import mlflow
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

# Import database connector
from db_connector import query_to_df

# --- Load environment variables ---
dotenv_path = r"path to env file\.env"
load_dotenv(dotenv_path)

# --- Config ---
OUTPUT_FOLDER = r"output folder where you will save your model, weights and etc\Model Artifacts"

# JSON parameters file path
PARAMS_FILE_PATH = r"path to model parameters file\Model Parameters.json"

DEFAULT_PARAMS = {
    "SEQ_LENGTH": 5,
    "HIDDEN_SIZE": 64,
    "NUM_LAYERS": 2,
    "DROPOUT": 0.2,
    "BATCH_SIZE": 64,
    "EPOCHS": 1,
    "LR": 0.001,
    "WEIGHT_DECAY": 1e-4,
    "XGB_N_ESTIMATORS": 100,
    "XGB_MAX_DEPTH": 3,
    "XGB_LEARNING_RATE": 0.1
}

# --- Load parameters from JSON file ---
def load_params_from_file(file_path: str) -> Dict[str, Any]:
    """Load parameters from JSON file"""
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                params = json.load(f)
            print(f"✓ Loaded parameters from file: {file_path}")
            return params
        else:
            print(f"⚠ Parameters file not found: {file_path}")
            return {}
    except Exception as e:
        print(f"✗ Error loading parameters file: {e}")
        return {}

# --- MLflow ---
mlflow_url = os.getenv("MLFLOW_URL", "http://localhost:5000")
mlflow.set_tracking_uri(mlflow_url)
mlflow.set_experiment("News Data Ml Pipeline")

# --- Pydantic model ---
class TrainingParams(BaseModel):
    SEQ_LENGTH: Optional[int] = None
    HIDDEN_SIZE: Optional[int] = None
    NUM_LAYERS: Optional[int] = None
    DROPOUT: Optional[float] = None
    BATCH_SIZE: Optional[int] = None
    EPOCHS: Optional[int] = None
    LR: Optional[float] = None
    WEIGHT_DECAY: Optional[float] = None
    XGB_N_ESTIMATORS: Optional[int] = None
    XGB_MAX_DEPTH: Optional[int] = None
    XGB_LEARNING_RATE: Optional[float] = None


# --- FastAPI app ---
app = FastAPI(
    title="Economic News Data ML Pipeline",
    docs_url=None,
    redoc_url=None
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Utility ---
def load_callable(module_name: str, candidates=("run_training", "run_train", "run")):
    try:
        mod = importlib.import_module(module_name)
    except Exception as e:
        raise ImportError(f"Could not import module '{module_name}': {e}")
    for name in candidates:
        fn = getattr(mod, name, None)
        if callable(fn):
            return fn
    for alt in ("main", module_name):
        fn = getattr(mod, alt, None)
        if callable(fn):
            return fn
    raise ImportError(f"No callable found in module '{module_name}'.")


def safe_import(module, candidates):
    try:
        return load_callable(module, candidates), None
    except ImportError as e:
        return None, str(e)


train_fn, train_err = safe_import("train", ("run_training",))
validate_fn, validate_err = safe_import("validate", ("run_validation",))
test_fn, test_err = safe_import("test", ("run_test",))


def fetch_mlflow_params():
    try:
        client = mlflow.tracking.MlflowClient()
        exp = client.get_experiment_by_name("News Data Ml Pipeline")
        if not exp:
            return DEFAULT_PARAMS.copy()
        runs = client.search_runs([exp.experiment_id], order_by=["attributes.start_time desc"], max_results=1)
        if not runs:
            return DEFAULT_PARAMS.copy()
        run = runs[0]
        params = DEFAULT_PARAMS.copy()
        for k in DEFAULT_PARAMS.keys():
            if k in run.data.params:
                val = run.data.params[k]
                try:
                    if isinstance(DEFAULT_PARAMS[k], int):
                        params[k] = int(val)
                    elif isinstance(DEFAULT_PARAMS[k], float):
                        params[k] = float(val)
                    else:
                        params[k] = val
                except:
                    pass
        return params
    except Exception as e:
        print(f"Error fetching MLflow parameters: {e}")
        return DEFAULT_PARAMS.copy()


# --- Custom Swagger UI ---
@app.get("/docs", include_in_schema=False)
def custom_swagger_ui_html():
    html = get_swagger_ui_html(
        openapi_url="/openapi.json",
        title="Economic News Data ML Pipeline - Docs",
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist/swagger-ui.css",
    )

    custom_style = """
    <style>
    div.models, section.models, .models {display: none !important;}
    .opblock-tag-section {display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 20px;}
    .opblock-tag.no-desc span {font-weight: bold; padding: 6px 12px; border-radius: 6px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);}
    .opblock-tag[data-tag='train'] .nostyle span {background: #d8f3dc; color: #1b4332;}
    .opblock-tag[data-tag='validate'] .nostyle span {background: #e0e7ff; color: #1e3a8a;}
    .opblock-tag[data-tag='test'] .nostyle span {background: #fde2e4; color: #7f1d1d;}
    </style>
    """

    return HTMLResponse(html.body.decode("utf-8") + custom_style)


# --- Health check with DB test ---
@app.get("/health")
def health_check():
    try:
        # Test database connection
        conn = None
        db_status = "unknown"
        try:
            from db_connector import get_connection
            conn = get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
            db_status = "healthy"
            cursor.close()
            conn.close()
        except Exception as e:
            db_status = f"error: {str(e)}"
        
        return {
            "status": "ok",
            "database": db_status,
            "train_available": bool(train_fn),
            "validate_available": bool(validate_fn),
            "test_available": bool(test_fn),
            "parameters_file_exists": os.path.exists(PARAMS_FILE_PATH)
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


# --- Endpoints ---
@app.get("/")
def root():
    file_exists = os.path.exists(PARAMS_FILE_PATH)
    file_status = "✓ Found" if file_exists else "✗ Not found"
    
    return {
        "message": "Economic News Data Pipeline (use /docs for UI)",
        "train_available": bool(train_fn),
        "validate_available": bool(validate_fn),
        "test_available": bool(test_fn),
        "parameters_file": file_status,
        "train_error": train_err if not train_fn else None,
        "validate_error": validate_err if not validate_fn else None,
        "test_error": test_err if not test_fn else None
    }


@app.post("/train", tags=["train"])
def train_endpoint(
    params: Optional[TrainingParams] = None,
    use_file_params: bool = Query(False, description="Use parameters from JSON file"),
    merge_params: bool = Query(True, description="Merge file parameters with manual parameters")
):
    """
    Train endpoint with hybrid parameter input:
    - use_file_params=True: Load parameters from JSON file
    - merge_params=True: Merge file params with manual params (manual overrides file)
    - Send params in body for manual override
    """
    if not train_fn:
        return JSONResponse({"error": f"Training function not available. {train_err}"})
    
    try:
        # Start with empty parameters
        final_params = {}
        
        # Load from file if requested
        if use_file_params:
            file_params = load_params_from_file(PARAMS_FILE_PATH)
            if file_params:
                final_params.update(file_params)
                print(f"Using {len(file_params)} parameters from file")
        
        # Merge with manual parameters if provided
        if params:
            manual_params = params.dict(exclude_none=True)
            if merge_params and manual_params:
                final_params.update(manual_params)  # Manual params override file params
                print(f"Applied {len(manual_params)} manual parameter overrides")
            elif not merge_params and manual_params:
                final_params = manual_params  # Use only manual params
                print("Using only manual parameters (file ignored)")
        
        # If no parameters at all, use defaults
        if not final_params:
            final_params = DEFAULT_PARAMS.copy()
            print("Using default parameters")
        
        print(f"Final training parameters: {final_params}")
        
        # Run training with combined parameters
        result = train_fn(OUTPUT_FOLDER, params=final_params)
        return {"message": "Training is complete", "parameters_used": final_params, "result": result}
        
    except Exception as e:
        tb = traceback.format_exc()
        return JSONResponse({"error": f"Training failed: {str(e)}", "traceback": tb})


@app.post("/validate", tags=["validate"])
def validate_endpoint():
    if not validate_fn:
        return JSONResponse({"error": f"Validation function not available. {validate_err}"})
    try:
        validate_fn(OUTPUT_FOLDER)
        return {"message": "Validation is complete"}
    except Exception as e:
        tb = traceback.format_exc()
        return JSONResponse({"error": f"Validation failed: {str(e)}", "traceback": tb})


@app.post("/test", tags=["test"])
def test_endpoint():
    if not test_fn:
        return JSONResponse({"error": f"Test function not available. {test_err}"})
    try:
        test_fn(OUTPUT_FOLDER)
        return {"message": "Testing is complete"}
    except Exception as e:
        tb = traceback.format_exc()
        return JSONResponse({"error": f"Testing failed: {str(e)}", "traceback": tb})


# --- Automation endpoint ---
@app.post("/automate", tags=["automation"])
def automate_pipeline(
    use_file_params: bool = Query(True, description="Use parameters from JSON file"),
    skip_training: bool = Query(False, description="Skip training if models already exist"),
    timeout: int = Query(3600, description="Timeout in seconds for each endpoint")
):
    """
    Automated pipeline execution:
    1. Train (optional)
    2. Validate
    3. Test
    Runs endpoints sequentially with timeout handling
    """
    results = {}
    
    try:
        # Check if training is needed
        models_exist = (
            os.path.exists(os.path.join(OUTPUT_FOLDER, "rnn_model.pt")) and
            os.path.exists(os.path.join(OUTPUT_FOLDER, "xgb_model.joblib"))
        )
        
        # Step 1: Training
        if not skip_training or not models_exist:
            print("\n" + "="*80)
            print("STARTING TRAINING STEP")
            print("="*80)
            
            # Load parameters from file for automation
            training_params = {}
            if use_file_params:
                training_params = load_params_from_file(PARAMS_FILE_PATH)
                if not training_params:
                    training_params = DEFAULT_PARAMS.copy()
            
            if train_fn:
                train_result = train_fn(OUTPUT_FOLDER, params=training_params)
                results["training"] = {"status": "success", "result": train_result}
                print("✓ Training completed")
            else:
                results["training"] = {"status": "skipped", "reason": "Train function not available"}
                print("⚠ Training skipped: Function not available")
        else:
            results["training"] = {"status": "skipped", "reason": "Models already exist"}
            print("⚠ Training skipped: Models already exist")
        
        # Step 2: Validation
        print("\n" + "="*80)
        print("STARTING VALIDATION STEP")
        print("="*80)
        
        if validate_fn:
            validate_fn(OUTPUT_FOLDER)
            results["validation"] = {"status": "success"}
            print("✓ Validation completed")
        else:
            results["validation"] = {"status": "skipped", "reason": "Validate function not available"}
            print("⚠ Validation skipped: Function not available")
        
        # Step 3: Testing
        print("\n" + "="*80)
        print("STARTING TESTING STEP")
        print("="*80)
        
        if test_fn:
            test_result = test_fn(OUTPUT_FOLDER)
            results["testing"] = {"status": "success", "result": test_result}
            print("✓ Testing completed")
        else:
            results["testing"] = {"status": "skipped", "reason": "Test function not available"}
            print("⚠ Testing skipped: Function not available")
        
        print("\n" + "="*80)
        print("PIPELINE AUTOMATION COMPLETE")
        print("="*80)
        
        return {
            "message": "Pipeline automation completed",
            "results": results,
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
    except Exception as e:
        tb = traceback.format_exc()
        return JSONResponse({
            "error": f"Pipeline automation failed: {str(e)}",
            "traceback": tb,
            "partial_results": results
        })


# --- Run ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=9009,
        reload=True,
        timeout_keep_alive=300,  # Keep connections alive longer
        timeout_graceful_shutdown=10  # Graceful shutdown timeout
    )