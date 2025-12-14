"""
Automation script for ML Pipeline
Triggers endpoints in order: Train → Validate → Test
Runs without using the Swagger UI
"""

import requests
import time
import json
import sys
from typing import Dict, Any

# Configuration
BASE_URL = "http://127.0.0.1:9009"
TIMEOUT = 3600  # 1 hour timeout per request

def check_server_ready():
    """Check if FastAPI server is ready"""
    max_retries = 30
    for i in range(max_retries):
        try:
            response = requests.get(f"{BASE_URL}/health", timeout=10)
            if response.status_code == 200:
                print("✓ Server is ready")
                return True
        except requests.exceptions.ConnectionError:
            print(f"  Waiting for server... ({i+1}/{max_retries})")
            time.sleep(2)
    
    print("✗ Server not ready after waiting")
    return False

def load_params_from_file(file_path: str) -> Dict[str, Any]:
    """Load parameters from JSON file"""
    try:
        with open(file_path, 'r') as f:
            params = json.load(f)
        print(f"✓ Loaded parameters from {file_path}")
        return params
    except Exception as e:
        print(f"✗ Error loading parameters file: {e}")
        return {}

def call_endpoint(endpoint: str, method: str = "POST", params: Dict = None, json_data: Dict = None):
    """Call API endpoint with error handling"""
    url = f"{BASE_URL}{endpoint}"
    
    print(f"\n{'='*80}")
    print(f"CALLING: {endpoint}")
    print(f"{'='*80}")
    
    try:
        if method == "POST":
            response = requests.post(url, params=params, json=json_data, timeout=TIMEOUT)
        else:
            response = requests.get(url, params=params, timeout=TIMEOUT)
        
        response.raise_for_status()
        result = response.json()
        
        print(f"✓ {endpoint} completed successfully")
        if "message" in result:
            print(f"  Message: {result['message']}")
        
        return result
    
    except requests.exceptions.Timeout:
        print(f"✗ {endpoint} timed out after {TIMEOUT} seconds")
        return {"error": f"Timeout after {TIMEOUT} seconds"}
    
    except requests.exceptions.RequestException as e:
        print(f"✗ {endpoint} failed: {e}")
        if hasattr(e.response, 'text'):
            print(f"  Response: {e.response.text}")
        return {"error": str(e)}

def run_automation(
    use_file_params: bool = True,
    params_file_path: str = r"path to model parameters file\Model Parameters.json",
    skip_training: bool = False
):
    """
    Run full automation pipeline
    """
    print("\n" + "="*80)
    print("ML PIPELINE AUTOMATION RUNNER")
    print("="*80)
    
    # Step 1: Check server
    print("\n[1/4] Checking server availability...")
    if not check_server_ready():
        print("Please start the FastAPI server first:")
        print("  python main.py")
        print("Or visit: http://127.0.0.1:9009/docs")
        return False
    
    # Step 2: Check health
    print("\n[2/4] Checking system health...")
    health = call_endpoint("/health", method="GET")
    if "status" in health and health["status"] != "ok":
        print("✗ System health check failed")
        print(f"  Details: {health}")
        return False
    
    print("✓ System health: OK")
    print(f"  Train available: {health.get('train_available', False)}")
    print(f"  Validate available: {health.get('validate_available', False)}")
    print(f"  Test available: {health.get('test_available', False)}")
    
    # Step 3: Run automation endpoint (simplified)
    print("\n[3/4] Starting pipeline automation...")
    
    # Option A: Use the new /automate endpoint (recommended)
    print("Using unified /automate endpoint...")
    result = call_endpoint(
        "/automate",
        params={
            "use_file_params": use_file_params,
            "skip_training": skip_training,
            "timeout": TIMEOUT
        }
    )
    
    # Option B: Manual sequential calls (fallback)
    if "error" in result:
        print("\n⚠ Unified automation failed, trying sequential calls...")
        
        # Load parameters if needed
        training_params = {}
        if use_file_params:
            training_params = load_params_from_file(params_file_path)
        
        # Train
        train_result = call_endpoint(
            "/train",
            params={
                "use_file_params": use_file_params,
                "merge_params": True
            },
            json_data=training_params if not use_file_params else None
        )
        
        # Validate
        validate_result = call_endpoint("/validate")
        
        # Test
        test_result = call_endpoint("/test")
        
        result = {
            "training": train_result,
            "validation": validate_result,
            "testing": test_result
        }
    
    # Step 4: Display results
    print("\n[4/4] Automation Results:")
    print("="*80)
    
    if "results" in result:
        for step, step_result in result["results"].items():
            status = step_result.get("status", "unknown")
            if status == "success":
                print(f"✓ {step.capitalize()}: SUCCESS")
            elif status == "skipped":
                print(f"⚠ {step.capitalize()}: SKIPPED ({step_result.get('reason', 'No reason')})")
            else:
                print(f"✗ {step.capitalize()}: FAILED")
    
    print("\n" + "="*80)
    print("AUTOMATION COMPLETE")
    print("="*80)
    
    return True

def create_parameters_file(file_path: str):
    """Create a sample parameters file if it doesn't exist"""
    sample_params = {
        "SEQ_LENGTH": 5,
        "HIDDEN_SIZE": 64,
        "NUM_LAYERS": 2,
        "DROPOUT": 0.2,
        "BATCH_SIZE": 64,
        "EPOCHS": 10,
        "LR": 0.001,
        "WEIGHT_DECAY": 1e-4,
        "XGB_N_ESTIMATORS": 100,
        "XGB_MAX_DEPTH": 3,
        "XGB_LEARNING_RATE": 0.1,
        "description": "Model parameters for economic news ML pipeline",
        "version": "1.0",
        "last_updated": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(sample_params, f, indent=2)
        print(f"✓ Created sample parameters file: {file_path}")
        return True
    except Exception as e:
        print(f"✗ Error creating parameters file: {e}")
        return False

if __name__ == "__main__":
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="ML Pipeline Automation Runner")
    parser.add_argument("--no-file", action="store_true", help="Don't use parameters from file")
    parser.add_argument("--skip-train", action="store_true", help="Skip training step")
    parser.add_argument("--create-params", action="store_true", help="Create sample parameters file")
    parser.add_argument("--params-file", default=r"path to model parameters file\Model Parameters.json",
                       help="Path to parameters JSON file")
    
    args = parser.parse_args()
    
    # Create parameters file if requested
    if args.create_params:
        create_parameters_file(args.params_file)
        sys.exit(0)
    
    # Check if parameters file exists
    if not os.path.exists(args.params_file) and not args.no_file:
        print(f"⚠ Parameters file not found: {args.params_file}")
        create = input("Create sample parameters file? (y/n): ")
        if create.lower() == 'y':
            create_parameters_file(args.params_file)
    
    # Run automation
    success = run_automation(
        use_file_params=not args.no_file,
        params_file_path=args.params_file,
        skip_training=args.skip_train
    )
    
    sys.exit(0 if success else 1)