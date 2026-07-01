import modal
import os
import sys

app = modal.App("heretic-optimization-ara")

# Define the image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch>=2.5.0",
        "transformers>=4.48.0",
        "accelerate>=1.1.0",
        "datasets>=3.1.0",
        "optuna>=4.1.0",
        "peft>=0.14.0",
        "pydantic>=2.10.0",
        "rich>=13.9.0",
        "questionary>=2.1.0",
        "tomli-w>=1.1.0",
        "psutil>=6.1.0",
        "huggingface-hub>=0.26.0",
        "lm-eval>=0.4.7"
    )
)

# Define the remote volume for caching models and saving checkpoints
volume = modal.Volume.from_name("heretic-cache-volume", create_if_missing=True)

# Define the local source code directory to mount
heretic_source = modal.Mount.from_local_dir(
    "C:/Users/shaik/heretic", 
    remote_path="/heretic",
    condition=lambda p: ".git" not in p and "__pycache__" not in p
)

@app.function(
    image=image,
    gpu="H100", # We need VRAM, but ARA limits overhead
    timeout=86400, # 24 hours
    volumes={"/data": volume},
    mounts=[heretic_source],
    secrets=[modal.Secret.from_name("huggingface-secret")] # Ensure you have HF_TOKEN configured in Modal secrets
)
def run_heretic():
    # Set up environment variables
    os.environ["HF_HOME"] = "/data/huggingface"
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    
    # Change to the Heretic directory
    os.chdir("/heretic")
    
    # Add src to python path so we can import heretic
    sys.path.insert(0, "/heretic/src")
    
    print("Starting Heretic ARA Optimization run on Modal...")
    
    from heretic.main import run
    
    # Simulate command line arguments
    sys.argv = ["heretic"]
    
    # Create checkpoints directory on the volume if it doesn't exist
    os.makedirs("/data/checkpoints", exist_ok=True)
    
    # Execute the run
    run()

@app.local_entrypoint()
def main():
    print("Deploying Heretic ARA to Modal...")
    run_heretic.remote()
