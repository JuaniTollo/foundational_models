import os
from huggingface_hub import login, HfApi, create_repo
from dotenv import load_dotenv

# Load the Hugging Face token from the .env file
load_dotenv()
hf_token = os.getenv('HUGGINGFACE_TOKEN')

if hf_token is None:
    raise ValueError("Hugging Face token not found in .env file")

# Login to Hugging Face using the token from .env
login(token=hf_token)

# Accept the output_dir as a parameter
import sys
if len(sys.argv) < 2:
    raise ValueError("Output directory parameter is required")
output_dir = sys.argv[1]

# Add a timestamp to the output directory
timestamp = time.strftime("%Y%m%d-%H%M%S")
repo = f"{output_dir}_{timestamp}"

# Create a repository with the name including the timestamp
create_repo(f"juantollo/{repo}", private=False)

# Initialize HfApi to handle the upload
api = HfApi()

# Upload the model folder to Hugging Face Hub
api.upload_folder(
    folder_path=output_dir,
    path_in_repo="",
    repo_id=f"juantollo/{repo}",
    repo_type="model",
    ignore_patterns="**/logs/*.txt",
)

print(f"Model uploaded to repository juantollo/{output_dir}")
