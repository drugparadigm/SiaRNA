import torch
import os

# --- Configuration ---
# Directory where the model will be saved locally
LOCAL_MODEL_DIR = "src/data/input/t5_local"
# LOCAL_MODEL_DIR = "/raid/apitempfiles/siarna/t5_local"

# The model identifier on Hugging Face Model Hub
MODEL_NAME = "Rostlab/prot_t5_xl_half_uniref50-enc"
# Determine the device for the model
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def check_and_download_model(model_dir: str = LOCAL_MODEL_DIR):
    """
    Checks if the specified model directory exists and contains the necessary files.
    If it doesn't exist or is incomplete, it downloads the model and tokenizer
    from the Hugging Face Model Hub and saves them to the local directory.
    """
    # Check for the existence of the directory and some key files
    # A simple check for the directory is often sufficient for initial setup
    if os.path.isdir(model_dir) and \
       os.path.exists(os.path.join(model_dir, 'config.json')) and \
       os.path.exists(os.path.join(model_dir, 'tokenizer_config.json')):
        print(f"Model and tokenizer already exist in '{model_dir}'. Skipping download.")
        return False, f"Model and tokenizer found in {model_dir}."
    else:
        print(f"Model directory '{model_dir}' not found or incomplete. Starting download...")
        
        try:
            # Import transformers only when needed, especially for the download step
            # to keep the initial import fast if the model is already present.
            from transformers import T5Tokenizer, T5EncoderModel

            # 1. Download and load the tokenizer
            print(f"Downloading tokenizer from {MODEL_NAME}...")
            tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, do_lower_case=False)
            
            # 2. Download and load the model
            print(f"Downloading model from {MODEL_NAME}...")
            model = T5EncoderModel.from_pretrained(MODEL_NAME)
            
            # 3. Move model to the selected device
            model.to(DEVICE)
            print(f"Model loaded and moved to device: {DEVICE}")

            # 4. Save model and tokenizer locally
            print(f"Saving model and tokenizer to '{model_dir}'...")
            model.save_pretrained(model_dir)
            tokenizer.save_pretrained(model_dir)
            
            print(f"Download and saving complete. Model saved to '{model_dir}'.")
            return True, f"Model and tokenizer successfully downloaded and saved to {model_dir}."

        except Exception as e:
            # Handle missing dependencies
            if isinstance(e, ImportError):
                raise ValueError("Error: 'transformers' or 'torch' library not installed. Please install them.") from e

            # Handle any other failure
            print(f"An error occurred during download: {e}")

            # Attempt to clean up partial downloads
            if os.path.exists(model_dir):
                print(f"Attempting to clean up partial download in {model_dir}...")

            # Raise a descriptive error for the calling code
            raise ValueError(f"Download failed: {str(e)}") from e
