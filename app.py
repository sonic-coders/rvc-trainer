# GUI BY BF667
# Improved version with better code organization, error handling, and logging

import gradio as gr
import os
import re
import shutil
import traceback
import sys
import glob
import logging
import zipfile
import tempfile
from datetime import datetime
from subprocess import PIPE, STDOUT, Popen, run, CalledProcessError
from typing import Optional, Tuple, List, Dict, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import argparse

# HuggingFace Hub
try:
    from huggingface_hub import HfApi, create_repo, RepoCard, upload_folder, login, whoami
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    logger_warning = "huggingface_hub not installed. Run: pip install huggingface_hub"

# ========================================================================== #
# CONFIGURATION & CONSTANTS
# ========================================================================== #

parser = argparse.ArgumentParser()
parser.add_argument("--colab", action="store_true", help="Launch in colab")
args = parser.parse_args()

BASE_ROOT = os.getcwd()
sys.path.append(BASE_ROOT)

@dataclass
class Config:
    """Application configuration settings."""
    n_cpu: int = os.cpu_count() or 4
    iscolab: bool = args.colab
    noautoopen: bool = False
    listen_port: int = 7860
    root_dir: str = f"{BASE_ROOT}/rvc-trainer"
    save_dir: str = f"{BASE_ROOT}/rvc-trainer/drive/MyDrive/rvc-trainer"
    
    # Default training parameters
    default_batch_size: int = 8
    default_epochs: int = 150
    default_save_epoch: int = 25
    default_sample_rate: str = "40k"
    default_f0_method: str = "rmvpe"
    default_version: str = "v2"
    
    # Paths
    pretrained_dir: str = "assets/pretrained_v2"
    weights_dir: str = "assets/weights"
    logs_dir: str = "logs"

config = Config()

# ========================================================================== #
# LOGGING CONFIGURATION
# ========================================================================== #

def setup_logging() -> logging.Logger:
    """Configure application-wide logging."""
    logger = logging.getLogger("rvc_trainer")
    logger.setLevel(logging.DEBUG)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler for errors
    try:
        file_handler = logging.FileHandler("rvc_trainer.log")
        file_handler.setLevel(logging.ERROR)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except (IOError, OSError):
        logger.warning("Could not create log file, continuing without file logging")
    
    return logger

logger = setup_logging()

# ========================================================================== #
# GPU DETECTION
# ========================================================================== #

class GPUManager:
    """Manages GPU detection and configuration."""
    
    def __init__(self):
        self._gpus: str = "0"
        self._gpu_info: str = "No GPU detected"
        self._cuda_available: bool = False
        self._device_count: int = 0
        self._detect_gpus()
    
    def _detect_gpus(self) -> None:
        """Detect available GPUs."""
        try:
            import torch
            self._cuda_available = torch.cuda.is_available()
            if self._cuda_available:
                self._device_count = torch.cuda.device_count()
                self._gpus = ",".join(str(i) for i in range(self._device_count))
                self._gpu_info = f"Available GPUs: {self._device_count}"
                logger.info(f"GPU detection successful: {self._gpu_info}")
            else:
                logger.warning("CUDA is not available, running on CPU")
        except ImportError:
            logger.warning("PyTorch not installed, GPU features disabled")
        except Exception as e:
            logger.error(f"Error during GPU detection: {e}")
    
    @property
    def gpus(self) -> str:
        return self._gpus
    
    @property
    def gpu_info(self) -> str:
        return self._gpu_info
    
    @property
    def cuda_available(self) -> bool:
        return self._cuda_available
    
    @property
    def device_count(self) -> int:
        return self._device_count
    
    def validate_gpu_string(self, gpu_str: str) -> Tuple[bool, str]:
        """
        Validate GPU string format.
        
        Args:
            gpu_str: GPU identifier string (e.g., "0", "0-1-2")
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not gpu_str or not gpu_str.strip():
            return False, "GPU string cannot be empty"
        
        # Accept formats: "0", "0-1-2", "0,1,2"
        valid_pattern = r"^[\d,\-]+$"
        if not re.match(valid_pattern, gpu_str.strip()):
            return False, f"Invalid GPU format: {gpu_str}. Use format like '0' or '0-1-2'"
        
        return True, ""

gpu_manager = GPUManager()
gpus = gpu_manager.gpus
gpu_info = gpu_manager.gpu_info
F0GPUVisible = True
default_batch_size = config.default_batch_size

# ========================================================================== #
# UTILITY FUNCTIONS
# ========================================================================== #

class PretrainedModelFinder:
    """Handles finding pretrained model files."""
    
    def __init__(self, pretrained_dir: Optional[str] = None):
        self.pretrained_dir = pretrained_dir or config.pretrained_dir
        self._cache: Dict[str, List[str]] = {}
    
    def _get_pretrained_files(
        self, 
        sr_val: str, 
        letter: str,
        use_cache: bool = True
    ) -> List[str]:
        """
        Get list of pretrained model files matching criteria.
        
        Args:
            sr_val: Sample rate value (e.g., '40k', '32k', '48k')
            letter: Model type letter ('G' for Generator, 'D' for Discriminator)
            use_cache: Whether to use cached results
            
        Returns:
            List of absolute paths to matching .pth files
        """
        cache_key = f"{sr_val}_{letter}"
        
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]
        
        try:
            if not os.path.exists(self.pretrained_dir):
                logger.warning(f"Pretrained directory not found: {self.pretrained_dir}")
                return []
            
            files = [
                os.path.abspath(os.path.join(self.pretrained_dir, file))
                for file in os.listdir(self.pretrained_dir)
                if file.endswith('.pth') and sr_val in file and letter in file
            ]
            
            self._cache[cache_key] = files
            return files
            
        except OSError as e:
            logger.error(f"Error reading pretrained directory: {e}")
            return []
    
    def get_generator_choices(self, sr_val: str) -> List[str]:
        """Get available generator model choices."""
        return self._get_pretrained_files(sr_val, 'G')
    
    def get_discriminator_choices(self, sr_val: str) -> List[str]:
        """Get available discriminator model choices."""
        return self._get_pretrained_files(sr_val, 'D')
    
    def clear_cache(self) -> None:
        """Clear the file cache."""
        self._cache.clear()


# Global instance
pretrained_finder = PretrainedModelFinder()


def validate_model_name(name: str) -> Tuple[bool, str]:
    """
    Validate model name for allowed characters.
    
    Args:
        name: Model name to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not name or not name.strip():
        return False, "Model name cannot be empty"
    
    if not re.match(r"^[a-zA-Z0-9_\-]+$", name.strip()):
        return False, (
            f"Name '{name}' contains invalid characters! "
            "Use only letters, numbers, underscores, and hyphens."
        )
    
    if len(name) > 50:
        return False, "Model name too long (max 50 characters)"
    
    return True, ""


def validate_dataset_folder(folder_path: str) -> Tuple[bool, str]:
    """
    Validate dataset folder exists and contains files.
    
    Args:
        folder_path: Path to dataset folder
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not folder_path or not folder_path.strip():
        return False, "Dataset folder path cannot be empty"
    
    if not os.path.exists(folder_path):
        return False, f"Folder '{folder_path}' does not exist!"
    
    if not os.path.isdir(folder_path):
        return False, f"'{folder_path}' is not a directory!"
    
    try:
        contents = os.listdir(folder_path)
        if not contents:
            return False, f"Folder '{folder_path}' is empty!"
        
        # Check for audio files
        audio_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
        has_audio = any(
            Path(f).suffix.lower() in audio_extensions 
            for f in contents
        )
        if not has_audio:
            logger.warning(f"No common audio files found in {folder_path}")
            
    except PermissionError:
        return False, f"Permission denied accessing '{folder_path}'"
    
    return True, ""


def get_sample_rate(sr: str, version: str) -> str:
    """
    Convert UI sample rate to actual sample rate value.
    
    Args:
        sr: Sample rate from UI ('40k' or '32k')
        version: Model version ('v1' or 'v2')
        
    Returns:
        Actual sample rate string ('48000' or '32000')
    """
    if sr == "40k" or version == "v1":
        return "48000"
    return "32000"


def run_subprocess_command(
    cmd: str, 
    timeout: Optional[int] = None,
    description: str = "command"
) -> Tuple[int, str, str]:
    """
    Execute a subprocess command with proper error handling.
    
    Args:
        cmd: Command string to execute
        timeout: Optional timeout in seconds
        description: Description of the command for logging
        
    Returns:
        Tuple of (exit_code, stdout, stderr)
    """
    logger.info(f"Executing {description}: {cmd[:100]}...")
    
    try:
        result = run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=os.getcwd()
        )
        
        if result.returncode != 0:
            logger.error(f"{description} failed with exit code {result.returncode}")
            if result.stderr:
                logger.error(f"STDERR: {result.stderr[:500]}")
        else:
            logger.info(f"{description} completed successfully")
            
        return result.returncode, result.stdout, result.stderr
        
    except subprocess.TimeoutExpired:
        logger.error(f"{description} timed out after {timeout} seconds")
        return -1, "", f"Command timed out after {timeout} seconds"
    except Exception as e:
        logger.error(f"Error executing {description}: {e}")
        return -1, "", str(e)


# Import subprocess module properly
import subprocess as sp


# ========================================================================== #
# HUGGINGFACE UPLOAD FUNCTIONALITY
# ========================================================================== #

class HuggingFaceUploader:
    """
    Handles uploading trained RVC models to Hugging Face Hub.
    
    Features:
    - Auto-create repository if not exists
    - Auto-generate README with RVC Trainer attribution
    - Auto-detect zip file from model name
    - Support for private/public repositories
    """
    
    README_TEMPLATE = '''---
license: mit
tags:
- rvc
- voice-conversion
- tts
pipeline_tag: text-to-speech
---

# {model_name}

This model was uploaded with [RVC TRAINER](https://github.com/sonic-coders/rvc-trainer)

## Model Information
- **Model Name**: {model_name}
- **Uploaded Date**: {upload_date}
- **Sample Rate**: {sample_rate}kHz
- **Version**: {version}

## Usage
This is an RVC (Retrieval-based Voice Conversion) model trained using [RVC Trainer](https://github.com/sonic-coders/rvc-trainer).

For usage instructions, please refer to the [RVC documentation](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion).

## Files Included
{files_list}
'''
    
    def __init__(self):
        self.api: Optional[HfApi] = None
        self._authenticated: bool = False
    
    def authenticate(self, api_key: str) -> Tuple[bool, str]:
        """
        Authenticate with HuggingFace using API key.
        
        Args:
            api_key: HuggingFace API token
            
        Returns:
            Tuple of (success, message)
        """
        if not HF_AVAILABLE:
            return False, "huggingface_hub library is not installed. Run: pip install huggingface_hub"
        
        if not api_key or not api_key.strip():
            return False, "API Key cannot be empty"
        
        try:
            # Validate API key by getting user info
            self.api = HfApi(token=api_key.strip())
            user_info = whoami(token=api_key.strip())
            self._authenticated = True
            username = user_info.get('name', 'Unknown')
            logger.info(f"HuggingFace authentication successful for user: {username}")
            return True, f"✅ Authenticated as: {username}"
            
        except Exception as e:
            self._authenticated = False
            error_msg = str(e)
            if "401" in error_msg or "Invalid" in error_msg:
                return False, "❌ Invalid API Key. Please check your token."
            elif "403" in error_msg:
                return False, "❌ API Key does not have write permission."
            else:
                logger.error(f"HuggingFace auth error: {e}")
                return False, f"❌ Authentication failed: {error_msg}"
    
    def find_model_zip(self, model_name: str) -> Tuple[Optional[str], str]:
        """
        Find the zip file for a trained model.
        
        Search locations:
        1. assets/weights/{model_name}/
        2. logs/{model_name}/
        3. {save_dir}/{model_name}/
        
        Args:
            model_name: Name of the trained model
            
        Returns:
            Tuple of (zip_path or None, message)
        """
        search_paths = [
            f'{config.weights_dir}/{model_name}',
            f'{config.logs_dir}/{model_name}',
            f'{config.save_dir}/{model_name}',
        ]
        
        # Search for .zip files first
        for search_path in search_paths:
            if os.path.exists(search_path):
                zip_files = glob.glob(f'{search_path}/*.zip')
                if zip_files:
                    # Return the most recent zip file
                    latest_zip = max(zip_files, key=os.path.getmtime)
                    msg = f"Found zip: {os.path.basename(latest_zip)}"
                    logger.info(msg)
                    return latest_zip, msg
        
        # If no zip found, check if we can create one from the folder
        for search_path in search_paths:
            if os.path.exists(search_path) and os.listdir(search_path):
                # Found folder with files but no zip - we'll zip it during upload
                msg = f"Will create zip from: {search_path}"
                logger.info(msg)
                return search_path, msg
        
        # No files found anywhere
        error_msg = (
            f"No model files found for '{model_name}'. "
            f"Searched in: {', '.join(search_paths)}"
        )
        logger.warning(error_msg)
        return None, error_msg
    
    def create_readme_content(
        self,
        model_name: str,
        sample_rate: str = "40k",
        version: str = "v2",
        files_list: Optional[List[str]] = None
    ) -> str:
        """
        Generate README content for the model repository.
        
        Args:
            model_name: Name of the model
            sample_rate: Sample rate used
            version: RVC version
            files_list: List of files included
            
        Returns:
            Formatted README content
        """
        # Format files list
        if files_list:
            files_md = "\n".join([f"- `{f}`" for f in files_list])
        else:
            files_md = "- Model weights\n- Index file\n- Configuration"
        
        return self.README_TEMPLATE.format(
            model_name=model_name,
            upload_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC"),
            sample_rate=sample_rate,
            version=version,
            files_list=files_md
        )
    
    def ensure_repo_exists(
        self,
        repo_id: str,
        private: bool = False,
        token: Optional[str] = None
    ) -> Tuple[bool, str]:
        """
        Ensure repository exists, create if it doesn't.
        
        Args:
            repo_id: Full repo ID (username/repo_name)
            private: Whether repo should be private
            token: HF API token
            
        Returns:
            Tuple of (success, message)
        """
        if not self.api:
            return False, "Not authenticated. Please enter API key first."
        
        try:
            # Check if repo exists
            try:
                repo_info = self.api.repo_info(repo_id=repo_id, token=token)
                logger.info(f"Repository exists: {repo_id}")
                return True, f"✅ Repository already exists: {repo_id}"
            except Exception:
                # Repo doesn't exist, create it
                pass
            
            # Create new repository
            logger.info(f"Creating new repository: {repo_id}")
            create_repo(
                repo_id=repo_id,
                token=token,
                private=private,
                repo_type="model",
                exist_ok=True
            )
            
            return True, f"✅ Created new repository: {repo_id}"
            
        except Exception as e:
            logger.error(f"Error creating repository: {e}")
            return False, f"❌ Error creating repository: {str(e)}"
    
    def upload_model(
        self,
        model_name: str,
        repo_id: str,
        api_key: str,
        sample_rate: str = "40k",
        version: str = "v2",
        private: bool = False
    ) -> str:
        """
        Upload model to HuggingFace.
        
        Complete workflow:
        1. Authenticate
        2. Find/create zip file
        3. Ensure repo exists
        4. Create README
        5. Upload files
        
        Args:
            model_name: Name of the trained model
            repo_id: Target HF repository (username/repo_name)
            api_key: HuggingFace API key
            sample_rate: Sample rate of the model
            version: RVC version
            private: Whether to make repo private
            
        Returns:
            Status message with result or error
        """
        # Step 1: Authenticate
        success, msg = self.authenticate(api_key)
        if not success:
            return msg
        
        # Step 2: Find model files
        zip_path, msg = self.find_model_zip(model_name)
        if zip_path is None:
            return f"❌ {msg}"
        
        logger.info(f"Model path: {zip_path}")
        
        # Step 3: Ensure repository exists
        success, msg = self.ensure_repo_exists(repo_id, private, api_key.strip())
        if not success:
            return msg
        
        # Step 4 & 5: Prepare and upload
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                # If it's a folder, zip it first
                if os.path.isdir(zip_path):
                    zip_filename = f"{model_name}.zip"
                    zip_filepath = os.path.join(temp_dir, zip_filename)
                    
                    logger.info(f"Creating zip archive: {zip_filename}")
                    with zipfile.ZipFile(zip_filepath, 'w', zipfile.ZIP_DEFLATED) as zipf:
                        for root, dirs, files in os.walk(zip_path):
                            for file in files:
                                file_path = os.path.join(root, file)
                                arcname = os.path.relpath(file_path, zip_path)
                                zipf.write(file_path, arcname)
                    
                    upload_dir = temp_dir
                    files_in_upload = [zip_filename]
                else:
                    # It's already a zip file, copy to temp dir
                    shutil.copy2(zip_path, temp_dir)
                    upload_dir = temp_dir
                    files_in_upload = [os.path.basename(zip_path)]
                
                # Create README
                readme_content = self.create_readme_content(
                    model_name=model_name,
                    sample_rate=sample_rate,
                    version=version,
                    files_list=files_in_upload
                )
                readme_path = os.path.join(temp_dir, "README.md")
                with open(readme_path, 'w', encoding='utf-8') as f:
                    f.write(readme_content)
                
                # Upload to HuggingFace
                logger.info(f"Uploading to {repo_id}...")
                
                api = HfApi(token=api_key.strip())
                
                # Upload all files in temp directory
                upload_result = api.upload_folder(
                    folder_path=temp_dir,
                    repo_id=repo_id,
                    repo_type="model"
                )
                
                hf_url = f"https://huggingface.co/{repo_id}"
                success_msg = (
                    f"✅ Successfully uploaded to HuggingFace!\n\n"
                    f"📦 Repository: [{repo_id}]({hf_url})\n"
                    f"🔗 URL: {hf_url}\n\n"
                    f"Files uploaded:\n" + "\n".join([f"  ✅ {f}" for f in files_in_upload]) + "\n"
                    f"  ✅ README.md (auto-generated)"
                )
                
                logger.info(success_msg)
                return success_msg
                
        except Exception as e:
            logger.exception(f"Error uploading to HuggingFace: {e}")
            return f"❌ Error during upload: {str(e)}"
    
    @staticmethod
    def validate_repo_id(repo_id: str) -> Tuple[bool, str]:
        """
        Validate HuggingFace repository ID format.
        
        Valid format: username/repo-name
        
        Args:
            repo_id: Repository ID to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not repo_id or not repo_id.strip():
            return False, "Repository ID cannot be empty"
        
        repo_id = repo_id.strip()
        
        # Must contain exactly one /
        if repo_id.count('/') != 1:
            return False, 'Repository ID must be in format: "username/repo-name"'
        
        username, repo_name = repo_id.split('/')
        
        # Validate username
        if not username or not re.match(r'^[a-zA-Z0-9_-]+$', username):
            return False, "Invalid username format. Use only letters, numbers, hyphens, underscores."
        
        # Validate repo name
        if not repo_name or not re.match(r'^[a-zA-Z0-9._-]+$', repo_name):
            return False, "Invalid repository name format. Use only letters, numbers, dots, hyphens, underscores."
        
        if len(repo_id) > 200:
            return False, "Repository ID too long (max 200 characters)"
        
        return True, ""


# Global instance
hf_uploader = HuggingFaceUploader()

# ========================================================================== #
# UI CALLBACK FUNCTIONS
# ========================================================================== #

def change_f0_method(f0_method: str) -> str:
    """
    Update GPU visibility based on F0 method selection.
    
    Args:
        f0_method: Selected F0 extraction method
        
    Returns:
        GPU string for the method
    """
    if f0_method == "rmvpe_gpu":
        return gpus
    return "0"


def change_sr2(sr: str, if_f0: str, version: str) -> Tuple[gr.update, gr.update]:
    """
    Update pretrained model choices when sample rate changes.
    
    Args:
        sr: Selected sample rate
        if_f0: Whether F0 is enabled
        version: Model version
        
    Returns:
        Tuple of Gradio updates for G and D dropdowns
    """
    sr2_val = "40k" if version == "v1" else sr
    
    g_choices = pretrained_finder.get_generator_choices(sr2_val)
    d_choices = pretrained_finder.get_discriminator_choices(sr2_val)
    
    return (
        gr.update(
            choices=g_choices, 
            value=g_choices[0] if g_choices else ''
        ),
        gr.update(
            choices=d_choices, 
            value=d_choices[0] if d_choices else ''
        )
    )


def change_version19(sr: str, if_f0: str, version: str) -> Tuple[gr.update, gr.update, gr.update]:
    """
    Handle version change and update related options.
    
    Args:
        sr: Selected sample rate
        if_f0: Whether F0 is enabled
        version: Model version ('v1' or 'v2')
        
    Returns:
        Tuple of Gradio updates for G, D dropdowns, and SR visibility
    """
    if version == "v1":
        sr2_val = "40k"
        sr_update = gr.update(value="40k", visible=False)
    else:
        sr2_val = sr
        sr_update = gr.update(visible=True)
    
    g_choices = pretrained_finder.get_generator_choices(sr2_val)
    d_choices = pretrained_finder.get_discriminator_choices(sr2_val)
    
    return (
        gr.update(
            choices=g_choices, 
            value=g_choices[0] if g_choices else ''
        ),
        gr.update(
            choices=d_choices, 
            value=d_choices[0] if d_choices else ''
        ),
        sr_update
    )


def change_f0(if_f0: bool, sr: str, version: str) -> Tuple[gr.update, gr.update, gr.update]:
    """
    Update F0 method choices based on singing mode.
    
    Args:
        if_f0: Whether model will be used for singing
        sr: Selected sample rate
        version: Model version
        
    Returns:
        Tuple of Gradio updates for F0 method, G, and D dropdowns
    """
    if if_f0:
        f0_choices = ["rmvpe", "hpa-rmvpe"]
        f0_value = "rmvpe_gpu"
    else:
        f0_choices = ["crepe", "rmvpe", "hpa-rmvpe"]
        f0_value = "crepe"
    
    sr_val = "40k" if version == "v1" else sr
    g_choices = pretrained_finder.get_generator_choices(sr_val)
    d_choices = pretrained_finder.get_discriminator_choices(sr_val)
    
    return (
        gr.update(choices=f0_choices, value=f0_value),
        gr.update(choices=g_choices, value=g_choices[0] if g_choices else ''),
        gr.update(choices=d_choices, value=d_choices[0] if d_choices else '')
    )


# ========================================================================== #
# STEP 2: DATA PROCESSING FUNCTIONS
# ========================================================================== #

def preprocess_dataset(
    dataset_folder: str, 
    training_name: str, 
    sr2: str, 
    np7: float
) -> str:
    """
    Preprocess dataset for training.
    
    This function handles:
    - Model name validation
    - Dataset folder validation
    - Audio segmentation and resampling
    
    Args:
        dataset_folder: Path to the dataset folder
        training_name: Name for the training model
        sr2: Target sample rate ('40k' or '32k')
        np7: Number of CPU processes for pitch extraction
        
    Returns:
        Status message indicating success or failure
    """
    try:
        # Validate inputs
        valid, error_msg = validate_model_name(training_name)
        if not valid:
            return f"❌ Error: {error_msg}"
        
        valid, error_msg = validate_dataset_folder(dataset_folder)
        if not valid:
            return f"❌ Error: {error_msg}"
        
        # Create output directory
        model_dir = f'{config.save_dir}/{training_name}'
        os.makedirs(model_dir, exist_ok=True)
        logger.info(f"Created/verified model directory: {model_dir}")
        
        # Convert parameters
        sample_rate = get_sample_rate(sr2, "v2")  # Use v2 for preprocessing
        percentage = 3.0
        normalize = True
        
        # Build and execute preprocess command
        preprocess_script = f"{config.root_dir}/rvc/train/preprocess/preprocess.py"
        cmd = (
            f"python \"{preprocess_script}\" "
            f"\"{config.save_dir}/{training_name}\" "
            f"\"{dataset_folder}\" "
            f"{percentage} {sample_rate} {normalize}"
        )
        
        exit_code, stdout, stderr = run_subprocess_command(cmd, description="preprocessing")
        
        if exit_code != 0:
            return f"❌ Error during preprocessing! Exit code: {exit_code}\n{stderr[:200]}"
        
        logger.info(f"Preprocessing completed for model: {training_name}")
        return "✅ Data preprocessing completed successfully!"
        
    except Exception as e:
        logger.exception(f"Exception during preprocessing: {e}")
        return f"❌ Error during preprocessing: {str(e)}"


def extract_f0_feature(
    gpus6: str, 
    np7: float, 
    f0method8: str, 
    if_f0_3: bool, 
    training_name: str, 
    version19: str,
    gpus_rmvpe: str
) -> str:
    """
    Extract F0 (pitch) and feature vectors from preprocessed data.
    
    Args:
        gpus6: GPU identifiers for processing
        np7: Number of CPU processes
        f0method8: F0 extraction method
        if_f0_3: Whether F0 extraction is enabled
        training_name: Name of the training model
        version19: Model version
        gpus_rmvpe: GPU identifiers for RMVPE
        
    Returns:
        Status message indicating success or failure
    """
    try:
        # Convert parameters
        sample_rate = get_sample_rate("48k" if version19 == "v2" else "32k", version19)
        arch_fairseq = "Fairseq"
        f0_method = f0method8.replace("_gpu", "")
        
        # Build and execute feature extraction command
        preparing_data_script = f"{config.root_dir}/rvc/train/preprocess/preparing_data.py"
        cmd = (
            f"python \"{preparing_data_script}\" "
            f"\"{config.save_dir}/{training_name}\" "
            f"{arch_fairseq} {f0_method} {sample_rate} 2"
        )
        
        exit_code, stdout, stderr = run_subprocess_command(cmd, description="feature extraction")
        
        if exit_code != 0:
            return f"❌ Error during feature extraction! Exit code: {exit_code}\n{stderr[:200]}"
        
        logger.info(f"Feature extraction completed for model: {training_name}")
        return "✅ Feature extraction completed successfully!"
        
    except Exception as e:
        logger.exception(f"Exception during feature extraction: {e}")
        return f"❌ Error during feature extraction: {str(e)}"


def train_index(training_name: str, version19: str) -> str:
    """
    Train the index file for fast similarity search during inference.
    
    Args:
        training_name: Name of the training model
        version19: Model version (affects index algorithm choice)
        
    Returns:
        Status message indicating success or failure
    """
    try:
        index_algorithm = "Faiss"
        
        # Build and execute index training command
        extract_index_script = f"{config.root_dir}/rvc/train/preprocess/extract_index.py"
        cmd = (
            f"python \"{extract_index_script}\" "
            f"\"{config.save_dir}/{training_name}\" {index_algorithm}"
        )
        
        exit_code, stdout, stderr = run_subprocess_command(cmd, description="index training")
        
        if exit_code != 0:
            return f"❌ Error during index training! Exit code: {exit_code}\n{stderr[:200]}"
        
        logger.info(f"Index training completed for model: {training_name}")
        return "✅ Index training completed successfully!"
        
    except Exception as e:
        logger.exception(f"Exception during index training: {e}")
        return f"❌ Error during index training: {str(e)}"


# ========================================================================== #
# STEP 3: MODEL TRAINING FUNCTIONS
# ========================================================================== #

# Unwanted log patterns to filter out
UNWANTED_LOG_PATTERNS = [
    "All log messages before absl::InitializeLog()",
    "Unable to register cuDNN factory",
    "Unable to register cuBLAS factory",
    "computation placer already registered"
]


def filter_training_output(line: str) -> bool:
    """
    Filter unwanted lines from training output.
    
    Args:
        line: Line of output to check
        
    Returns:
        True if line should be included, False otherwise
    """
    return not any(pattern in line for pattern in UNWANTED_LOG_PATTERNS)


def click_train(
    training_name: str,
    sr2: str,
    if_f0_3: bool,
    spk_id5: int,
    save_epoch10: int,
    total_epoch11: int,
    batch_size12: int,
    if_save_latest13: str,
    pretrained_G14: Optional[str],
    pretrained_D15: Optional[str],
    gpus16: str,
    if_cache_gpu17: str,
    if_save_every_weights18: str,
    version19: str,
    # Additional parameters from Settings tab
    optimizer: str = "AdamW",
    vocoder: str = "HiFi-GAN",
    save_half: bool = True,
    save_to_zip: bool = True,
) -> str:
    """
    Execute model training with the specified parameters.
    
    This is the main training function that:
    - Validates all input parameters
    - Builds the training command
    - Monitors training progress
    - Handles errors gracefully
    
    Args:
        training_name: Name for the model
        sr2: Target sample rate
        if_f0_3: Whether F0 is enabled
        spk_id5: Speaker ID
        save_epoch10: Epoch interval for saving checkpoints
        total_epoch11: Total number of epochs to train
        batch_size12: Training batch size
        if_save_latest13: Whether to save latest checkpoint
        pretrained_G14: Path to pretrained generator (optional)
        pretrained_D15: Path to pretrained discriminator (optional)
        gpus16: GPU identifiers to use
        if_cache_gpu17: Whether to cache dataset on GPU
        if_save_every_weights18: Whether to save weights at each checkpoint
        version19: Model version
        optimizer: Optimizer to use (from Settings)
        vocoder: Vocoder type (from Settings)
        save_half: Whether to save in half precision (from Settings)
        save_to_zip: Whether to package as ZIP (from Settings)
        
    Returns:
        Status message with training results or error information
    """
    try:
        # Validate model name
        valid, error_msg = validate_model_name(training_name)
        if not valid:
            return f"❌ Error: {error_msg}"
        
        # Validate GPU string
        valid, error_msg = gpu_manager.validate_gpu_string(gpus16)
        if not valid:
            return f"❌ Error: {error_msg}"
        
        # Convert and validate parameters
        sample_rate = get_sample_rate(sr2, version19)
        
        try:
            save_epoch_interval = int(save_epoch10)
            total_epochs = int(total_epoch11)
            batch_size = int(batch_size12)
            spk_id = int(spk_id5)
        except ValueError as e:
            return f"❌ Error: Invalid numeric parameter: {e}"
        
        # Validate parameter ranges
        if total_epochs < 1:
            return "❌ Error: Total epochs must be at least 1"
        if batch_size < 1:
            return "❌ Error: Batch size must be at least 1"
        if save_epoch_interval < 1:
            return "❌ Error: Save epoch interval must be at least 1"
        
        # Log training configuration
        logger.info("=" * 50)
        logger.info("Starting Training Session")
        logger.info("=" * 50)
        logger.info(f"Model: {training_name}")
        logger.info(f"Version: {version19}")
        logger.info(f"Sample Rate: {sample_rate}Hz")
        logger.info(f"Epochs: {total_epochs}")
        logger.info(f"Batch Size: {batch_size}")
        logger.info(f"Save Every: {save_epoch_interval} epochs")
        logger.info(f"Optimizer: {optimizer}")
        logger.info(f"Vocoder: {vocoder}")
        logger.info(f"GPUs: {gpus16}")
        logger.info("=" * 50)
        
        # Handle pretrained models
        pretrained_G = pretrained_G14 if pretrained_G14 else None
        pretrained_D = pretrained_D15 if pretrained_D15 else None
        
        # Validate pretrained model paths if provided
        if pretrained_G and not os.path.exists(pretrained_G):
            logger.warning(f"Pretrained Generator path does not exist: {pretrained_G}")
        if pretrained_D and not os.path.exists(pretrained_D):
            logger.warning(f"Pretrained Discriminator path does not exist: {pretrained_D}")
        
        # Build training command
        train_script = f"{config.root_dir}/rvc/train/train.py"
        cmd_parts = [
            f'python "{train_script}"',
            f'--experiment_dir "{config.save_dir}"',
            f'--model_name "{training_name}"',
            f'--batch_size {batch_size}',
            f'--sample_rate {sample_rate}',
            f'--total_epoch {total_epochs}',
            f'--save_every_epoch {save_epoch_interval}',
            f'--vocoder "{vocoder}"',
            f'--optimizer {optimizer}',
            f'--save_to_zip {save_to_zip}',
            f'--save_half {save_half}',
        ]
        
        if pretrained_G is not None:
            cmd_parts.append(f'--pretrain_g "{pretrained_G}"')
        if pretrained_D is not None:
            cmd_parts.append(f'--pretrain_d "{pretrained_D}"')
        
        cmd = " ".join(cmd_parts)
        logger.debug(f"Full training command: {cmd}")
        
        # Execute training process
        p = Popen(
            cmd,
            bufsize=1,
            text=True,
            shell=True,
            stdout=PIPE,
            stderr=STDOUT,
            cwd=os.getcwd(),
            universal_newlines=True,
        )
        
        # Capture and filter output
        output_lines = []
        for line in p.stdout:
            line = line.strip()
            if filter_training_output(line):
                output_lines.append(line)
                print(line)  # Still print to console for real-time feedback
        
        p.wait()
        
        if p.returncode != 0:
            error_msg = f"❌ Error during training! Exit code: {p.returncode}"
            logger.error(error_msg)
            
            # Save detailed error log
            try:
                error_log_path = f"{config.save_dir}/{training_name}/error_log.txt"
                os.makedirs(os.path.dirname(error_log_path), exist_ok=True)
                with open(error_log_path, "w") as f:
                    f.write(f"Training failed with exit code: {p.returncode}\n\n")
                    f.write("Output:\n")
                    f.write("\n".join(output_lines[-100:]))  # Last 100 lines
                    f.write("\n\nTraceback:\n")
                    f.write(traceback.format_exc())
            except Exception as e:
                logger.warning(f"Could not write error log: {e}")
            
            return error_msg
        
        logger.info(f"Training completed successfully for model: {training_name}")
        return "✅ Training completed successfully!"
        
    except Exception as e:
        logger.exception(f"Unexpected exception during training: {e}")
        
        # Save error details
        try:
            error_log_path = f"{config.save_dir}/{training_name}/error_log.txt"
            os.makedirs(os.path.dirname(error_log_path), exist_ok=True)
            with open(error_log_path, "w") as f:
                f.write("An unexpected error occurred:\n")
                f.write(traceback.format_exc())
        except Exception:
            pass
            
        return f"❌ Error during training: {str(e)}"


def train1key(
    training_name: str,
    sr2: str,
    if_f0_3: bool,
    dataset_folder: str,
    spk_id5: int,
    np7: float,
    f0method8: str,
    save_epoch10: int,
    total_epoch11: int,
    batch_size12: int,
    if_save_latest13: str,
    pretrained_G14: Optional[str],
    pretrained_D15: Optional[str],
    gpus16: str,
    if_cache_gpu17: str,
    if_save_every_weights18: str,
    version19: str,
    gpus_rmvpe: str,
) -> str:
    """
    Execute complete one-click training pipeline.
    
    Runs all training steps in sequence:
    1. Data preprocessing
    2. Feature extraction
    3. Index training
    4. Model training
    
    Args:
        All parameters from click_train plus:
        dataset_folder: Path to dataset
        np7: CPU processes for pitch extraction
        f0method8: F0 extraction method
        gpus_rmvpe: GPUs for RMVPE
        
    Returns:
        Combined status message from all steps
    """
    logger.info(f"Starting one-click training pipeline for: {training_name}")
    
    results = []
    
    # Step 1: Preprocess
    logger.info("Step 1/4: Preprocessing dataset...")
    result1 = preprocess_dataset(dataset_folder, training_name, sr2, np7)
    results.append(result1)
    if "❌" in result1:
        logger.error(f"Pipeline stopped at preprocessing: {result1}")
        return result1
    
    # Step 2: Extract features
    logger.info("Step 2/4: Extracting features...")
    result2 = extract_f0_feature(
        gpus16, np7, f0method8, if_f0_3, training_name, version19, gpus_rmvpe
    )
    results.append(result2)
    if "❌" in result2:
        logger.error(f"Pipeline stopped at feature extraction: {result2}")
        return f"{result1}\n\n{result2}"
    
    # Step 3: Train index
    logger.info("Step 3/4: Training index...")
    result3 = train_index(training_name, version19)
    results.append(result3)
    if "❌" in result3:
        logger.error(f"Pipeline stopped at index training: {result3}")
        return f"{result1}\n\n{result2}\n\n{result3}"
    
    # Step 4: Train model
    logger.info("Step 4/4: Training model...")
    result4 = click_train(
        training_name, sr2, if_f0_3, spk_id5, save_epoch10, total_epoch11,
        batch_size12, if_save_latest13, pretrained_G14, pretrained_D15,
        gpus16, if_cache_gpu17, if_save_every_weights18, version19,
    )
    results.append(result4)
    
    # Combine all results
    final_message = "✅ All steps completed!\n\n" + "\n\n".join(results)
    logger.info("One-click training pipeline completed!")
    
    return final_message


def download_model_files(training_name: str) -> Tuple[List[str], str]:
    """
    Find and list downloadable model files.
    
    Args:
        training_name: Name of the trained model
        
    Returns:
        Tuple of (list of file paths, status message)
    """
    try:
        weights_path = f'{config.weights_dir}/{training_name}'
        logs_path = f'{config.logs_dir}/{training_name}'
        
        files = []
        
        # Check weights directory
        if os.path.exists(weights_path):
            try:
                files.extend([
                    os.path.join(weights_path, f) 
                    for f in os.listdir(weights_path) 
                    if f.endswith('.pth')
                ])
            except PermissionError:
                logger.warning(f"Permission denied reading: {weights_path}")
        
        # Check logs directory for index files
        if os.path.exists(logs_path):
            try:
                files.extend(glob.glob(f'{logs_path}/added_*.index'))
            except PermissionError:
                logger.warning(f"Permission denied reading: {logs_path}")
        
        if not files:
            msg = f"No model files found for '{training_name}'"
            logger.warning(msg)
            return [], msg
        
        logger.info(f"Found {len(files)} model files for {training_name}")
        return files, f"Found {len(files)} files"
        
    except Exception as e:
        logger.exception(f"Error finding model files: {e}")
        return [], f"Error: {str(e)}"


def handle_upload(files: List[Any], folder: str) -> str:
    """
    Handle file upload to dataset folder.
    
    Args:
        files: List of uploaded file objects
        folder: Target folder path
        
    Returns:
        Status message
    """
    if not folder or not folder.strip():
        gr.Warning('Please enter a folder name for your dataset')
        return "Please enter a folder name for your dataset"
    
    try:
        os.makedirs(folder, exist_ok=True)
        uploaded_count = 0
        
        for f in files:
            if hasattr(f, 'name') and os.path.exists(f.name):
                dest = os.path.join(folder, os.path.basename(f.name))
                shutil.copy2(f.name, dest)
                uploaded_count += 1
            else:
                logger.warning(f"Invalid file object: {f}")
        
        msg = f"Uploaded {uploaded_count} files to {folder}"
        logger.info(msg)
        return msg
        
    except Exception as e:
        logger.exception(f"Error uploading files: {e}")
        return f"❌ Error uploading files: {str(e)}"


# ========================================================================== #
# GRADIO UI - UNCHANGED STRUCTURE
# ========================================================================== #

with gr.Blocks(
    title="🔊 RVC Trainer",
    theme=gr.themes.Base(primary_hue="rose", neutral_hue="zinc")
) as app:
    with gr.Row():
        with gr.Column():
            gr.Markdown("# 🔊 RVC Voice Trainer")
            gr.Markdown("Train high-quality voice models using RVC technology")

    with gr.Tabs():
        # ===================== TRAIN TAB =====================
        with gr.TabItem("Train"):
            with gr.Row():
                # Left Column - Data Processing
                with gr.Column(scale=1):
                    gr.Markdown("### 📁 Dataset & Preprocessing")
                    
                    training_name = gr.Textbox(
                        label="Model Name",
                        value="My-Voice",
                        placeholder="Enter model name (a-z, 0-9, _ only)"
                    )
                    
                    dataset_folder = gr.Textbox(
                        label="Dataset Folder",
                        value='dataset',
                        placeholder="/path/to/dataset"
                    )
                    
                    easy_uploader = gr.File(
                        label="Drop your audio files here",
                        file_types=['audio'],
                        file_count="multiple"
                    )
                    
                    upload_status = gr.Textbox(label="Upload Status", value="", visible=True)
                    
                    easy_uploader.change(
                        fn=handle_upload,
                        inputs=[easy_uploader, dataset_folder],
                        outputs=[upload_status]
                    )
                    
                    np7 = gr.Slider(
                        minimum=0,
                        maximum=config.n_cpu,
                        step=1,
                        label="CPU Processes for Pitch Extraction",
                        value=int(np.ceil(config.n_cpu / 1.5)),
                        interactive=True,
                    )
                    
                    sr2 = gr.Radio(
                        label="Sampling Rate",
                        choices=["40k", "32k"],
                        value="40k",
                        interactive=True,
                    )
                    
                    f0method8 = gr.Radio(
                        label="F0 Extraction Method",
                        choices=["rmvpe", "hpa-rmvpe"],
                        value="rmvpe",
                        interactive=True,
                    )
                    
                    gpus_rmvpe = gr.Textbox(
                        label="GPU numbers (e.g., 0-1-2)",
                        value=gpus,
                        interactive=True,
                    )
                    
                    but1 = gr.Button("1. Process Data", variant="primary")
                    info1 = gr.Textbox(label="Status", value="", visible=True, max_lines=5)

                # Middle Column - Feature Extraction
                with gr.Column(scale=1):
                    gr.Markdown("### 🎵 Feature Extraction")
                    
                    version19 = gr.Radio(
                        label="Version",
                        choices=["v1", "v2"],
                        value="v2",
                        interactive=True,
                    )
                    
                    if_f0_3 = gr.Radio(
                        label="Will model be used for singing?",
                        choices=[True, False],
                        value=True,
                        interactive=True,
                    )
                    
                    gpus6 = gr.Textbox(
                        label="GPU numbers (e.g., 0-1-2)",
                        value=gpus,
                        interactive=True,
                    )
                    
                    gpu_info9 = gr.Textbox(
                        label="GPU Info",
                        value=gpu_info,
                        interactive=False,
                    )
                    
                    spk_id5 = gr.Slider(
                        minimum=0,
                        maximum=4,
                        step=1,
                        label="Speaker ID",
                        value=0,
                        interactive=True,
                    )
                    
                    but2 = gr.Button("2. Extract Features", variant="primary")
                    info2 = gr.Textbox(label="Status", value="", max_lines=5)

                # Right Column - Training
                with gr.Column(scale=1):
                    gr.Markdown("### 🚀 Training")
                    
                    total_epoch11 = gr.Slider(
                        minimum=2,
                        maximum=1000,
                        step=1,
                        label="Epochs (more = better quality)",
                        value=150,
                        interactive=True,
                    )
                    
                    but4 = gr.Button("3. Train Index", variant="primary")
                    but3 = gr.Button("4. Train Model", variant="primary")
                    
                    info3 = gr.Textbox(label="Status", value="", max_lines=10)
                    
                    with gr.Accordion(label="Advanced Settings", open=False):
                        gpus16 = gr.Textbox(
                            label="GPUs (e.g., 0-1-2)",
                            value="0",
                            interactive=True,
                        )
                        
                        save_epoch10 = gr.Slider(
                            minimum=1,
                            maximum=50,
                            step=1,
                            label="Save Frequency (epochs)",
                            value=25,
                            interactive=True,
                        )
                        
                        batch_size12 = gr.Slider(
                            minimum=2,
                            maximum=16,
                            step=2,
                            label="Batch Size",
                            value=default_batch_size,
                            interactive=True,
                        )
                        
                        if_cache_gpu17 = gr.Radio(
                            label="Cache dataset to GPU for faster training",
                            choices=["yes", "no"],
                            value="no",
                            interactive=True,
                        )
                        
                        if_save_every_weights18 = gr.Radio(
                            label="Save small model at every save point",
                            choices=["yes", "no"],
                            value="yes",
                            interactive=True,
                        )
                        
                        with gr.Accordion(label="Pretrained Models", open=False):
                            pretrained_G14 = gr.Dropdown(
                                label="Pretrained Generator",
                                choices=[],
                                value=None,
                                interactive=True,
                            )
                            
                            pretrained_D15 = gr.Dropdown(
                                label="Pretrained Discriminator",
                                choices=[],
                                value=None,
                                interactive=True,
                            )

                    with gr.Row():
                        but5 = gr.Button("🌟 One Click Training", variant="primary", size="lg")

                    with gr.Row():
                        download_model = gr.Button("5. Download Model", variant="secondary")
                        model_files = gr.Files(label="Model Files")

                    # Update pretrained dropdowns when sampling rate changes
                    def update_pretrained(sr_val, f0_val, version_val):
                        sr = "40k" if version_val == "v1" else sr_val
                        g_choices = pretrained_finder.get_generator_choices(sr)
                        d_choices = pretrained_finder.get_discriminator_choices(sr)
                        return (
                            gr.update(choices=g_choices, value=g_choices[0] if g_choices else None),
                            gr.update(choices=d_choices, value=d_choices[0] if d_choices else None)
                        )
                    
                    sr2.change(
                        update_pretrained,
                        [sr2, f0method8, version19],
                        [pretrained_G14, pretrained_D15]
                    )
                    
                    version19.change(
                        update_pretrained,
                        [sr2, f0method8, version19],
                        [pretrained_G14, pretrained_D15]
                    )

            # ===================== EVENT HANDLERS =====================

            # Button 1: Process Data
            but1.click(
                preprocess_dataset,
                inputs=[dataset_folder, training_name, sr2, np7],
                outputs=[info1]
            )

            # Button 2: Extract Features
            but2.click(
                extract_f0_feature,
                inputs=[gpus6, np7, f0method8, if_f0_3, training_name, version19, gpus_rmvpe],
                outputs=[info2]
            )

            # Button 3: Train Index
            but4.click(
                train_index,
                inputs=[training_name, version19],
                outputs=[info3]
            )

            # Button 4: Train Model
            but3.click(
                click_train,
                inputs=[
                    training_name,
                    sr2,
                    if_f0_3,
                    spk_id5,
                    save_epoch10,
                    total_epoch11,
                    batch_size12,
                    gr.State("yes"),  # if_save_latest13
                    pretrained_G14,
                    pretrained_D15,
                    gpus16,
                    if_cache_gpu17,
                    if_save_every_weights18,
                    version19,
                ],
                outputs=[info3]
            )

            # Button 5: One Click Training
            but5.click(
                train1key,
                inputs=[
                    training_name,
                    sr2,
                    if_f0_3,
                    dataset_folder,
                    spk_id5,
                    np7,
                    f0method8,
                    save_epoch10,
                    total_epoch11,
                    batch_size12,
                    gr.State("yes"),  # if_save_latest13
                    pretrained_G14,
                    pretrained_D15,
                    gpus16,
                    if_cache_gpu17,
                    if_save_every_weights18,
                    version19,
                    gpus_rmvpe,
                ],
                outputs=[info3]
            )

            # Download Model
            download_model.click(
                download_model_files,
                inputs=[training_name],
                outputs=[model_files, info3]
            )

            # F0 method change
            f0method8.change(
                change_f0_method,
                inputs=[f0method8],
                outputs=[gpus_rmvpe]
            )

        # ===================== HUGGINGFACE TAB =====================
        with gr.TabItem("🤗 HuggingFace"):
            gr.Markdown("### 🚀 Push Model to HuggingFace")
            gr.Markdown("Upload your trained model to HuggingFace Hub for easy sharing and deployment.")
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("#### 🔑 Authentication")
                    
                    hf_api_key = gr.Textbox(
                        label="HuggingFace API Key",
                        type="password",
                        placeholder="hf_... (Get from https://huggingface.co/settings/tokens)",
                        interactive=True
                    )
                    
                    auth_button = gr.Button("🔐 Verify API Key", variant="secondary")
                    auth_status = gr.Textbox(label="Auth Status", value="", max_lines=3)
                
                with gr.Column(scale=1):
                    gr.Markdown("#### 📦 Repository Settings")
                    
                    hf_repo_id = gr.Textbox(
                        label="Repository ID",
                        placeholder="username/model-name",
                        value="",
                        interactive=True
                    )
                    
                    hf_private = gr.Checkbox(
                        label="Private Repository",
                        value=False,
                        interactive=True
                    )
                    
                    # Auto-detect info
                    auto_repo_info = gr.Textbox(
                        label="Auto-detected Info",
                        value="Enter model name to detect zip path",
                        interactive=False
                    )
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### 🎯 Model Selection")
                    
                    # Model name input (can be same as training name)
                    hf_model_name = gr.Textbox(
                        label="Model Name (for zip detection)",
                        placeholder="My-Voice",
                        value="",
                        interactive=True
                    )
                    
                    # Sample rate for metadata
                    hf_sample_rate = gr.Radio(
                        label="Sample Rate",
                        choices=["40k", "32k"],
                        value="40k",
                        interactive=True
                    )
                    
                    hf_version = gr.Radio(
                        label="RVC Version",
                        choices=["v1", "v2"],
                        value="v2",
                        interactive=True
                    )
                
                with gr.Column():
                    gr.Markdown("#### ⬆️ Upload Actions")
                    
                    detect_button = gr.Button("🔍 Detect Zip File", variant="secondary")
                    detect_status = gr.Textbox(label="Detection Result", value="", max_lines=5)
                    
                    push_button = gr.Button("🚀 Push to HuggingFace", variant="primary", size="lg")
                    push_output = gr.Textbox(label="Upload Status", value="", max_lines=15)
            
            gr.Markdown("---")
            gr.Markdown("### 📖 How to Get API Key")
            gr.Markdown("""
            1. Go to [HuggingFace Settings/Tokens](https://huggingface.co/settings/tokens)
            2. Click "New Token"
            3. Select **Write** permission
            4. Copy the token (starts with `hf_`)
            5. Paste it above
            """)
            
            # Event Handlers for HuggingFace Tab
            
            # Auto-fill model name when changed
            hf_model_name.change(
                fn=lambda name: f"Searching for: {name}.zip..." if name else "Enter model name",
                inputs=[hf_model_name],
                outputs=[auto_repo_info]
            )
            
            # Authentication button
            auth_button.click(
                fn=hf_uploader.authenticate,
                inputs=[hf_api_key],
                outputs=[auth_status]
            )
            
            # Detect zip file
            def detect_zip_file(model_name: str) -> str:
                """Detect zip file for the given model."""
                if not model_name or not model_name.strip():
                    return "❌ Please enter a model name"
                
                zip_path, msg = hf_uploader.find_model_zip(model_name.strip())
                if zip_path:
                    return f"✅ {msg}\n📍 Path: {zip_path}"
                return f"❌ {msg}"
            
            detect_button.click(
                fn=detect_zip_file,
                inputs=[hf_model_name],
                outputs=[detect_status]
            )
            
            # Main push function
            def push_to_hf(
                model_name: str,
                repo_id: str,
                api_key: str,
                sample_rate: str,
                version: str,
                private: bool
            ) -> str:
                """
                Push model to HuggingFace with validation.
                """
                # Validate model name
                if not model_name or not model_name.strip():
                    return "❌ Please enter a model name"
                
                # Validate repo ID
                valid, error_msg = HuggingFaceUploader.validate_repo_id(repo_id)
                if not valid:
                    return f"❌ Invalid Repository ID: {error_msg}"
                
                # Validate API key
                if not api_key or not api_key.strip():
                    return "❌ Please enter your HuggingFace API Key"
                
                # Perform upload
                return hf_uploader.upload_model(
                    model_name=model_name.strip(),
                    repo_id=repo_id.strip(),
                    api_key=api_key,
                    sample_rate=sample_rate,
                    version=version,
                    private=private
                )
            
            push_button.click(
                fn=push_to_hf,
                inputs=[
                    hf_model_name,
                    hf_repo_id,
                    hf_api_key,
                    hf_sample_rate,
                    hf_version,
                    hf_private
                ],
                outputs=[push_output]
            )

        # ===================== SETTINGS TAB =====================
        with gr.TabItem("Settings"):
            gr.Markdown("### ⚙️ Training Settings")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### Data Processing")
                    normalize = gr.Checkbox(label="Normalize Audio", value=True)
                    create_index = gr.Checkbox(label="Create Index File", value=True)
                    percentage = gr.Slider(
                        minimum=1.0,
                        maximum=5.0,
                        step=0.5,
                        label="Fragment Length (seconds)",
                        value=3.0,
                        interactive=True
                    )

                with gr.Column():
                    gr.Markdown("#### Training Parameters")
                    optimizer_settings = gr.Dropdown(
                        label="Optimizer",
                        choices=["AdamW", "AdaBelief"],
                        value="AdamW",
                        interactive=True
                    )
                    vocoder_settings = gr.Dropdown(
                        label="Vocoder",
                        choices=["HiFi-GAN"],
                        value="HiFi-GAN",
                        interactive=True
                    )
                    save_half_settings = gr.Checkbox(label="Save with Half Precision", value=True)
                    save_to_zip_settings = gr.Checkbox(label="Package Model in ZIP", value=True)

            gr.Markdown("---")
            gr.Markdown("### 📖 Documentation")
            gr.Markdown("""
            - **Model Name**: Use only letters, numbers, and underscores
            - **Dataset**: Minimum 10 minutes of clean audio (30+ recommended)
            - **Sampling Rate**: 40k for higher quality, 32k for faster training
            - **Epochs**: 100-200 minimum, 200-500 optimal
            - **Batch Size**: Start with 8, adjust based on VRAM
            """)

    # ===================== LAUNCH =====================
    if config.iscolab:
        app.queue(max_size=20).launch(
            share=True,
            show_error=True,
            debug=True
        )
    else:
        app.queue(max_size=1022).launch(
            server_name="0.0.0.0",
            inbrowser=not config.noautoopen,
            server_port=config.listen_port,
            quiet=True,
            debug=True
        )
