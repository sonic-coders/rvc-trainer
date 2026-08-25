# GUI BY BF667

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
from typing import Optional, Tuple, List, Dict, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import argparse

from configs.config import (
    # Main configuration
    Config,
    config,
    args,
    iscolab,
    BASE_ROOT,
    
    # Classes
    GPUManager,
    PretrainedModelFinder,
    
    # Utility functions
    setup_logging,
    validate_model_name,
    validate_dataset_folder,
    get_sample_rate,
    filter_training_output,
    
    # Constants
    UNWANTED_LOG_PATTERNS,
)

# Initialize logging using config module's function
logger = setup_logging()

from huggingface_hub import HfApi, create_repo, RepoCard, upload_folder, login, whoami
    logger.info("Importing RVC training modules directly...")

from rvc.train.preprocess.preprocess import PreProcess, preprocess_trainset  
from rvc.train.preprocess.preparing_data import DataPreprocessor, generate_filelist
import faiss
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from multiprocessing import cpu_count
from rvc.train.train import main as train_main, get_hparams, run as train_run

logger.info("✅ All RVC modules imported successfully!")

gpu_manager = GPUManager()
gpus = gpu_manager.gpus
gpu_info = gpu_manager.gpu_info
F0GPUVisible = True
default_batch_size = config.default_batch_size
pretrained_finder = PretrainedModelFinder()

def preprocess_dataset_direct(
    dataset_folder: str, 
    training_name: str, 
    sr2: str, 
    num_processes: int
) -> str:
    """
    Preprocess dataset by calling PreProcess class directly.
    
    This replaces the subprocess call to preprocess.py with a direct
    function call to the PreProcess class.
    
    Args:
        dataset_folder: Path to the dataset folder
        training_name: Name for the training model
        sr2: Target sample rate ('40k' or '32k')
        num_processes: Number of CPU processes for parallel processing
        
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
        sample_rate = int(get_sample_rate(sr2, "v2"))
        percentage = config.default_percentage
        normalize = config.default_normalize
        
        logger.info(f"Starting direct preprocessing...")
        logger.info(f"  Dataset: {dataset_folder}")
        logger.info(f"  Sample Rate: {sample_rate}Hz")
        logger.info(f"  Processes: {num_processes}")
        
        # Call the preprocessing function DIRECTLY (no subprocess!)
        try:
            preprocess_trainset(
                input_root=dataset_folder,
                sample_rate=sample_rate,
                num_processes=num_processes,
                exp_dir=model_dir,
                percentage=percentage,
                normalize=normalize
            )
        except FileNotFoundError as e:
            return f"❌ Error during preprocessing: {str(e)}"
        except Exception as e:
            logger.exception("Exception during preprocessing")
            return f"❌ Error during preprocessing: {str(e)}"
        
        logger.info(f"✅ Preprocessing completed for model: {training_name}")
        return "✅ Data preprocessing completed successfully!"
        
    except Exception as e:
        logger.exception(f"Unexpected exception during preprocessing: {e}")
        return f"❌ Error during preprocessing: {str(e)}"


def extract_features_direct(
    training_name: str,
    version19: str,
    f0_method: str,
    include_mutes: int = 2
) -> str:
    """
    Extract F0 and features by calling DataPreprocessor directly.
    
    This replaces the subprocess call to preparing_data.py with a direct
    instantiation and method call on DataPreprocessor.
    
    Args:
        training_name: Name of the training model
        version19: Model version ('v1' or 'v2')
        f0_method: F0 extraction method ('rmvpe', 'crepe', 'hpa-rmvpe')
        include_mutes: Number of mute files to include
        
    Returns:
        Status message indicating success or failure
    """
    try:
        model_dir = f'{config.save_dir}/{training_name}'
        
        # Validate model directory exists
        if not os.path.exists(model_dir):
            return f"❌ Error: Model directory '{model_dir}' does not exist. Run preprocessing first!"
        
        # Convert parameters
        sample_rate = int(get_sample_rate("48k" if version19 == "v2" else "32k", version19))
        arch_fairseq = "Fairseq"
        
        # Normalize f0 method name
        f0_method_clean = f0_method.replace("_gpu", "").replace("+", "+")
        
        logger.info(f"Starting direct feature extraction...")
        logger.info(f"  Model: {training_name}")
        logger.info(f"  F0 Method: {f0_method_clean}")
        logger.info(f"  Sample Rate: {sample_rate}Hz")
        
        # Create DataPreprocessor instance and process files DIRECTLY
        try:
            preprocessor = DataPreprocessor()
            preprocessor.process_files()
            
            # Generate filelist after feature extraction
            generate_filelist(model_dir, sample_rate, include_mutes)
            
        except FileNotFoundError as e:
            return f"❌ Error during feature extraction: {str(e)}"
        except Exception as e:
            logger.exception("Exception during feature extraction")
            return f"❌ Error during feature extraction: {str(e)}"
        
        logger.info(f"✅ Feature extraction completed for model: {training_name}")
        return "✅ Feature extraction completed successfully!"
        
    except Exception as e:
        logger.exception(f"Unexpected exception during feature extraction: {e}")
        return f"❌ Error during feature extraction: {str(e)}"


def train_index_direct(
    training_name: str,
    version19: str,
    index_algorithm: str = "Faiss"
) -> str:
    """
    Train index file by calling FAISS directly.
    
    This replaces the subprocess call to extract_index.py with
    direct Python code execution.
    
    Args:
        training_name: Name of the training model
        version19: Model version (for logging)
        index_algorithm: Index algorithm to use ('Faiss', 'Auto', 'KMeans')
        
    Returns:
        Status message indicating success or failure
    """
    try:
        model_dir = f'{config.save_dir}/{training_name}'
        
        # Validate model directory exists
        if not os.path.exists(model_dir):
            return f"❌ Error: Model directory '{model_dir}' does not exist."
        
        # Check if features directory exists
        feature_dir = os.path.join(model_dir, "data", "features")
        if not os.path.exists(feature_dir):
            return f"❌ Error: Features directory not found. Run feature extraction first!"
        
        model_name = os.path.basename(model_dir)
        index_filename = f"{model_name}.index"
        index_filepath = os.path.join(model_dir, index_filename)
        
        logger.info(f"Starting direct index generation...")
        logger.info(f"  Model: {training_name}")
        logger.info(f"  Algorithm: {index_algorithm}")
        
        # Check if index already exists
        if os.path.exists(index_filepath):
            logger.info(f"Index already exists: {index_filepath}")
            return f"✅ Index already exists: {index_filename}"
        
        # Load all feature numpy arrays
        logger.info("Loading feature files...")
        npys = []
        listdir_res = sorted(os.listdir(feature_dir))
        
        for name in listdir_res:
            file_path = os.path.join(feature_dir, name)
            phone = np.load(file_path)
            npys.append(phone)
        
        if not npys:
            return f"❌ Error: No feature files found in {feature_dir}"
        
        # Concatenate all features
        logger.info(f"Concatenating {len(npys)} feature files...")
        big_npy = np.concatenate(npys, axis=0)
        
        # Shuffle for better indexing
        big_npy_idx = np.arange(big_npy.shape[0])
        np.random.shuffle(big_npy_idx)
        big_npy = big_npy[big_npy_idx]
        
        logger.info(f"Total features shape: {big_npy.shape}")
        
        # Apply KMeans clustering if too many features
        if big_npy.shape[0] > 2e5 and index_algorithm in ("Auto", "KMeans"):
            logger.info("Applying KMeans clustering for large dataset...")
            big_npy = (
                MiniBatchKMeans(
                    n_clusters=10000,
                    verbose=True,
                    batch_size=256 * cpu_count(),
                    compute_labels=False,
                    init="random",
                )
                .fit(big_npy)
                .cluster_centers_
            )
        
        # Create FAISS index
        logger.info("Building FAISS index...")
        n_ivf = min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39)
        
        index_added = faiss.index_factory(768, f"IVF{n_ivf},Flat")
        index_ivf_added = faiss.extract_index_ivf(index_added)
        index_ivf_added.nprobe = 1
        index_added.train(big_npy)
        
        # Add vectors to index in batches
        batch_size_add = 8192
        for i in range(0, big_npy.shape[0], batch_size_add):
            index_added.add(big_npy[i : i + batch_size_add])
        
        # Save index
        faiss.write_index(index_added, index_filepath)
        
        logger.info(f"✅ Index saved: {index_filepath}")
        return f"✅ Index training completed successfully!\n📁 Saved: {index_filename}"
        
    except Exception as e:
        logger.exception(f"Unexpected exception during index training: {e}")
        return f"❌ Error during index training: {str(e)}"


def train_model_direct(
    training_name: str,
    sr2: str,
    total_epoch11: int,
    batch_size12: int,
    save_epoch10: int,
    version19: str,
    optimizer: str = "AdamW",
    vocoder: str = "HiFi-GAN",
    pretrained_G14: Optional[str] = None,
    pretrained_D15: Optional[str] = None,
    gpus16: str = "0",
    save_half: bool = True,
    save_to_zip: bool = True,
) -> str:
    """
    Train model by calling train.py functions directly.
    
    This replaces the subprocess call to train.py with direct
    function calls to main() and related functions.
    
    Args:
        training_name: Name for the model
        sr2: Target sample rate
        total_epoch11: Total number of epochs to train
        batch_size12: Training batch size
        save_epoch10: Epoch interval for saving checkpoints
        version19: Model version
        optimizer: Optimizer to use
        vocoder: Vocoder type
        pretrained_G14: Path to pretrained generator (optional)
        pretrained_D15: Path to pretrained discriminator (optional)
        gpus16: GPU identifiers to use
        save_half: Whether to save in half precision
        save_to_zip: Whether to package as ZIP
        
    Returns:
        Status message with training results or error information
    """
    import torch.multiprocessing as mp
    
    try:
        # Validate model name
        valid, error_msg = validate_model_name(training_name)
        if not valid:
            return f"❌ Error: {error_msg}"
        
        # Convert parameters
        sample_rate = int(get_sample_rate(sr2, version19))
        
        # Validate numeric parameters
        try:
            total_epochs = int(total_epoch11)
            batch_size = int(batch_size12)
            save_epoch_interval = int(save_epoch10)
        except ValueError as e:
            return f"❌ Error: Invalid numeric parameter: {e}"
        
        # Validate ranges
        if total_epochs < 1:
            return "❌ Error: Total epochs must be at least 1"
        if batch_size < 1:
            return "❌ Error: Batch size must be at least 1"
        if save_epoch_interval < 1:
            return "❌ Error: Save epoch interval must be at least 1"
        
        # Log configuration
        logger.info("=" * 50)
        logger.info("Starting Direct Training Session")
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
        
        # Set up sys.argv for get_hparams() compatibility
        original_argv = sys.argv.copy()
        sys.argv = [
            'train.py',
            '--experiment_dir', config.save_dir,
            '--model_name', training_name,
            '--total_epoch', str(total_epochs),
            '--save_every_epoch', str(save_epoch_interval),
            '--batch_size', str(batch_size),
            '--sample_rate', str(sample_rate),
            '--vocoder', vocoder,
            '--optimizer', optimizer,
            '--gpus', gpus16,
            '--save_to_zip', str(save_to_zip).lower(),
            '--save_half', str(save_half).lower(),
        ]
        
        # Add pretrained paths if provided
        if pretrained_G14:
            sys.argv.extend(['--pretrain_g', pretrained_G14])
        if pretrained_D15:
            sys.argv.extend(['--pretrain_d', pretrained_D15])
        
        # Run training DIRECTLY (no subprocess!)
        logger.info("Launching training process directly...")
        
        try:
            # Set multiprocessing start method
            mp.set_start_method("spawn", force=True)
            
            # Call main() from train.py directly
            train_main()
            
            success_msg = (
                f"✅ Training completed successfully!\n\n"
                f"📦 Model: {training_name}\n"
                f"⏱️ Epochs: {total_epochs}\n"
                f"📊 Batch Size: {batch_size}\n"
                f"🎵 Sample Rate: {sample_rate}Hz\n"
                f"🔧 Optimizer: {optimizer}\n"
                f"💾 Checkpoints saved every: {save_epoch_interval} epochs"
            )
            
            logger.info(success_msg)
            return success_msg
            
        except SystemExit as e:
            # train_main() calls sys.exit(0) on success
            if e.code == 0 or e.code is None:
                success_msg = f"✅ Training completed for model: {training_name}"
                logger.info(success_msg)
                return success_msg
            else:
                error_msg = f"❌ Training exited with code: {e.code}"
                logger.error(error_msg)
                return error_msg
                
        except Exception as e:
            logger.exception(f"Error during direct training: {e}")
            
            # Save error log
            try:
                error_log_path = f"{config.save_dir}/{training_name}/error_log.txt"
                os.makedirs(os.path.dirname(error_log_path), exist_ok=True)
                with open(error_log_path, "w") as f:
                    f.write(f"Training failed:\n\n{traceback.format_exc()}")
            except Exception:
                pass
            
            return f"❌ Error during training: {str(e)}"
            
        finally:
            # Restore original argv
            sys.argv = original_argv
            
    except Exception as e:
        logger.exception(f"Unexpected exception in train_model_direct: {e}")
        return f"❌ Error during training: {str(e)}"


# ========================================================================== #
# UI CALLBACK FUNCTIONS (Updated to use direct calls)
# ========================================================================== #

def change_f0_method(f0_method: str) -> str:
    """Update GPU visibility based on F0 method selection."""
    if f0_method == "rmvpe_gpu":
        return gpus
    return "0"


def change_sr2(sr: str, if_f0: str, version: str) -> Tuple[gr.update, gr.update]:
    """Update pretrained model choices when sample rate changes."""
    sr2_val = "40k" if version == "v1" else sr
    
    g_choices = pretrained_finder.get_generator_choices(sr2_val)
    d_choices = pretrained_finder.get_discriminator_choices(sr2_val)
    
    return (
        gr.update(choices=g_choices, value=g_choices[0] if g_choices else ''),
        gr.update(choices=d_choices, value=d_choices[0] if d_choices else '')
    )


def change_version19(sr: str, if_f0: str, version: str) -> Tuple[gr.update, gr.update, gr.update]:
    """Handle version change and update related options."""
    if version == "v1":
        sr2_val = "40k"
        sr_update = gr.update(value="40k", visible=False)
    else:
        sr2_val = sr
        sr_update = gr.update(visible=True)
    
    g_choices = pretrained_finder.get_generator_choices(sr2_val)
    d_choices = pretrained_finder.get_discriminator_choices(sr2_val)
    
    return (
        gr.update(choices=g_choices, value=g_choices[0] if g_choices else ''),
        gr.update(choices=d_choices, value=d_choices[0] if d_choices else ''),
        sr_update
    )


def change_f0(if_f0: bool, sr: str, version: str) -> Tuple[gr.update, gr.update, gr.update]:
    """Update F0 method choices based on singing mode."""
    if if_f0:
        f0_choices = config.f0_method_choices_singing
        f0_value = config.f0_default_singing
    else:
        f0_choices = config.f0_method_choices_non_singing
        f0_value = config.f0_default_non_singing
    
    sr_val = "40k" if version == "v1" else sr
    g_choices = pretrained_finder.get_generator_choices(sr_val)
    d_choices = pretrained_finder.get_discriminator_choices(sr_val)
    
    return (
        gr.update(choices=f0_choices, value=f0_value),
        gr.update(choices=g_choices, value=g_choices[0] if g_choices else ''),
        gr.update(choices=d_choices, value=d_choices[0] if d_choices else '')
    )


# ========================================================================== #
# STEP FUNCTIONS (Now using direct imports!)
# ========================================================================== #

def preprocess_dataset(dataset_folder, training_name, sr2, np7):
    """UI callback for Step 1: Process Data (now direct call)."""
    return preprocess_dataset_direct(
        dataset_folder=dataset_folder,
        training_name=training_name,
        sr2=sr2,
        num_processes=int(np7)
    )


def extract_f0_feature(gpus6, np7, f0method8, if_f0_3, training_name, version19, gpus_rmvpe):
    """UI callback for Step 2: Extract Features (now direct call)."""
    # Set CUDA device for feature extraction
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus_rmvpe if f0method8 == "rmvpe_gpu" else "0"
    
    return extract_features_direct(
        training_name=training_name,
        version19=version19,
        f0_method=f0method8.replace("_gpu", ""),
        include_mutes=2
    )


def train_index(training_name, version19):
    """UI callback for Step 3: Train Index (now direct call)."""
    return train_index_direct(
        training_name=training_name,
        version19=version19,
        index_algorithm="Faiss"
    )


def click_train(
    training_name,
    sr2,
    if_f0_3,
    spk_id5,
    save_epoch10,
    total_epoch11,
    batch_size12,
    if_save_latest13,
    pretrained_G14,
    pretrained_D15,
    gpus16,
    if_cache_gpu17,
    if_save_every_weights18,
    version19,
    # Additional parameters from Settings tab
    optimizer="AdamW",
    vocoder="HiFi-GAN",
    save_half=True,
    save_to_zip=True,
):
    """UI callback for Step 4: Train Model (now direct call)."""
    return train_model_direct(
        training_name=training_name,
        sr2=sr2,
        total_epoch11=total_epoch11,
        batch_size12=batch_size12,
        save_epoch10=save_epoch10,
        version19=version19,
        optimizer=optimizer,
        vocoder=vocoder,
        pretrained_G14=pretrained_G14,
        pretrained_D15=pretrained_D15,
        gpus16=gpus16,
        save_half=save_half,
        save_to_zip=save_to_zip,
    )


def train1key(
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
    if_save_latest13,
    pretrained_G14,
    pretrained_D15,
    gpus16,
    if_cache_gpu17,
    if_save_every_weights18,
    version19,
    gpus_rmvpe,
):
    """One-click training pipeline with direct imports."""
    logger.info(f"Starting one-click training pipeline for: {training_name}")
    
    results = []
    
    # Step 1: Preprocess (direct)
    logger.info("Step 1/4: Preprocessing dataset...")
    result1 = preprocess_dataset(dataset_folder, training_name, sr2, np7)
    results.append(result1)
    if "❌" in result1:
        logger.error(f"Pipeline stopped at preprocessing: {result1}")
        return result1
    
    # Step 2: Extract features (direct)
    logger.info("Step 2/4: Extracting features...")
    result2 = extract_f0_feature(gpus16, np7, f0method8, if_f0_3, training_name, version19, gpus_rmvpe)
    results.append(result2)
    if "❌" in result2:
        logger.error(f"Pipeline stopped at feature extraction: {result2}")
        return f"{result1}\n\n{result2}"
    
    # Step 3: Train index (direct)
    logger.info("Step 3/4: Training index...")
    result3 = train_index(training_name, version19)
    results.append(result3)
    if "❌" in result3:
        logger.error(f"Pipeline stopped at index training: {result3}")
        return f"{result1}\n\n{result2}\n\n{result3}"
    
    # Step 4: Train model (direct)
    logger.info("Step 4/4: Training model...")
    result4 = click_train(
        training_name, sr2, if_f0_3, spk_id5, save_epoch10, total_epoch11,
        batch_size12, if_save_latest13, pretrained_G14, pretrained_D15,
        gpus16, if_cache_gpu17, if_save_every_weights18, version19,
    )
    results.append(result4)
    
    final_message = "✅ All steps completed!\n\n" + "\n\n".join(results)
    logger.info("One-click training pipeline completed!")
    
    return final_message


def download_model_files(training_name):
    """Find and list downloadable model files."""
    try:
        weights_path = f'{config.weights_dir}/{training_name}'
        logs_path = f'{config.logs_dir}/{training_name}'
        
        files = []
        
        if os.path.exists(weights_path):
            try:
                files.extend([
                    os.path.join(weights_path, f) 
                    for f in os.listdir(weights_path) 
                    if f.endswith('.pth')
                ])
            except PermissionError:
                logger.warning(f"Permission denied reading: {weights_path}")
        
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


def handle_upload(files, folder):
    """Handle file upload to dataset folder."""
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
        """Authenticate with HuggingFace using API key."""
        if not HF_AVAILABLE:
            return False, "huggingface_hub library is not installed. Run: pip install huggingface_hub"
        
        if not api_key or not api_key.strip():
            return False, "API Key cannot be empty"
        
        try:
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
        """Find the zip file for a trained model."""
        search_paths = [
            f'{config.weights_dir}/{model_name}',
            f'{config.logs_dir}/{model_name}',
            f'{config.save_dir}/{model_name}',
        ]
        
        for search_path in search_paths:
            if os.path.exists(search_path):
                zip_files = glob.glob(f'{search_path}/*.zip')
                if zip_files:
                    latest_zip = max(zip_files, key=os.path.getmtime)
                    msg = f"Found zip: {os.path.basename(latest_zip)}"
                    logger.info(msg)
                    return latest_zip, msg
        
        for search_path in search_paths:
            if os.path.exists(search_path) and os.listdir(search_path):
                msg = f"Will create zip from: {search_path}"
                logger.info(msg)
                return search_path, msg
        
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
        """Generate README content for the model repository."""
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
        """Ensure repository exists, create if it doesn't."""
        if not self.api:
            return False, "Not authenticated. Please enter API key first."
        
        try:
            try:
                repo_info = self.api.repo_info(repo_id=repo_id, token=token)
                logger.info(f"Repository exists: {repo_id}")
                return True, f"✅ Repository already exists: {repo_id}"
            except Exception:
                pass
            
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
        """Upload model to HuggingFace."""
        success, msg = self.authenticate(api_key)
        if not success:
            return msg
        
        zip_path, msg = self.find_model_zip(model_name)
        if zip_path is None:
            return f"❌ {msg}"
        
        logger.info(f"Model path: {zip_path}")
        
        success, msg = self.ensure_repo_exists(repo_id, private, api_key.strip())
        if not success:
            return msg
        
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
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
                    shutil.copy2(zip_path, temp_dir)
                    upload_dir = temp_dir
                    files_in_upload = [os.path.basename(zip_path)]
                
                readme_content = self.create_readme_content(
                    model_name=model_name,
                    sample_rate=sample_rate,
                    version=version,
                    files_list=files_in_upload
                )
                readme_path = os.path.join(temp_dir, "README.md")
                with open(readme_path, 'w', encoding='utf-8') as f:
                    f.write(readme_content)
                
                logger.info(f"Uploading to {repo_id}...")
                
                api = HfApi(token=api_key.strip())
                
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
        """Validate HuggingFace repository ID format."""
        if not repo_id or not repo_id.strip():
            return False, "Repository ID cannot be empty"
        
        repo_id = repo_id.strip()
        
        if repo_id.count('/') != 1:
            return False, 'Repository ID must be in format: "username/repo-name"'
        
        username, repo_name = repo_id.split('/')
        
        if not username or not re.match(r'^[a-zA-Z0-9_-]+$', username):
            return False, "Invalid username format. Use only letters, numbers, hyphens, underscores."
        
        if not repo_name or not re.match(r'^[a-zA-Z0-9._-]+$', repo_name):
            return False, "Invalid repository name format. Use only letters, numbers, dots, hyphens, underscores."
        
        if len(repo_id) > 200:
            return False, "Repository ID too long (max 200 characters)"
        
        return True, ""


# Global instance
hf_uploader = HuggingFaceUploader()


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
                        choices=config.sample_rate_choices,
                        value=config.default_sample_rate,
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
                        choices=config.version_choices,
                        value=config.default_version,
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
                        value=config.default_epochs,
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
                            value=config.default_save_epoch,
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
                    
                    auto_repo_info = gr.Textbox(
                        label="Auto-detected Info",
                        value="Enter model name to detect zip path",
                        interactive=False
                    )
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### 🎯 Model Selection")
                    
                    hf_model_name = gr.Textbox(
                        label="Model Name (for zip detection)",
                        placeholder="My-Voice",
                        value="",
                        interactive=True
                    )
                    
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
            
            hf_model_name.change(
                fn=lambda name: f"Searching for: {name}.zip..." if name else "Enter model name",
                inputs=[hf_model_name],
                outputs=[auto_repo_info]
            )
            
            auth_button.click(
                fn=hf_uploader.authenticate,
                inputs=[hf_api_key],
                outputs=[auth_status]
            )
            
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
            
            def push_to_hf(
                model_name: str,
                repo_id: str,
                api_key: str,
                sample_rate: str,
                version: str,
                private: bool
            ) -> str:
                """Push model to HuggingFace with validation."""
                if not model_name or not model_name.strip():
                    return "❌ Please enter a model name"
                
                valid, error_msg = HuggingFaceUploader.validate_repo_id(repo_id)
                if not valid:
                    return f"❌ Invalid Repository ID: {error_msg}"
                
                if not api_key or not api_key.strip():
                    return "❌ Please enter your HuggingFace API Key"
                
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
                    normalize = gr.Checkbox(label="Normalize Audio", value=config.default_normalize)
                    create_index = gr.Checkbox(label="Create Index File", value=config.default_create_index)
                    percentage = gr.Slider(
                        minimum=1.0,
                        maximum=5.0,
                        step=0.5,
                        label="Fragment Length (seconds)",
                        value=config.default_percentage,
                        interactive=True
                    )

                with gr.Column():
                    gr.Markdown("#### Training Parameters")
                    optimizer_settings = gr.Dropdown(
                        label="Optimizer",
                        choices=["AdamW", "AdaBelief"],
                        value=config.default_optimizer,
                        interactive=True
                    )
                    vocoder_settings = gr.Dropdown(
                        label="Vocoder",
                        choices=["HiFi-GAN"],
                        value=config.default_vocoder,
                        interactive=True
                    )
                    save_half_settings = gr.Checkbox(label="Save with Half Precision", value=config.default_save_half)
                    save_to_zip_settings = gr.Checkbox(label="Package Model in ZIP", value=config.default_save_to_zip)

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
