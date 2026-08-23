# ========================================================================== #
# CONFIGURATION MODULE
# All configuration classes and constants for RVC Trainer
# ========================================================================== #

import os
import sys
import logging
import argparse
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

# ========================================================================== #
# ARGUMENT PARSING
# ========================================================================== #

parser = argparse.ArgumentParser()
parser.add_argument("--colab", action="store_true", help="Launch in colab")
args = parser.parse_args()

# ========================================================================== #
# BASE PATHS
# ========================================================================== #

BASE_ROOT = os.getcwd()
sys.path.append(BASE_ROOT)

# ========================================================================== #
# MAIN CONFIGURATION CLASS
# ========================================================================== #

@dataclass
class Config:
    """
    Main application configuration settings.
    
    This class centralizes all configuration parameters for the RVC Trainer
    application, making it easy to manage and modify settings.
    
    Attributes:
        n_cpu: Number of CPU cores available for processing
        iscolab: Whether running in Google Colab environment
        noautoopen: Whether to auto-open browser on launch
        listen_port: Port number for Gradio server
        root_dir: Root directory of the application
        save_dir: Directory for saving trained models
        default_batch_size: Default batch size for training
        default_epochs: Default number of training epochs
        default_save_epoch: Default epoch interval for saving checkpoints
        default_sample_rate: Default sample rate setting
        default_f0_method: Default F0 extraction method
        default_version: Default RVC version
        pretrained_dir: Directory containing pretrained models
        weights_dir: Directory for output model weights
        logs_dir: Directory for training logs
    """
    # System Settings
    n_cpu: int = os.cpu_count() or 4
    iscolab: bool = args.colab
    noautoopen: bool = False
    listen_port: int = 7860
    
    # Path Configuration
    root_dir: str = f"{BASE_ROOT}/rvc-trainer"
    save_dir: str = f"{BASE_ROOT}/rvc-trainer/drive/MyDrive/rvc-trainer"
    
    # Default Training Parameters
    default_batch_size: int = 8
    default_epochs: int = 150
    default_save_epoch: int = 25
    default_sample_rate: str = "40k"
    default_f0_method: str = "rmvpe"
    default_version: str = "v2"
    
    # Directory Paths
    pretrained_dir: str = "assets/pretrained_v2"
    weights_dir: str = "assets/weights"
    logs_dir: str = "logs"
    
    # Training Defaults
    default_optimizer: str = "AdamW"
    default_vocoder: str = "HiFi-GAN"
    default_save_half: bool = True
    default_save_to_zip: bool = True
    
    # GPU Settings
    default_gpus: str = "0"
    
    # Data Processing Defaults
    default_normalize: bool = True
    default_create_index: bool = True
    default_percentage: float = 3.0
    
    # F0 Method Options
    f0_method_choices_singing: List[str] = field(default_factory=lambda: ["rmvpe", "hpa-rmvpe"])
    f0_method_choices_non_singing: List[str] = field(default_factory=lambda: ["crepe", "rmvpe", "hpa-rmvpe"])
    f0_default_singing: str = "rmvpe_gpu"
    f0_default_non_singing: str = "crepe"
    
    # Sample Rate Options
    sample_rate_choices: List[str] = field(default_factory=lambda: ["40k", "32k"])
    version_choices: List[str] = field(default_factory=lambda: ["v1", "v2"])
    
    # Validation Patterns
    valid_model_name_pattern: str = r"^[a-zA-Z0-9_\-]+$"
    max_model_name_length: int = 50
    valid_gpu_pattern: str = r"^[\d,\-]+$"
    
    # Logging
    log_file: str = "rvc_trainer.log"
    log_level_console: int = logging.INFO
    log_level_file: int = logging.ERROR


# Global configuration instance
config = Config()


# ========================================================================== #
# GPU MANAGER CLASS
# ========================================================================== #

class GPUManager:
    """
    Manages GPU detection and configuration.
    
    This class handles automatic detection of available GPUs,
    validation of GPU identifier strings, and provides
    configuration information for GPU-dependent operations.
    
    Attributes:
        _gpus: Comma-separated string of available GPU IDs
        _gpu_info: Human-readable GPU information string
        _cuda_available: Whether CUDA is available
        _device_count: Number of detected GPU devices
    """
    
    def __init__(self):
        """Initialize GPU manager and detect available GPUs."""
        self._gpus: str = "0"
        self._gpu_info: str = "No GPU detected"
        self._cuda_available: bool = False
        self._device_count: int = 0
        self._detect_gpus()
    
    def _detect_gpus(self) -> None:
        """
        Detect available CUDA GPUs.
        
        Attempts to import PyTorch and detect CUDA-capable devices.
        Falls back gracefully if PyTorch or CUDA is not available.
        """
        try:
            import torch
            self._cuda_available = torch.cuda.is_available()
            if self._cuda_available:
                self._device_count = torch.cuda.device_count()
                self._gpus = ",".join(str(i) for i in range(self._device_count))
                self._gpu_info = f"Available GPUs: {self._device_count}"
                
                # Setup logger if available
                try:
                    logger = logging.getLogger("rvc_trainer")
                    logger.info(f"GPU detection successful: {self._gpu_info}")
                except Exception:
                    pass
            else:
                try:
                    logger = logging.getLogger("rvc_trainer")
                    logger.warning("CUDA is not available, running on CPU")
                except Exception:
                    pass
        except ImportError:
            try:
                logger = logging.getLogger("rvc_trainer")
                logger.warning("PyTorch not installed, GPU features disabled")
            except Exception:
                pass
        except Exception as e:
            try:
                logger = logging.getLogger("rvc_trainer")
                logger.error(f"Error during GPU detection: {e}")
            except Exception:
                pass
    
    @property
    def gpus(self) -> str:
        """Get comma-separated GPU ID string."""
        return self._gpus
    
    @property
    def gpu_info(self) -> str:
        """Get human-readable GPU information."""
        return self._gpu_info
    
    @property
    def cuda_available(self) -> bool:
        """Check if CUDA is available."""
        return self._cuda_available
    
    @property
    def device_count(self) -> int:
        """Get number of detected GPU devices."""
        return self._device_count
    
    def validate_gpu_string(self, gpu_str: str) -> Tuple[bool, str]:
        """
        Validate GPU identifier string format.
        
        Args:
            gpu_str: GPU identifier string (e.g., "0", "0-1-2")
            
        Returns:
            Tuple of (is_valid, error_message)
            
        Examples:
            >>> manager.validate_gpu_string("0")
            (True, "")
            >>> manager.validate_gpu_string("0-1-2")
            (True, "")
            >>> manager.validate_gpu_string("")
            (False, "GPU string cannot be empty")
        """
        if not gpu_str or not gpu_str.strip():
            return False, "GPU string cannot be empty"
        
        if not re.match(config.valid_gpu_pattern, gpu_str.strip()):
            return False, (
                f"Invalid GPU format: {gpu_str}. "
                "Use format like '0' or '0-1-2'"
            )
        
        return True, ""


# Import re for GPU validation pattern matching
import re


# ========================================================================== #
# PRETRAINED MODEL FINDER CLASS
# ========================================================================== #

class PretrainedModelFinder:
    """
    Handles finding and caching pretrained model files.
    
    This class manages the discovery of pretrained generator and
    discriminator models from the pretrained directory, with optional
    caching for improved performance.
    
    Attributes:
        pretrained_dir: Directory containing pretrained model files
        _cache: Dictionary cache of found model files
    """
    
    def __init__(self, pretrained_dir: Optional[str] = None):
        """
        Initialize the pretrained model finder.
        
        Args:
            pretrained_dir: Path to directory containing pretrained models.
                           If None, uses the default from Config.
        """
        self.pretrained_dir = pretrained_dir or config.pretrained_dir
        self._cache: dict = {}
    
    def _get_pretrained_files(
        self, 
        sr_val: str, 
        letter: str,
        use_cache: bool = True
    ) -> List[str]:
        """
        Get list of pretrained model files matching criteria.
        
        Searches the pretrained directory for .pth files that match
        the specified sample rate value and model type letter.
        
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
                try:
                    logger = logging.getLogger("rvc_trainer")
                    logger.warning(f"Pretrained directory not found: {self.pretrained_dir}")
                except Exception:
                    pass
                return []
            
            files = [
                os.path.abspath(os.path.join(self.pretrained_dir, file))
                for file in os.listdir(self.pretrained_dir)
                if file.endswith('.pth') and sr_val in file and letter in file
            ]
            
            self._cache[cache_key] = files
            return files
            
        except OSError as e:
            try:
                logger = logging.getLogger("rvc_trainer")
                logger.error(f"Error reading pretrained directory: {e}")
            except Exception:
                pass
            return []
    
    def get_generator_choices(self, sr_val: str) -> List[str]:
        """
        Get available generator model choices for a sample rate.
        
        Args:
            sr_val: Sample rate value to filter by
            
        Returns:
            List of paths to generator (.pth) files
        """
        return self._get_pretrained_files(sr_val, 'G')
    
    def get_discriminator_choices(self, sr_val: str) -> List[str]:
        """
        Get available discriminator model choices for a sample rate.
        
        Args:
            sr_val: Sample rate value to filter by
            
        Returns:
            List of paths to discriminator (.pth) files
        """
        return self._get_pretrained_files(sr_val, 'D')
    
    def clear_cache(self) -> None:
        """Clear the file cache to force refresh on next query."""
        self._cache.clear()


# ========================================================================== #
# LOGGING CONFIGURATION
# ========================================================================== #

def setup_logging(log_file: Optional[str] = None) -> logging.Logger:
    """
    Configure application-wide logging.
    
    Sets up both console and file handlers with appropriate formatting
    and log levels as defined in the Config class.
    
    Args:
        log_file: Optional custom log file path. If None, uses config.log_file
        
    Returns:
        Configured Logger instance for the application
    """
    logger = logging.getLogger("rvc_trainer")
    logger.setLevel(logging.DEBUG)
    
    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(config.log_level_console)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler for errors
    log_path = log_file or config.log_file
    try:
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(config.log_level_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except (IOError, OSError) as e:
        logger.warning(f"Could not create log file at {log_path}: {e}")
    
    return logger


# ========================================================================== #
# UTILITY FUNCTIONS
# ========================================================================== #

def validate_model_name(name: str) -> Tuple[bool, str]:
    """
    Validate model name for allowed characters.
    
    Checks that the model name contains only valid characters
    and meets length requirements as defined in Config.
    
    Args:
        name: Model name to validate
        
    Returns:
        Tuple of (is_valid, error_message)
        
    Examples:
        >>> validate_model_name("my-model_v1")
        (True, "")
        >>> validate_model_name("invalid name!")
        (False, "Name 'invalid name!' contains invalid characters...")
    """
    if not name or not name.strip():
        return False, "Model name cannot be empty"
    
    if not re.match(config.valid_model_name_pattern, name.strip()):
        return False, (
            f"Name '{name}' contains invalid characters! "
            "Use only letters, numbers, underscores, and hyphens."
        )
    
    if len(name) > config.max_model_name_length:
        return False, f"Model name too long (max {config.max_model_name_length} characters)"
    
    return True, ""


def validate_dataset_folder(folder_path: str) -> Tuple[bool, str]:
    """
    Validate dataset folder exists and contains files.
    
    Performs comprehensive validation of the dataset folder including
    existence check, permission verification, and content inspection.
    
    Args:
        folder_path: Path to dataset folder
        
    Returns:
        Tuple of (is_valid, error_message)
        
    Examples:
        >>> validate_dataset_folder("/path/to/dataset")
        (True, "")
        >>> validate_dataset_folder("/nonexistent/path")
        (False, "Folder '/nonexistent/path' does not exist!")
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
            try:
                logger = logging.getLogger("rvc_trainer")
                logger.warning(f"No common audio files found in {folder_path}")
            except Exception:
                pass
            
    except PermissionError:
        return False, f"Permission denied accessing '{folder_path}'"
    
    return True, ""


def get_sample_rate(sr: str, version: str) -> str:
    """
    Convert UI sample rate to actual sample rate value.
    
    Maps the user-friendly sample rate labels to their actual
    numeric values used in processing.
    
    Args:
        sr: Sample rate from UI ('40k' or '32k')
        version: Model version ('v1' or 'v2')
        
    Returns:
        Actual sample rate string ('48000' or '32000')
        
    Examples:
        >>> get_sample_rate("40k", "v2")
        '48000'
        >>> get_sample_rate("32k", "v2")
        '32000'
    """
    if sr == "40k" or version == "v1":
        return "48000"
    return "32000"


# Import Path for dataset validation
from pathlib import Path


# ========================================================================== #
# UNWANTED LOG PATTERNS FOR TRAINING OUTPUT FILTERING
# ========================================================================== #

UNWANTED_LOG_PATTERNS = [
    "All log messages before absl::InitializeLog()",
    "Unable to register cuDNN factory",
    "Unable to register cuBLAS factory",
    "computation placer already registered"
]


def filter_training_output(line: str) -> bool:
    """
    Filter unwanted lines from training output.
    
    Removes common noisy log messages from TensorFlow/CUDA
    that don't provide useful information to users.
    
    Args:
        line: Line of output to check
        
    Returns:
        True if line should be included, False otherwise
    """
    return not any(pattern in line for pattern in UNWANTED_LOG_PATTERNS)


# ========================================================================== #
# EXPORTS
# ========================================================================== #

__all__ = [
    # Main config
    'Config',
    'config',
    'args',
    'iscolab',
    'BASE_ROOT',
    
    # Classes
    'GPUManager',
    'PretrainedModelFinder',
    
    # Functions
    'setup_logging',
    'validate_model_name',
    'validate_dataset_folder',
    'get_sample_rate',
    'filter_training_output',
    
    # Constants
    'UNWANTED_LOG_PATTERNS',
]
