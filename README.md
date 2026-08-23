# 🔊 RVC Voice Trainer

<p align="center">
  <strong>Train high-quality AI voice models with RVC (Retrieval-based Voice Conversion)</strong>
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#installation">Installation</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#usage">Usage</a> •
  <a href="#huggingface-integration">HuggingFace</a> •
  <a href="#configuration">Configuration</a> •
  <a href="#license">License</a>
</p>

---

## 📖 About

**RVC Trainer** is a user-friendly Gradio-based web interface for training RVC (Retrieval-based Voice Conversion) voice models. This project provides an intuitive GUI that simplifies the entire voice model training pipeline, from data preprocessing to model deployment.

This is an enhanced fork of [Poltrain](https://github.com/poltrain/rvc-trainer) with additional features including:

- 🚀 **One-click training pipeline**
- 🤗 **HuggingFace Hub integration** for easy model sharing
- 📊 **Improved code architecture** with proper error handling and logging
- ✅ **Input validation** and comprehensive documentation

---

## ✨ Features

### Core Training Pipeline
- **📁 Dataset Management**: Upload, organize, and preprocess audio datasets
- **🎵 Feature Extraction**: F0 (pitch) extraction using RMVPE, Crepe, or HPA-RMVPE
- **🔍 Index Training**: Generate Faiss indexes for fast similarity search
- **🚀 Model Training**: Train RVC v1/v2 models with customizable parameters
- **📦 Model Export**: Download trained models as ZIP files

### Advanced Features
- **🌟 One-Click Training**: Automate the entire training workflow with a single click
- **⚙️ Advanced Settings**: Fine-tune optimizer, vocoder, batch size, epochs, and more
- **🎛️ GPU Support**: Multi-GPU training with automatic detection
- **📈 Real-time Monitoring**: Live training output with filtered logs

### HuggingFace Integration
- **🔐 Secure Authentication**: API key validation before upload
- **📦 Auto-Detection**: Automatically find model zip files by name
- **🏗️ Repository Management**: Auto-create repositories if they don't exist
- **📝 Auto-Generated README**: Professional model cards with attribution
- **🔒 Privacy Options**: Choose between public or private repositories

---

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (recommended, CPU mode available)
- At least 10 minutes of clean audio data for training

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/sonic-coders/rvc-trainer.git
cd rvc-trainer

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Install additional dependencies for HuggingFace upload
pip install huggingface_hub

# Run the application
python app.py
```

### Dependencies

The application requires the following main packages:
- `gradio` - Web UI framework
- `torch` - Deep learning framework
- `numpy` - Numerical computations
- `huggingface_hub` - HuggingFace API (optional, for uploads)

---

## 🚀 Quick Start

### 1. Launch the Application

```bash
# Local development
python app.py

# Google Colab mode
python app.py --colab
```

The application will open in your default browser at `http://localhost:7860`

### 2. Basic Workflow

#### Step 1: Prepare Your Dataset
1. Gather clean audio recordings (10+ minutes recommended)
2. Place audio files in a folder (e.g., `dataset/`)
3. Supported formats: `.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a`

#### Step 2: Configure Training
1. **Model Name**: Enter a unique identifier (letters, numbers, underscores only)
2. **Dataset Folder**: Path to your audio files
3. **Sampling Rate**: 
   - `40k` - Higher quality (48kHz), recommended for singing
   - `32k` - Faster training (32kHz), good for speech
4. **F0 Method**: 
   - `rmvpe` - Best accuracy for most cases
   - `crepe` - Good for non-singing voice
   - `hpa-rmvpe` - Hybrid approach

#### Step 3: Run Training

**Option A: Step-by-Step**
1. Click **"1. Process Data"** - Preprocess audio files
2. Click **"2. Extract Features"** - Extract pitch and characteristics
3. Click **"3. Train Index"** - Build similarity index
4. Click **"4. Train Model"** - Train the voice model

**Option B: One-Click Training**
- Click **"🌟 One Click Training"** - Runs all steps automatically

#### Step 4: Download Model
- Click **"5. Download Model"** to get your trained model files

---

## 📱 Interface Overview

The application consists of four main tabs:

### 🎯 Tab 1: Train
Main training interface with three columns:

| Column | Purpose |
|--------|---------|
| **Left** | Dataset & Preprocessing settings |
| **Middle** | Feature Extraction configuration |
| **Right** | Training parameters & execution |

**Key Controls:**
- **Model Name**: Unique identifier for your model
- **Dataset Folder**: Path to your audio dataset
- **CPU Processes**: Number of cores for pitch extraction
- **Sampling Rate**: Audio quality setting (40k/32k)
- **F0 Method**: Pitch extraction algorithm
- **Version**: RVC version (v1/v2)
- **Speaker ID**: Multi-speaker support (0-4)
- **Epochs**: Training iterations (100-500 recommended)
- **Batch Size**: Samples per batch (adjust based on VRAM)

**Advanced Settings (Accordion):**
- GPU selection
- Save frequency
- Batch size optimization
- Cache options
- Pretrained model selection

### ⚙️ Tab 2: Settings
Global training settings:
- Data processing options (normalization, fragment length)
- Training parameters (optimizer, vocoder)
- Export options (half precision, ZIP packaging)

### 🤗 Tab 3: HuggingFace
Upload your trained models to HuggingFace Hub:
- **API Key Authentication**: Secure token validation
- **Repository ID**: Target repository (username/repo-name)
- **Model Selection**: Auto-detect zip from model name
- **Privacy Control**: Public or private repository
- **Auto README**: Generated model card with attribution

---

## 🤗 HuggingFace Integration

### Getting Started

1. **Get API Key**:
   - Visit [HuggingFace Settings/Tokens](https://huggingface.co/settings/tokens)
   - Create new token with **Write** permission
   - Copy token (starts with `hf_`)

2. **Configure Upload**:
   ```
   API Key: hf_xxxxxxxxxxxxxxxxxxxx
   Repository ID: username/my-model-name
   Model Name: My-Voice (same as training name)
   ```

3. **Push to Hub**:
   - Click "Verify API Key" to validate credentials
   - Click "Detect Zip File" to confirm model location
   - Click "Push to HuggingFace" to upload

### Auto-Generated README

Every uploaded model includes a professional README:

```markdown
---
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
- **Uploaded Date**: {timestamp}
- **Sample Rate**: {sample_rate}kHz
- **Version**: {version}
```

---

## ⚙️ Configuration

### Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--colab` | flag | False | Enable Google Colab mode |

### Default Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Port | 7860 | Gradio server port |
| Batch Size | 8 | Training batch size |
| Epochs | 150 | Default training epochs |
| Sample Rate | 40k | Default audio quality |
| Version | v2 | RVC version |
| F0 Method | rmvpe | Pitch extraction |

### Environment Variables

You can customize behavior through environment variables or by editing `configs/config.py`:

```python
# Example configuration
config = {
    'n_cpu': 4,              # Number of CPU cores
    'listen_port': 7860,      # Server port
    'default_batch_size': 8,  # Default batch size
    ...
}
```

---

## 📁 Project Structure

```
rvc-trainer/
├── app.py                      # Main Gradio application
├── configs/
│   └── config.py               # Configuration settings
├── rvc/
│   ├── train/
│   │   ├── train.py            # Training script
│   │   ├── preprocess/
│   │   │   ├── preprocess.py   # Data preprocessing
│   │   │   ├── preparing_data.py # Feature extraction
│   │   │   ├── extract_index.py # Index generation
│   │   │   └── slicer.py       # Audio slicing
│   │   ├── losses.py           # Loss functions
│   │   ├── mel_processing.py   # Mel spectrogram
│   │   └── visualization.py    # Training visualization
│   ├── lib/
│   │   ├── audio.py            # Audio utilities
│   │   ├── fairseq.py          # Fairseq integration
│   │   ├── rmvpe.py            # RMVPE pitch extraction
│   │   └── algorithm/          # Model architectures
│   └── configs/                # Model configurations
├── assets/
│   ├── pretrained_v2/          # Pretrained weights
│   └── weights/                # Trained model outputs
├── logs/                       # Training logs and features
├── download_files.py           # Asset downloader
├── README.md                   # This file
└── LICENSE                     # MIT License
```

---

## ❓ FAQ

### Q: How much audio data do I need?
**A:** Minimum 10 minutes, but 30+ minutes recommended for best quality.

### Q: Which sampling rate should I choose?
**A:** 
- Use **40k** for singing or high-quality voice conversion
- Use **32k** for faster training with speech-only content

### Q: What's the difference between v1 and v2?
**A:** 
- **v1**: Original RVC architecture, stable and well-tested
- **v2**: Improved architecture with better quality and efficiency

### Q: My training is slow. How can I speed it up?
**A:**
- Increase batch size (if VRAM allows)
- Use GPU acceleration
- Reduce sample rate to 32k
- Decrease total epochs

### Q: Can I use multiple GPUs?
**A:** Yes! Enter GPU IDs as `0-1-2` for multi-GPU training.

---

## 🐛 Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| "No GPU detected" | Install CUDA toolkit and PyTorch with CUDA support |
| "CUDA out of memory" | Reduce batch size or use gradient accumulation |
| "Folder does not exist" | Check dataset path and ensure correct absolute path |
| "Invalid API Key" | Verify token has Write permission at HF settings |
| "Preprocessing failed" | Check audio format and ensure files aren't corrupted |

### Getting Help

- Open an issue on [GitHub Issues](https://github.com/sonic-coders/rvc-trainer/issues)
- Check the [RVC Documentation](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion)
- Join community discussions

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Code Style
- Follow PEP 8 guidelines
- Add docstrings to all functions
- Include type hints where appropriate
- Test changes before submitting PR

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2026 sonic-coders

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 🙏 Acknowledgments

- [RVC Project](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion) - Original RVC implementation
- [Poltrain](https://github.com/poltrain/rvc-trainer) - Base fork
- [Gradio](https://gradio.app/) - Web UI framework
- [HuggingFace](https://huggingface.co/) - Model hosting platform
- [PyTorch](https://pytorch.org/) - Deep learning framework

---

## 📞 Contact

- **GitHub**: [sonic-coders/rvc-trainer](https://github.com/sonic-coders/rvc-trainer)
- **Issues**: [Report a bug](https://github.com/sonic-coders/rvc-trainer/issues)
- **Discussions**: [Join discussion](https://github.com/sonic-coders/rvc-trainer/discussions)

---

<p align="center">
  <strong>Made with ❤️ by <a href="https://github.com/sonic-coders">sonic-coders</a></strong>
</p>

<p align="center">
  <sub>This model was uploaded with <a href="https://github.com/sonic-coders/rvc-trainer">RVC TRAINER</a></sub>
</p>
