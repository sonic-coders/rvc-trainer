# GUI BY BF667


import gradio as gr
import os
import re
import shutil
import traceback
from urllib.parse import urlparse
from subprocess import PIPE, STDOUT, Popen
import numpy as np


BASE_ROOT = sys.path.append(os.getcwd())

# Configuration
ROOT_DIR = f"{BASE_ROOT}/rvc-trainer"
SAVE_DIR = f"{ROOT_DIR}/drive/MyDrive/rvc-trainer"
config = type('config', (), {
    'n_cpu': os.cpu_count() or 4,
    'iscolab': True,
    'noautoopen': False,
    'listen_port': 7860
})()

# GPU detection
try:
    import torch
    gpus = ",".join([str(i) for i in range(torch.cuda.device_count())]) if torch.cuda.is_available() else "0"
    gpu_info = f"Available GPUs: {torch.cuda.device_count()}" if torch.cuda.is_available() else "No GPU detected"
except:
    gpus = "0"
    gpu_info = "No GPU detected"

F0GPUVisible = True
default_batch_size = 8

def change_f0_method(f0_method):
    if f0_method == "rmvpe_gpu":
        return gpus
    return "0"

def change_sr2(sr, if_f0, version):
    pretrained = lambda sr_val, letter: [os.path.abspath(os.path.join('assets/pretrained_v2', file)) 
                                        for file in os.listdir('assets/pretrained_v2') 
                                        if file.endswith('.pth') and sr_val in file and letter in file]
    
    if version == "v1":
        sr2_val = "40k"
    else:
        sr2_val = sr
    
    g_choices = pretrained(sr2_val, 'G')
    d_choices = pretrained(sr2_val, 'D')
    
    return (
        gr.update(choices=g_choices, value=g_choices[0] if g_choices else ''),
        gr.update(choices=d_choices, value=d_choices[0] if d_choices else '')
    )

def change_version19(sr, if_f0, version):
    if version == "v1":
        sr2_val = "40k"
        sr_update = gr.update(value="40k", visible=False)
    else:
        sr2_val = sr
        sr_update = gr.update(visible=True)
    
    pretrained = lambda sr_val, letter: [os.path.abspath(os.path.join('assets/pretrained_v2', file)) 
                                        for file in os.listdir('assets/pretrained_v2') 
                                        if file.endswith('.pth') and sr_val in file and letter in file]
    
    g_choices = pretrained(sr2_val, 'G')
    d_choices = pretrained(sr2_val, 'D')
    
    return (
        gr.update(choices=g_choices, value=g_choices[0] if g_choices else ''),
        gr.update(choices=d_choices, value=d_choices[0] if d_choices else ''),
        sr_update
    )

def change_f0(if_f0, sr, version):
    if if_f0:
        f0_choices = ["rmvpe", "hpa-rmvpe"]
        f0_value = "rmvpe_gpu"
    else:
        f0_choices = ["crepe", "rmvpe", "hpa-rmvpe"]
        f0_value = "crepe"
    
    pretrained = lambda sr_val, letter: [os.path.abspath(os.path.join('assets/pretrained_v2', file)) 
                                        for file in os.listdir('assets/pretrained_v2') 
                                        if file.endswith('.pth') and sr_val in file and letter in file]
    
    sr_val = "40k" if version == "v1" else sr
    g_choices = pretrained(sr_val, 'G')
    d_choices = pretrained(sr_val, 'D')
    
    return (
        gr.update(choices=f0_choices, value=f0_value),
        gr.update(choices=g_choices, value=g_choices[0] if g_choices else ''),
        gr.update(choices=d_choices, value=d_choices[0] if d_choices else '')
    )

# ========================================================================== #
# STEP 2: Data Processing Functions
# ========================================================================== #

def preprocess_dataset(dataset_folder, training_name, sr2, np7):
    try:
        # Validate model name
        if not re.match(r"^[a-zA-Z0-9_]+$", training_name):
            return f"Error: Name '{training_name}' contains invalid characters!"
        
        # Create directories
        os.makedirs(f'{SAVE_DIR}/{training_name}', exist_ok=True)
        
        # Check dataset folder
        if not os.path.exists(dataset_folder):
            return f"Error: Folder '{dataset_folder}' does not exist!"
        if not os.listdir(dataset_folder):
            return f"Error: Folder '{dataset_folder}' is empty!"
        
        # Convert sample rate
        sample_rate = "48000" if sr2 == "40k" else "32000"
        percentage = 3.0
        normalize = True
        
        # Dataset segmentation and resampling
        preprocess_script = f"{ROOT_DIR}/rvc/train/preprocess/preprocess.py"
        !python {preprocess_script} {SAVE_DIR}/{training_name} {dataset_folder} {percentage} {sample_rate} {normalize}
        
        return "✅ Data preprocessing completed successfully!"
    except Exception as e:
        return f"❌ Error during preprocessing: {str(e)}"

def extract_f0_feature(gpus6, np7, f0method8, if_f0_3, training_name, version19, gpus_rmvpe):
    try:
        # Convert sample rate
        sample_rate = "48000" if version19 == "v2" else "32000"
        arch_fairseq = "Fairseq"
        f0_method = f0method8.replace("_gpu", "")
        
        # Extraction of average pitch and sound characteristics
        preparing_data_script = f"{ROOT_DIR}/rvc/train/preprocess/preparing_data.py"
        !python {preparing_data_script} {SAVE_DIR}/{training_name} {arch_fairseq} {f0_method} {sample_rate} 2
        
        return "✅ Feature extraction completed successfully!"
    except Exception as e:
        return f"❌ Error during feature extraction: {str(e)}"

def train_index(training_name, version19):
    try:
        index_algorithm = "Faiss"
        
        # Generate index file based on characteristics
        extract_index_script = f"{ROOT_DIR}/rvc/train/preprocess/extract_index.py"
        !python {extract_index_script} {SAVE_DIR}/{training_name} {index_algorithm}
        
        return "✅ Index training completed successfully!"
    except Exception as e:
        return f"❌ Error during index training: {str(e)}"

# ========================================================================== #
# STEP 3: Model Training Functions
# ========================================================================== #

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
):
    try:
        # Convert sample rate
        sample_rate = "48000" if sr2 == "40k" else "32000"
        
        # Convert parameters
        save_epoch_interval = int(save_epoch10)
        total_epochs = int(total_epoch11)
        batch_size = int(batch_size12)
        vocoder = "HiFi-GAN"
        optimizer = "AdamW"
        
        # Handle pretrained models
        pretrained_G = pretrained_G14 if pretrained_G14 else None
        pretrained_D = pretrained_D15 if pretrained_D15 else None
        
        # Save settings
        save_to_zip = True
        save_half = True
        
        print("\nStarting training...")
        print(f"Model: {training_name}")
        print(f"Epochs: {total_epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Sample rate: {sample_rate}")
        
        # Build command
        cmd = (
            f'python {ROOT_DIR}/rvc/train/train.py '
            f'--experiment_dir "{SAVE_DIR}" '
            f'--model_name "{training_name}" '
            f'--batch_size {batch_size} '
            f'--sample_rate {sample_rate} '
            f'--total_epoch {total_epochs} '
            f'--save_every_epoch {save_epoch_interval} '
            f'--vocoder "{vocoder}" '
            f'--optimizer {optimizer} '
            f'--save_to_zip {save_to_zip} '
            f'--save_half {save_half} '
            f'{"--pretrain_g %s" % pretrained_G if pretrained_G is not None else ""} '
            f'{"--pretrain_d %s" % pretrained_D if pretrained_D is not None else ""}'
        )
        
        # Run training
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
        
        output_lines = []
        for line in p.stdout:
            line = line.strip()
            if not any(unwanted in line for unwanted in [
                "All log messages before absl::InitializeLog()",
                "Unable to register cuDNN factory",
                "Unable to register cuBLAS factory",
                "computation placer already registered"
            ]):
                output_lines.append(line)
                print(line)
        
        p.wait()
        
        return "✅ Training completed successfully!"
    except Exception as e:
        with open(f"{SAVE_DIR}/{training_name}/error_log.txt", "w") as f:
            f.write("An error occurred:\n")
            f.write(traceback.format_exc())
        return f"❌ Error during training: {str(e)}"

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
    try:
        # Step 1: Preprocess
        result1 = preprocess_dataset(dataset_folder, training_name, sr2, np7)
        if "Error" in result1:
            return result1
        
        # Step 2: Extract features
        result2 = extract_f0_feature(gpus16, np7, f0method8, if_f0_3, training_name, version19, gpus_rmvpe)
        if "Error" in result2:
            return result2
        
        # Step 3: Train index
        result3 = train_index(training_name, version19)
        if "Error" in result3:
            return result3
        
        # Step 4: Train model
        result4 = click_train(
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
        )
        
        return f"✅ All steps completed!\n\n{result1}\n{result2}\n{result3}\n{result4}"
    except Exception as e:
        return f"❌ Error during one-click training: {str(e)}"

def download_model_files(training_name):
    try:
        # Get model files
        weights_path = f'assets/weights/{training_name}'
        logs_path = f'logs/{training_name}'
        
        files = []
        if os.path.exists(weights_path):
            files.extend([os.path.join(weights_path, f) for f in os.listdir(weights_path) if f.endswith('.pth')])
        
        if os.path.exists(logs_path):
            files.extend(glob.glob(f'{logs_path}/added_*.index'))
        
        return files, f"Found {len(files)} files"
    except Exception as e:
        return [], f"Error: {str(e)}"

# ========================================================================== #
# GRADIO UI
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
                    
                    easy_uploader = gr.Files(
                        label="Drop your audio files here",
                        file_types=['audio']
                    )
                    
                    # Hidden upload handler
                    def handle_upload(files, folder):
                        if folder == "":
                            gr.Warning('Please enter a folder name for your dataset')
                            return
                        os.makedirs(folder, exist_ok=True)
                        for f in files:
                            shutil.copy2(f.name, os.path.join(folder, os.path.split(f.name)[1]))
                    
                    easy_uploader.upload(
                        fn=handle_upload,
                        inputs=[easy_uploader, dataset_folder],
                        outputs=[]
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
                        # This is a placeholder - actual pretrained files would be checked
                        return gr.update(), gr.update()
                    
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
                [dataset_folder, training_name, sr2, np7],
                [info1],
                api_name="preprocess"
            )
            
            # Button 2: Extract Features
            but2.click(
                extract_f0_feature,
                [gpus6, np7, f0method8, if_f0_3, training_name, version19, gpus_rmvpe],
                [info2],
                api_name="extract_features"
            )
            
            # Button 3: Train Index
            but4.click(
                train_index,
                [training_name, version19],
                [info3],
                api_name="train_index"
            )
            
            # Button 4: Train Model
            but3.click(
                click_train,
                [
                    training_name,
                    sr2,
                    if_f0_3,
                    spk_id5,
                    save_epoch10,
                    total_epoch11,
                    batch_size12,
                    "yes",  # if_save_latest13
                    pretrained_G14,
                    pretrained_D15,
                    gpus16,
                    if_cache_gpu17,
                    if_save_every_weights18,
                    version19,
                ],
                [info3],
                api_name="train_model"
            )
            
            # Button 5: One Click Training
            but5.click(
                train1key,
                [
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
                    "yes",  # if_save_latest13
                    pretrained_G14,
                    pretrained_D15,
                    gpus16,
                    if_cache_gpu17,
                    if_save_every_weights18,
                    version19,
                    gpus_rmvpe,
                ],
                [info3],
                api_name="one_click_train"
            )
            
            # Download Model
            download_model.click(
                download_model_files,
                [training_name],
                [model_files, info3]
            )
            
            # F0 method change
            f0method8.change(
                change_f0_method,
                [f0method8],
                [gpus_rmvpe]
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
                    optimizer = gr.Dropdown(
                        label="Optimizer",
                        choices=["AdamW", "AdaBelief"],
                        value="AdamW",
                        interactive=True
                    )
                    vocoder = gr.Dropdown(
                        label="Vocoder",
                        choices=["HiFi-GAN"],
                        value="HiFi-GAN",
                        interactive=True
                    )
                    save_half = gr.Checkbox(label="Save with Half Precision", value=True)
                    save_to_zip = gr.Checkbox(label="Package Model in ZIP", value=True)
            
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
