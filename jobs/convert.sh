#!/bin/bash
#SBATCH --job-name=test_opereCE
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=test_%j.out
set -x # Enable script debugging
 
echo "Loading module 2024..."
module load 2024
 
echo "Loading CUDA/12.1.1..."
module load CUDA/12.1.1
module load cudnn/8.6 # Ensure cuDNN module is loaded if separate
 
echo "Updating LD_LIBRARY_PATH..."
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$HOME/miniconda3/lib:$LD_LIBRARY_PATH
 
echo "Activating virtual environment..."
source /home/milerisoy/miniconda3/bin/activate audio
 
echo "Verifying Python and CUDA..."
which python
python --version
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('CUDA Devices:', torch.cuda.device_count())"

echo "Loading ffmpeg..."
module load FFmpeg/7.0.2-GCCcore-13.3.0

 
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
 
echo "Starting conversion script..."
# python aug/scripts/generate_test_set.py
python datasets/coughvid/coughvid_wav_file.py