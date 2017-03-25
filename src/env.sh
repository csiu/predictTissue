#Set up cudnn defaults
cudnn_path="/home/jgrewal/software/nvidia_libraries/cudnn/cudnn8.0_linux-x64-v5.1"
export LD_LIBRARY_PATH="$cudnn_path/lib64/:$LD_LIBRARY_PATH"
export CPATH="$cudnn_path/include:$CPATH"
export LIBRARY_PATH="$cudnn_path/lib64:$LIBRARY_PATH"

#Version in use for GPU:
# - python 2.7.13
# - theano 0.7.0
