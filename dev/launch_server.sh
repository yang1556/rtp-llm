export CUDA_VISIBLE_DEVICES="4"

# export SP_TOKENIZER_PATH=/root/models/mtp/target_model
# export SP_INT8_MODE=0
# export SP_TYPE=eagle
# export SP_WEIGHTS_TYPE=BF16
# export SP_MIN_TOKEN_MATCH=2
# export SP_MAX_TOKEN_MATCH=2
# export GEN_NUM_PER_CIRCLE=5

# export SP_CHECKPOINT_PATH=/root/models/mtp/draft_model

# export SP_WEIGHT_TYPE=BF16
# export SP_MODEL_TYPE=qwen_2-mtp
# export SP_ACT_TYPE=BF16


export CHECKPOINT_PATH=/home/admin/workspace/models/mtp/target_model
export TOKENIZER_PATH=/home/admin/workspace/models/mtp/target_model
export MODEL_TYPE=qwen_2 # 根据需要设置

export PYTHONPATH=~/workspace/hzy/rtp-llm/bazel-out/k8-opt/bin/:$PYTHONPATH
export TP_SIZE=1
export DP_SIZE=1
export WORLD_SIZE=1
export EP_SIZE=1
export START_PORT=26000
export USE_RPC_MODEL=1
export MAX_SEQ_LEN=15000
export ENABLE_FMHA=on
export WARM_UP=0
export LOG_LEVEL=INFO
export NCCL_DISABLE_ABORT=1
export FT_DISABLE_CUSTOM_AR=1
export CUDA_LAUNCH_BLOCKING=0
export NOT_USE_DEFAULT_STREAM=True
export DEVICE_RESERVE_MEMORY_BYTES=-20480000000
export ACT_TYPE=BF16
export RESERVER_RUNTIME_MEM_MB=8024
export ENABLE_COMM_OVERLAP=0
/opt/conda310/bin/python3 -m rtp_llm.start_server