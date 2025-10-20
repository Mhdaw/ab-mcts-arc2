#!/bin/bash

CONFIG_NAME=kaggle.yaml
EXP_ID=abmcts
MAX_NUM_NODES=250
ALGO_CLASS_NAME=ABMCTSA
DIST_TYPE=beta

# Number of parallel jobs
N_JOBS=3
INDICES_FILE="experiments/arc2/arc_agi_2_eval_short.txt"

# Track execution time
start_time=$(date +%s)

# Execute tasks in parallel using GNU parallel
# Each task ID from the indices file will be processed concurrently
cat $INDICES_FILE | parallel -j $N_JOBS "
    # Set task-specific variables
    export TASK_ID={}
    CKPT_PATH=outputs/arc2/arc2_${EXP_ID}_algo_${ALGO_CLASS_NAME}_${DIST_TYPE}/{}/checkpoints/checkpoint_latest.pkl
    
    # Create output directory for this task
    mkdir -p outputs/arc2/arc2_${EXP_ID}_algo_${ALGO_CLASS_NAME}_${DIST_TYPE}/{}
    
    # Build the command to run
    CMD='python experiments/arc2/run.py \\
        --config-name ${CONFIG_NAME} \\
        max_num_nodes=${MAX_NUM_NODES} \\
        task_id={} \\
        algo.class_name=${ALGO_CLASS_NAME} \\
        +algo.params.dist_type=${DIST_TYPE} \\
        hydra.run.dir=outputs/arc2/arc2_${EXP_ID}_algo_${ALGO_CLASS_NAME}_${DIST_TYPE}/{}'
    
    # Add checkpoint path if it exists
    if [ -e \"\$CKPT_PATH\" ]; then
        CMD=\"\$CMD checkpoint_path=\$CKPT_PATH\"
    fi
    
    echo 'Starting task: {}'
    eval \"\$CMD\"
"

# Calculate and display execution time
end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
elapsed_minutes=$(awk "BEGIN {printf \"%.2f\", $elapsed_time/60}")

echo "All tasks completed. Total time: ${elapsed_minutes} minutes."
