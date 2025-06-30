#!/bin/bash

exp_name_list=(
    arc2_abmcts_algo_ABMCTSA_beta
)

num_nodes="250"

path_to_arc_tasks="./experiments/arc2/arc_agi_2_eval_short.txt"
path_to_ckpt="./outputs/arc2/{exp_name}/{task_id}/checkpoints/checkpoint_n_answers_${num_nodes}.pkl"
for exp_name in ${exp_name_list[@]}; do
    uv run eval/proc_results.py \
        --exp_name=${exp_name} \
        --path_to_arc_tasks=${path_to_arc_tasks} \
        --path_to_ckpt=${path_to_ckpt} \
        --n_jobs=4
done

exp_names_str=$(IFS=','; echo "${exp_name_list[*]}")
export exp_names_str
uv run eval/visualize.py --path_to_arc_tasks=${path_to_arc_tasks}
