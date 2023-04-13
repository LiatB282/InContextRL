#! /bin/bash
export CUDA_VISIBLE_DEVICES="1,2,3,4,5,6,7"
/specific/netapp5_2/gamir/liat/anaconda3/bin/python train_text_generation.py \
--config_path /home/gamir/liat/InContextRL/scripts/training/task_configs/trivia_qa/t5_ppo.yml \
--entity_name liatbezalel \
--project_name IN_CONTEXT_RL \
--experiment_name eval0 \