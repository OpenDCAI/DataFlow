CUDA_VISIBLE_DEVICES=0

python evaluate_Multiprocess.py \
  --load_file_path "/clean_prm_sftData.json" \
  --task_name "math" \
  --file "math_500" \
  --propose_method "llama" \
  --value_method "local" \
  --mode "mcts" \
  --evaluate "math" \
  --iteration_limit 8 \
  --use_reflection "simple" \
  --branch 3 \
  --save_name "test" \