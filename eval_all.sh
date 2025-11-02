exp_name='ppo'

# auto_torchrun -m rl.eval_all \
#     ${exp_name}_0 \
#     rl/config_eval.py \
#     --load-ppo-from './work_dirs/ppo_0.zip'

python -m rl.utils_csv \
    "./work_dirs/rl_eval_${exp_name}_0.csv" \
    ./work_dirs/rl_eval_${exp_name}_0/*.csv

# auto_torchrun -m rl.eval_all \
#     ${exp_name}_1 \
#     rl/config_eval.py \
#     --load-model-from './work_dirs/model610000.pth' \
#     --retry-from "./work_dirs/rl_eval_${exp_name}_0.csv"

# python -m rl.utils_csv \
#     "./work_dirs/rl_eval_${exp_name}_1.csv" \
#     ./work_dirs/rl_eval_${exp_name}_0.csv ./work_dirs/rl_eval_${exp_name}_1/*.csv

# auto_torchrun -m rl.eval_all \
#     ${exp_name}_2 \
#     rl/config_eval.py \
#     --load-model-from './work_dirs/model610000.pth' \
#     --retry-from "./work_dirs/rl_eval_${exp_name}_1.csv"

# python -m rl.utils_csv \
#     "./work_dirs/rl_eval_${exp_name}_2.csv" \
#     ./work_dirs/rl_eval_${exp_name}_1.csv ./work_dirs/rl_eval_${exp_name}_2/*.csv

# auto_torchrun -m rl.eval_all \
#     ${exp_name}_3 \
#     rl/config_eval.py \
#     --load-model-from './work_dirs/model610000.pth' \
#     --retry-from "./work_dirs/rl_eval_${exp_name}_2.csv"

# python -m rl.utils_csv \
#     "./work_dirs/rl_eval_${exp_name}_3.csv" \
#     ./work_dirs/rl_eval_${exp_name}_2.csv ./work_dirs/rl_eval_${exp_name}_3/*.csv