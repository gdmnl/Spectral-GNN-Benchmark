# run_single, fullbatch
# chameleon_filtered
python run_single.py --data chameleon_filtered --model Iterative --conv AdjConv \
    --theta_scheme appr --theta_param 0.1 --alpha 0.0
# squirrel_filtered
python run_single.py --data squirrel_filtered --model Iterative --conv AdjConv \
    --theta_scheme appr --theta_param 0.1 --alpha 0.0