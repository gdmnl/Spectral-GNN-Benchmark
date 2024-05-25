# run_single, fullbatch
# Decoupled
python run_single.py --data cora --model DecoupledFixed --conv AdjConv \
    --theta_scheme appr --theta_param 0.1 --alpha 0.0
# Iterative
python run_single.py --data cora --model Iterative --conv AdjConv \
    --alpha 0.0
# ChebIIConv, BernConv
python run_single.py --data cora --model Iterative --conv Horner --alpha 0.0
# chameleon_filtered
python run_single.py --data chameleon_filtered --model Iterative --conv AdjConv \
    --theta_scheme appr --theta_param 0.1 --alpha 0.0
# squirrel_filtered
python run_single.py --data squirrel_filtered --model Iterative --conv AdjConv \
    --theta_scheme appr --theta_param 0.1 --alpha 0.0
