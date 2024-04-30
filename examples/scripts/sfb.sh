# run_single, fullbatch
# Decoupled
python run_single.py --data cora --model DecoupledFixed --conv AdjConv \
    --theta_scheme appr --theta_param 0.1 --alpha 0.0 \
    --num_hops 20 --hidden 256 \
    --lr 0.1 --wd 0.0005
# Iterative
python run_single.py --data cora --model Iterative --conv AdjConv \
    --alpha 0.0 \
    --num_hops 10 --hidden 256 \
    --lr 0.1 --wd 0.0005
