# ChebConv2, BernConv, ClenShaw, Horner
# python run_single.py --data cora --model Iterative --conv HornerConv --alpha 0.0
# python run_single.py --data cora --model DecoupledVar --conv HornerConv --alpha 0.0
# python run_single.py --data cora --model DecoupledVar --conv JacobiConv --alpha 0.0 --epoch 2000

CUDA_VISIBLE_DEVICES=3 python run_single.py --data 2dgrid --model DecoupledFixed --conv BernConv \
    --theta_scheme appr --theta_param 0.1 --alpha 0.0 --task filtering --epoch 2000 --num_hops 10 --filter_type band --patience 50