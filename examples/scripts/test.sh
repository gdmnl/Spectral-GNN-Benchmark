# ChebConv2, BernConv, ClenShaw, Horner
# python run_single.py --data cora --model Iterative --conv HornerConv --alpha 0.0
# python run_single.py --data cora --model DecoupledVar --conv HornerConv --alpha 0.0
# python run_single.py --data cora --model DecoupledVar --conv JacobiConv --alpha 0.0 --epoch 2000

CUDA_VISIBLE_DEVICES=3 python run_single.py --data 2dgrid --img_idx 0 --model DecoupledVar --conv BernConv --theta_scheme appr --data_split '60/20/20'\
 --theta_param 0.1 --in_layers 1 --out_layers 1 --task filtering --epoch 10000 --num_hops 4 --filter_type low --patience 200 --hidden 32 --dp_lin 0 --dp_conv 0 --lr_lin 0.01 --lr_conv 0.01