DEV=${1:-0}
FILTER_TYPES=("low" "high" "band" "rejection" "comb")

SCHEMES=("ones" "impulse" "mono" "appr" "hk" "gaussian")
THETAS=(0.5 0.5 0.5 0.5 1.0 1.0)
CONVS=("AdjConv" "HornerConv" "ChebConv" "ClenshawConv" "ChebIIConv" "BernConv" "LegendreConv" "JacobiConv" "FavardConv" "OptBasisConv")

for filter_type in ${FILTER_TYPES[@]}; do
    for img_idx in {0..5}; do

        for idx in ${!SCHEMES[@]}; do
            scheme=${SCHEMES[$idx]}
            theta_param=${THETAS[$idx]}
            python exp_regression.py --data 2dgrid --img_idx $img_idx --model "DecoupledFixed" --conv "AdjConv" --theta_scheme $scheme --data_split '60/20/20' --dev $DEV --suffix summary\
            --theta_param $theta_param --in_layers 1 --out_layers 1 --epoch 10000 --num_hops 10 --filter_type $filter_type --patience 200 --hidden 64 --dp_lin 0 --dp_conv 0 --lr_lin 0.01 --lr_conv 0.0005
        done

        for conv in ${CONVS[@]}; do
            python exp_regression.py --data 2dgrid --img_idx $img_idx --model "DecoupledVar" --conv $conv --data_split '60/20/20' --dev $DEV --suffix summary\
            --theta_param 1.0 --in_layers 1 --out_layers 1 --epoch 10000 --num_hops 10 --filter_type $filter_type --patience 200 --hidden 64 --dp_lin 0 --dp_conv 0 --lr_lin 0.01 --lr_conv 0.0005
        done

    done
done
