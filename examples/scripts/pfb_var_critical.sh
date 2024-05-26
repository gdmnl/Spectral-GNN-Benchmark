# run_param (critical params), fullbatch, Iterative/DecoupledVar
source scripts/ck_path.sh
DEV=${1:--1}
SEED_P=0
ARGS_P=(
    "--n_trials" "100"
    "--loglevel" "30"
    "--epoch" "200"
    "--patience" "50"
    "--theta_scheme" "ones"
    "--theta_param" "1.0"
    "--suffix" "fb_var"
)
MODELS=("Iterative" "DecoupledVar")
DATAS=("cora" "citeseer" "chameleon_filtered" "actor")
CONVS=("AdjConv" "HornerConv" "ChebConv" "ClenshawConv" "ChebIIConv" "BernConv" "LegendreConv" "JacobiConv" "FavardConv" "OptBasisConv")
PARLIST="num_hops,in_layers,out_layers,hidden,lr_lin,lr_conv"

for model in ${MODELS[@]}; do
    for data in ${DATAS[@]}; do
        for conv in ${CONVS[@]}; do
            python run_param.py --dev $DEV --seed $SEED_P --param $PARLIST \
                --data $data --model $model --conv $conv \
                "${ARGS_P[@]}"
        done
    done
done
