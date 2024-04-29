# run_param (critical params), fullbatch, Iterative/DecoupledVar
source scripts/ck_path.sh
DEV=${1:--1}
SEED_P=0
ARGS_P=(
    "--n_trials" "300"
    "--loglevel" "30"
    "--epoch" "200"
    "--patience" "50"
    "--theta_scheme" "ones"
    "--theta_param" "1.0"
)
# DATAS=("cora" "citeseer" "pubmed")
DATAS=("cora")
# MODELS=("Iterative" "DecoupledVar")
MODELS=("DecoupledVar")
CONVS=("AdjConv" "ChebConv")
PARLIST="num_hops,in_layers,out_layers,hidden,lr,wd"

for data in ${DATAS[@]}; do
    for model in ${MODELS[@]}; do
        for conv in ${CONVS[@]}; do
            python run_param.py --dev $DEV --seed $SEED_P --param $PARLIST \
                --data $data --model $model --conv $conv \
                "${ARGS_P[@]}"
        done
    done
done
