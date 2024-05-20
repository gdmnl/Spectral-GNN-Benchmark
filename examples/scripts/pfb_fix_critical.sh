# run_param (critical params), fullbatch, DecoupledFixed-theta
source scripts/ck_path.sh
DEV=${1:--1}
SEED_P=0
ARGS_P=(
    "--n_trials" "300"
    "--loglevel" "30"
    "--epoch" "200"
    "--patience" "50"
    "--suffix" "fb_fix"
)
DATAS=("cora" "citeseer" "pubmed")
MODELS=("DecoupledFixed")
CONVS=AdjConv
SCHEMES=("impulse" "mono" "appr" "hk" "gaussian")
PARLIST="num_hops,in_layers,out_layers,hidden,theta_param,lr_lin"

for data in ${DATAS[@]}; do
    for model in ${MODELS[@]}; do
        for scheme in ${SCHEMES[@]}; do
            python run_param.py --dev $DEV --seed $SEED_P --param $PARLIST \
                --data $data --model $model --conv $CONVS --theta_scheme $scheme \
                "${ARGS_P[@]}"
        done
    done
done
