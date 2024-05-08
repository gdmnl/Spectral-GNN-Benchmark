# run_param (critical params), fullbatch, Filter bank model+conv
source scripts/ck_path.sh
DEV=${1:--1}
SEED_P=0
ARGS_P=(
    "--n_trials" "300"
    "--loglevel" "30"
    "--epoch" "200"
    "--patience" "50"
    "--suffix" "fb_bank"
)
DATAS=("cora" "citeseer" "pubmed")
PARLIST="num_hops,in_layers,out_layers,hidden,lr_lin,lr_conv"

for data in ${DATAS[@]}; do
    # AdaGNN
    python run_param.py --dev $DEV --seed $SEED_P --param $PARLIST \
        --data $data --model AdaGNN --conv AdaConv \
        "${ARGS_P[@]}"

    # ACMGNN/FBGNN-I/II
    for alpha in 1 2; do
        for theta_scheme in "low-high-id" "low-high"; do
            python run_param.py --dev $DEV --seed $SEED_P --param $PARLIST \
                --data $data --model ACMGNN --conv ACMConv \
                --alpha $alpha --theta_scheme $theta_scheme \
                "${ARGS_P[@]}"
        done
    done
done
