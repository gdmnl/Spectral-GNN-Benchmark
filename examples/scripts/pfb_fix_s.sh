# run_param+run_best, fullbatch, DecoupledFixed-theta, small-scale dataset
source scripts/ck_path.sh
DEV=${1:--1}
SEED_P=1
SEED_S="20,21,22,23,24,25,26,27,28,29"
ARGS_P=(
    "--n_trials" "300"
    "--loglevel" "30"
    "--num_hops" "10"
    "--in_layers" "1"
    "--out_layers" "1"
    "--hidden" "128"
    "--epoch" "200"
    "--patience" "50"
    "--suffix" "small"
)
ARGS_S=(
    "--seed_param" "$SEED_P"
    "--loglevel" "25"
    "--num_hops" "10"
    "--in_layers" "1"
    "--out_layers" "1"
    "--hidden" "128"
    "--epoch" "500"
    "--patience" "-1"
)

DATAS=("cora" "citeseer" "pubmed")
MODELS=("DecoupledFixed")
CONVS=AdjConv
SCHEMES=("impulse" "appr" "nappr" "hk" "mono")
PARLIST="theta_param,normg,dp_lin,dp_conv,lr_lin,lr_conv,wd_lin,wd_conv"

for data in ${DATAS[@]}; do
    for model in ${MODELS[@]}; do
        for scheme in ${SCHEMES[@]}; do
            # Run hyperparameter search
            python run_param.py --dev $DEV --seed $SEED_P --param $PARLIST \
                --data $data --model $model --conv $CONVS --theta_scheme $scheme \
                "${ARGS_P[@]}"

            # Run repeatative with best hyperparameters
            python run_best.py --dev $DEV --seed $SEED_S --seed_param $SEED_P \
                --data $data --model $model --conv $CONVS --theta_scheme $scheme \
                "${ARGS_S[@]}"
        done
    done
done
