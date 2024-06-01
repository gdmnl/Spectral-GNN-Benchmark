# run_param+run_best, minibatch, PrecomputedVar, lar-scale dataset
source scripts/ck_path.sh
DEV=${1:--1}
SEED_P=1
SEED_S="20,21,22,23,24,25,26,27,28,29"
ARGS_P=(
    "--n_trials" "50"
    "--loglevel" "30"
    "--num_hops" "10"
    "--in_layers" "0"
    "--out_layers" "2"
    "--hidden" "128"
    "--epoch" "200"
    "--batch" "4096"
    "--patience" "50"
    "--theta_scheme" "ones"
    "--theta_param" "1.0"
    "--normf"
    "--suffix" "mb_var"
)
ARGS_S=(
    "--seed_param" "$SEED_P"
    "--loglevel" "25"
    "--num_hops" "10"
    "--in_layers" "0"
    "--out_layers" "2"
    "--hidden" "128"
    "--epoch" "500"
    "--batch" "4096"
    "--patience" "-1"
    "--theta_scheme" "ones"
    "--theta_param" "1.0"
    "--normf"
    "--suffix" "mb_var"
)

DATAS=("ogbn-products" "snap-patents")
MODELS=("PrecomputedVar")
CONVS=("AdjiConv" "AdjConv" "HornerConv" "ChebConv" "ClenshawConv" "ChebIIConv" "BernConv" "LegendreConv" "JacobiConv" "FavardConv" "OptBasisConv")

for data in ${DATAS[@]}; do
    for model in ${MODELS[@]}; do
        for conv in ${CONVS[@]}; do
            PARLIST="normg,dp_lin,dp_conv,lr_lin,lr_conv,wd_lin,wd_conv"
            ARGS_C=()
            # Add model/conv-specific args/params here
            if [[ "$conv" == "HornerConv" || "$conv" == "ClenshawConv" ]]; then
                PARLIST="$PARLIST,alpha"
            elif [[ "$conv" == "JacobiConv" ]]; then
                PARLIST="$PARLIST,alpha,beta"
            fi

            # Run hyperparameter search
            python run_param.py --dev $DEV --seed $SEED_P --param $PARLIST \
                --data $data --model $model --conv $conv \
                "${ARGS_P[@]}" "${ARGS_C[@]}"

            # Run repeatative with best hyperparameters
            python run_best.py --dev $DEV --seed $SEED_S --seed_param $SEED_P \
                --data $data --model $model --conv $conv \
                "${ARGS_S[@]}" "${ARGS_C[@]}"
        done
    done
done
