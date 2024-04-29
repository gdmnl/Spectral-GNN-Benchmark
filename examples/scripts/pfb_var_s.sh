# run_param+run_single, fullbatch, Iterative/DecoupledVar, small-scale
source scripts/ck_path.sh
DEV=${1:--1}
SEED_P=0
# SEED_S="20,21,22,23,24,25,26,27,28,29"
SEED_S="20,21,22,23,24"
ARGS_P=(
    "--n_trials" "500"
    "--loglevel" "25"
    "--num_hops" "10"
    "--hidden" "64"
    "--epoch" "200"
    "--patience" "50"
    "--theta_scheme" "ones"
    "--theta_param" "1.0"
)
ARGS_S=(
    "--seed_param" "$SEED_P"
    "--loglevel" "10"
    "--num_hops" "10"
    "--hidden" "64"
    "--epoch" "500"
    "--patience" "-1"
    "--theta_scheme" "ones"
    "--theta_param" "1.0"
)

DATAS=("cora" "citeseer" "pubmed")
MODELS=("Iterative" "DecoupledVar")
CONVS=("AdjConv" "ChebConv")

for data in ${DATAS[@]}; do
    for model in ${MODELS[@]}; do
        for conv in ${CONVS[@]}; do
            PARLIST="normg,dp,lr,wd"
            # Add model/conv-specific args/params here
            if [ "$conv" = "AdjConv" ]; then
                ARGS_C=("--alpha" "0.0")
            elif [ "$conv" = "ClenshawConv" ]; then
                PARLIST="$PARLIST,alpha"
            else
                ARGS_C=()
            fi

            python run_param.py --dev $DEV --seed $SEED_P --param $PARLIST \
                --data $data --model $model --conv $conv \
                "${ARGS_P[@]}" "${ARGS_C[@]}"

            # Run repeatative with best hyperparameters
            python run_best.py --dev $DEV --seed $SEED_S \
                --data $data --model $model --conv $conv \
                "${ARGS_S[@]}" "${ARGS_C[@]}"
        done
    done
done
