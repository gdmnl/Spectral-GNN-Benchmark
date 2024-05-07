# run_param+run_best, fullbatch, Iterative/DecoupledVar, small-scale dataset
source scripts/ck_path.sh
DEV=${1:--1}
SEED_P=1
# SEED_S="20,21,22,23,24,25,26,27,28,29"
SEED_S="20,21,22"
ARGS_P=(
    "--n_trials" "300"
    "--loglevel" "30"
    "--num_hops" "10"
    "--in_layers" "1"
    "--out_layers" "1"
    "--hidden" "128"
    "--epoch" "200"
    "--patience" "50"
    "--theta_scheme" "ones"
    "--theta_param" "1.0"
    "--suffix" "fb_var"
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
    "--theta_scheme" "ones"
    "--theta_param" "1.0"
    "--suffix" "fb_var"
)

# DATAS=("cora" "citeseer" "pubmed")
DATAS=("cora" "citeseer" "pubmed" "chameleon_filtered" "squirrel_filtered")
MODELS=("Iterative" "DecoupledVar")
CONVS=("AdjConv" "ChebConv" "ChebConv2" "BernConv" "ClenhawConv" "HornerConv")

for data in ${DATAS[@]}; do
    for model in ${MODELS[@]}; do
        for conv in ${CONVS[@]}; do
            PARLIST="normg,dp_lin,dp_conv,lr_lin,lr_conv,wd_lin,wd_conv"
            # Add model/conv-specific args/params here
            if [ "$conv" = "AdjConv" ]; then
                ARGS_C=("--alpha" "0.0")
            elif [ "$conv" = "ClenshawConv" ]; then
                PARLIST="$PARLIST,alpha"
            else
                ARGS_C=()
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
