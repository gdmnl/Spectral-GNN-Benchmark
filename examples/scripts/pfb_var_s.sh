# run_param+run_single, fullbatch, Iterative/DecoupledVar, small-scale
source scripts/ck_path.sh
DEV=${1:--1}
ARGS_P=(
    "--loglevel" "25"
    "--num_hops" "10"
    "--hidden" "64"
    "--epoch" "200"
    "--patience" "50"
    "--theta_scheme" "ones"
    "--theta_param" "1.0"
)
ARGS_S=(
    "--loglevel" "10"
    "--num_hops" "10"
    "--hidden" "64"
    "--epoch" "500"
    "--patience" "-1"
    "--theta_scheme" "ones"
    "--theta_param" "1.0"
)
PARAMS_P=(
    "--normg" "0.0,0.5,1.0"
    "--dp" "0.25,0.5,0.75"
    "--lr" "0.1,0.01,0.001"
    "--wd" "5e-4,5e-5,5e-6"
)
SEED_P=0
# SEED_S="20,21,22,23,24,25,26,27,28,29"
SEED_S="20,21,22"

DATAS=("cora" "citeseer" "pubmed")
MODELS=("Iterative" "DecoupledVar")
CONVS=("AdjConv" "ChebConv")

for data in ${DATAS[@]}; do
    for model in ${MODELS[@]}; do
        for conv in ${CONVS[@]}; do
            # Add model/conv-specific args/params here
            if [ "$conv" = "AdjConv" ]; then
                ARGS_C=("--alpha" "0.0")
            elif [ "$conv" = "ClenshawConv" ]; then
                PARAMS_C=("--alpha" "0.1,0.3,0.5,0.7,0.9")
            else
                ARGS_C=()
                PARAMS_C=()
            fi

            # Search hyperparameters
            PARLIST=""
            for ((i = 0; i < ${#PARAMS_P[@]}; i += 2)); do
                PARLIST+="$(echo "${PARAMS_P[i]}" | sed 's/--//'),"
            done
            for ((i = 0; i < ${#PARAMS_C[@]}; i += 2)); do
                PARLIST+="$(echo "${PARAMS_C[i]}" | sed 's/--//'),"
            done
            PARLIST="${PARLIST%,}"
            python run_param.py --dev $DEV --seed $SEED_P --param $PARLIST \
                --data $data --model $model --conv $conv \
                "${ARGS_P[@]}" "${ARGS_C[@]}" "${PARAMS_P[@]}" "${PARAMS_C[@]}"

            # Run repeatative with best hyperparameters
            PARFILE=../log/${model}/${data}/${conv}/param.sh
            source $PARFILE
            python run_single.py --dev $DEV --seed $SEED_S \
                --data $data --model $model --conv $conv \
                "${ARGS_S[@]}" "${ARGS_C[@]}" "${PARAM[@]}"
        done
    done
done
