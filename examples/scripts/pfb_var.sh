# run_param+run_Sultiple, fullbatch, Iterative/DecoupledVar
source scripts/ck_path.sh
DEV=${1:--1}
ARGS_P=(
    "--loglevel" "25"
    "--num_hops" "20"
    "--hidden" "256"
    "--epoch" "20"
    "--patience" "50"
    "--theta_scheme" "ones"
    "--theta_param" "1.0"
)
ARGS_S=(
    "--loglevel" "10"
    "--num_hops" "20"
    "--hidden" "256"
    "--epoch" "200"
    "--patience" "-1"
    "--theta_scheme" "ones"
    "--theta_param" "1.0"
)
PARAMS=(
    "--normg" "0.0,0.25,0.5,0.75,1.0"
    "--dp" "0.0,0.1,0.3,0.5,0.7"
    "--lr" "0.1,0.05,0.01,0.001"
    "--wd" "0.0,1e-2,5e-3,1e-3,1e-4,1e-5,1e-6"
)
SEED_P=0
SEED_S="20,21,22"

DATAS=("cora" "citeseer" "pubmed")
MODELS=("Iterative" "DecoupledVar")
CONVS=("AdjConv" "ChebConv")

for data in ${DATAS[@]}; do
    for model in ${MODELS[@]}; do
        for conv in ${CONVS[@]}; do
            # Add model/conv-specific args/params here
            if [ "$conv" = "AdjConv" ]; then
                ARGS_P+=("--alpha" "0.0")
                ARGS_S+=("--alpha" "0.0")
            elif [ "$conv" = "ClenshawConv" ]; then
                PARAMS+=("--alpha" "0.1,0.3,0.5,0.7,0.9")
            fi

            # Search hyperparameters
            PARLIST=""
            for ((i = 0; i < ${#PARAMS[@]}; i += 2)); do
                PARLIST+="$(echo "${PARAMS[i]}" | sed 's/--//'),"
            done
            PARLIST="${PARLIST%,}"
            python run_param.py --dev $DEV --seed $SEED_P --param $PARLIST \
                --data $data --model $model --conv $conv \
                "${ARGS_P[@]}" "${PARAMS[@]}"

            # Run repeatative with best hyperparameters
            PARFILE=../log/${model}/${data}/${conv}/param.sh
            source $PARFILE
            python run_single.py --dev $DEV --seed $SEED_S \
                --data $data --model $model --conv $conv \
                "${ARGS_S[@]}" "${PARAM[@]}"
        done
    done
done
