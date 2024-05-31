# run_param+run_best, minibatch, PrecomputedFixed-theta, small-scale dataset
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
    "--normf" "0"
    "--suffix" "mb_fix"
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
    "--normf" "0"
    "--suffix" "mb_fix"
)

# DATAS=("cora" "citeseer" "pubmed" "flickr" "chameleon_filtered" "squirrel_filtered" "actor" "roman_empire")
DATAS=("amazon_ratings" "minesweeper" "tolokers" "questions" "reddit" "penn94" "ogbn-arxiv" "arxiv-year" "genius" "twitch-gamer" "ogbn-mag" "pokec")
model=PrecomputedFixed
conv=AdjConv
SCHEMES=("impulse" "mono" "appr" "hk" "gaussian")

for data in ${DATAS[@]}; do
    PARLIST="normg,dp_lin,dp_conv,lr_lin,wd_lin"
    # MLP
    python run_param.py --dev $DEV --seed $SEED_P --param $PARLIST \
        --data $data --model MLP --theta_scheme impulse \
        "${ARGS_P[@]}"
    python run_best.py --dev $DEV --seed $SEED_S --seed_param $SEED_P \
        --data $data --model MLP --theta_scheme impulse \
        "${ARGS_S[@]}"

    # Linear
    python run_param.py --dev $DEV --seed $SEED_P --param $PARLIST \
        --data $data --model $model --conv AdjSkipConv --theta_scheme ones \
        --beta 1.0 "${ARGS_P[@]}"
    python run_best.py --dev $DEV --seed $SEED_S --seed_param $SEED_P \
        --data $data --model $model --conv AdjSkipConv --theta_scheme ones \
        --beta 1.0 "${ARGS_S[@]}"

    PARLIST="theta_param,$PARLIST"
    for scheme in ${SCHEMES[@]}; do
        ARGS_C=()
        # Run hyperparameter search
        python run_param.py --dev $DEV --seed $SEED_P --param $PARLIST \
            --data $data --model $model --conv $conv --theta_scheme $scheme \
            "${ARGS_P[@]}" "${ARGS_C[@]}"

        # Run repeatative with best hyperparameters
        python run_best.py --dev $DEV --seed $SEED_S --seed_param $SEED_P \
            --data $data --model $model --conv $conv --theta_scheme $scheme \
            "${ARGS_S[@]}" "${ARGS_C[@]}"
    done
done
