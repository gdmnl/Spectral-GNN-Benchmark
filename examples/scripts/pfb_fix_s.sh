# run_param+run_best, fullbatch, DecoupledFixed-theta, small-scale dataset
source scripts/ck_path.sh
DEV=${1:--1}
SEED_P=1
# SEED_S="20,21,22,23,24,25,26,27,28,29"
SEED_S="20,21,22"
ARGS_P=(
    "--n_trials" "100"
    "--loglevel" "30"
    "--num_hops" "10"
    "--in_layers" "1"
    "--out_layers" "1"
    "--hidden" "128"
    "--epoch" "200"
    "--patience" "50"
    "--suffix" "fb_fix"
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
    "--suffix" "fb_fix"
)

# DATAS=("cora" "citeseer" "pubmed" "flickr" "chameleon_filtered" "squirrel_filtered" "actor" "roman_empire")
# DATAS=("amazon_ratings" "minesweeper" "tolokers" "questions" "reddit" "penn94")
# DATAS=("ogbn-arxiv" "arxiv-year" "genius" "twitch-gamer" "ogbn-mag" "pokec")
DATAS=("twitch-gamer" "ogbn-mag" "pokec")
MODELS=("DecoupledFixed")
CONVS=AdjConv
SCHEMES=("impulse" "mono" "appr" "hk" "gaussian")
PARLIST="theta_param,normg,dp_lin,dp_conv,lr_lin,wd_lin"

for data in ${DATAS[@]}; do
    for model in ${MODELS[@]}; do
        for scheme in ${SCHEMES[@]}; do
            # Add model/conv-specific args/params here
            if [ "$scheme" = "gaussian" ]; then
                ARGS_C=("--alpha" "1.0")
            else
                ARGS_C=()
            fi

            # Run hyperparameter search
            python run_param.py --dev $DEV --seed $SEED_P --param $PARLIST \
                --data $data --model $model --conv $CONVS --theta_scheme $scheme \
                "${ARGS_P[@]}" "${ARGS_C[@]}"

            # Run repeatative with best hyperparameters
            python run_best.py --dev $DEV --seed $SEED_S --seed_param $SEED_P \
                --data $data --model $model --conv $CONVS --theta_scheme $scheme \
                "${ARGS_S[@]}" "${ARGS_C[@]}"
        done
    done
done
