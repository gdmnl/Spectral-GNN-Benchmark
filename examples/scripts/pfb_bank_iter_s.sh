# run_param (critical params), fullbatch, iterative, Filter bank model+conv
source scripts/ck_path.sh
DEV=${1:--1}
SEED_P=1
SEED_S="20,21,22,23,24,25,26,27,28,29"
ARGS_P=(
    "--n_trials" "100"
    "--loglevel" "30"
    "--num_hops" "2"
    "--in_layers" "1"
    "--out_layers" "1"
    "--hidden" "128"
    "--epoch" "200"
    "--patience" "50"
    "--combine" "sum_weighted"
    "--suffix" "fb_bank"
)
ARGS_S=(
    "--seed_param" "$SEED_P"
    "--loglevel" "25"
    "--num_hops" "2"
    "--in_layers" "1"
    "--out_layers" "1"
    "--hidden" "128"
    "--epoch" "500"
    "--patience" "-1"
    "--combine" "sum_weighted"
    "--suffix" "fb_bank"
)

# DATAS=("cora" "citeseer" "pubmed" "flickr" "chameleon_filtered" "squirrel_filtered" "actor" "roman_empire")
# DATAS=("amazon_ratings" "minesweeper" "tolokers" "questions" "reddit" "penn94")
DATAS=("ogbn-arxiv" "arxiv-year" "genius" "twitch-gamer" "ogbn-mag" "pokec")

for data in ${DATAS[@]}; do
    PARLIST="normg,dp_lin,dp_conv,lr_lin,lr_conv,wd_lin,wd_conv"
    # MLP
    python run_param.py --dev $DEV --seed $SEED_P --param $PARLIST \
        --data $data --model MLP \
        "${ARGS_P[@]}"
    python run_best.py --dev $DEV --seed $SEED_S --seed_param $SEED_P \
        --data $data --model MLP \
        "${ARGS_S[@]}"

    # ACMGNN/FBGNN-I/II
    for alpha in 1 2; do
        for theta_scheme in "low-high-id" "low-high"; do
            python run_param.py --dev $DEV --seed $SEED_P --param $PARLIST \
                --data $data --model ACMGNN --conv ACMConv \
                --alpha $alpha --theta_scheme $theta_scheme \
                "${ARGS_P[@]}"
            python run_best.py --dev $DEV --seed $SEED_S --seed_param $SEED_P \
                --data $data --model ACMGNN --conv ACMConv \
                --alpha $alpha --theta_scheme $theta_scheme \
                "${ARGS_S[@]}"
        done
    done

done
