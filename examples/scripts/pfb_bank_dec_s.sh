# run_param (critical params), fullbatch, decoupled, Filter bank model+conv
source scripts/ck_path.sh
DEV=${1:--1}
SEED_P=1
SEED_S="20,21,22,23,24,25,26,27,28,29"
ARGS_P=(
    "--n_trials" "100"
    "--loglevel" "30"
    "--num_hops" "10"
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
    "--num_hops" "10"
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
    # AdaGNN
    for conv in "LapiConv"; do
        python run_param.py --dev $DEV --seed $SEED_P --param $PARLIST \
            --data $data --model AdaGNN --conv $conv \
            "${ARGS_P[@]}"
        python run_best.py --dev $DEV --seed $SEED_S --seed_param $SEED_P \
            --data $data --model AdaGNN --conv $conv \
            "${ARGS_S[@]}"
    done

    # FiGURe
    python run_param.py --dev $DEV --seed $SEED_P --param $PARLIST \
        --data $data --model DecoupledVarCompose --conv AdjConv,ChebConv,BernConv \
        "${ARGS_P[@]}"
    python run_best.py --dev $DEV --seed $SEED_S --seed_param $SEED_P \
        --data $data --model DecoupledVarCompose --conv AdjConv,ChebConv,BernConv \
        "${ARGS_S[@]}"

    PARLIST="$PARLIST,beta"
    # FAGNN
    python run_param.py --dev $DEV --seed $SEED_P --param $PARLIST \
        --data $data --model DecoupledFixedCompose --conv AdjiConv,AdjiConv \
        --theta_scheme ones,ones --theta_param 1,1 --alpha 1.0,-1.0 \
        "${ARGS_P[@]}"
    python run_best.py --dev $DEV --seed $SEED_S --seed_param $SEED_P \
        --data $data --model DecoupledFixedCompose --conv AdjiConv,AdjiConv \
        --theta_scheme ones,ones --theta_param 1,1 --alpha 1.0,-1.0 \
        "${ARGS_S[@]}"

    PARLIST="$PARLIST,theta_param"
    # G2CN
    python run_param.py --dev $DEV --seed $SEED_P --param $PARLIST \
        --data $data --model DecoupledFixedCompose --conv Adji2Conv,Adji2Conv \
        --theta_scheme gaussian,gaussian --alpha="-1.0,-1,0" \
        "${ARGS_P[@]}"
    python run_best.py --dev $DEV --seed $SEED_S --seed_param $SEED_P \
        --data $data --model DecoupledFixedCompose --conv Adji2Conv,Adji2Conv \
        --theta_scheme gaussian,gaussian --alpha="-1.0,-1,0" \
        "${ARGS_S[@]}"

    # GNN-LF/HF
    python run_param.py --dev $DEV --seed $SEED_P --param $PARLIST \
        --data $data --model DecoupledFixedCompose --conv AdjDiffConv,AdjDiffConv \
        --theta_scheme appr,appr --alpha 1.0,1.0 \
        "${ARGS_P[@]}"
    python run_best.py --dev $DEV --seed $SEED_S --seed_param $SEED_P \
        --data $data --model DecoupledFixedCompose --conv AdjDiffConv,AdjDiffConv \
        --theta_scheme appr,appr --alpha 1.0,1.0 \
        "${ARGS_S[@]}"
done
