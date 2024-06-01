# run_eval, fullbatch, DecoupledFixed/Var
source scripts/ck_path.sh
DEV=${1:--1}
SEED_P=1
SEED_S="70,71,72"

DATAS=("cora" "citeseer")
# DATAS=("cora" "citeseer" "pubmed" "flickr" "chameleon_filtered" "squirrel_filtered" "actor" "roman_empire")
# DATAS=("amazon_ratings" "minesweeper" "tolokers" "questions" "reddit" "penn94")
# DATAS=("ogbn-arxiv" "arxiv-year" "genius" "twitch-gamer" "ogbn-mag" "pokec")
PARKEY="num_hops"
PARVALS=(2 4 6 8 10 12 14 16 18 20)

# ==========
for data in ${DATAS[@]}; do
    for parval in ${PARVALS[@]}; do
        ARGS_S=(
            "--eval_name" "dec_${PARKEY}"
            "--seed_param" "$SEED_P"
            "--loglevel" "25"
            "--in_layers" "1"
            "--out_layers" "1"
            "--hidden" "128"
            "--epoch" "500"
            "--patience" "-1"
            "--suffix" "efb_fix"
        )
        model=DecoupledFixed
        conv=AdjConv
        SCHEMES=("impulse" "mono" "appr" "hk" "gaussian")

        # MLP
        python run_eval.py --dev $DEV --seed $SEED_S --param $PARKEY --"$PARKEY" $parval \
            --data $data --model MLP --theta_scheme ones \
            "${ARGS_S[@]}"

        # Linear
        python run_eval.py --dev $DEV --seed $SEED_S --param $PARKEY --"$PARKEY" $parval \
            --data $data --model $model --conv AdjiConv --theta_scheme ones \
            --beta 1.0 "${ARGS_S[@]}"

        for scheme in ${SCHEMES[@]}; do
            ARGS_C=()
            python run_eval.py --dev $DEV --seed $SEED_S --param $PARKEY --"$PARKEY" $parval \
                --data $data --model $model --conv $conv --theta_scheme $scheme \
                "${ARGS_S[@]}" "${ARGS_C[@]}"
        done

        # ==========
        ARGS_S=(
            "--eval_name" "dec_${PARKEY}"
            "--seed_param" "$SEED_P"
            "--loglevel" "25"
            "--in_layers" "1"
            "--out_layers" "1"
            "--hidden" "128"
            "--epoch" "500"
            "--patience" "-1"
            "--theta_scheme" "ones"
            "--theta_param" "1.0"
            "--suffix" "efb_var"
        )
        model=DecoupledVar
        CONVS=("AdjiConv" "AdjConv" "HornerConv" "ChebConv" "ClenshawConv" "ChebIIConv" "BernConv" "LegendreConv" "JacobiConv" "FavardConv" "OptBasisConv")

        for conv in ${CONVS[@]}; do
            python run_eval.py --dev $DEV --seed $SEED_S --param $PARKEY --"$PARKEY" $parval \
                --data $data --model $model --conv $conv \
                "${ARGS_S[@]}" "${ARGS_C[@]}"
        done

    done
done
