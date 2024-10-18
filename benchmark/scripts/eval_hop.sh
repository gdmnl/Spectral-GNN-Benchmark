# eval: acc vs hop, fullbatch, DecoupledFixed/Var
DEV=${1:--1}
SEED_S="70,71,72"

DATAS=("cora" "actor" "citeseer" "pubmed" "minesweeper" "chameleon_filtered" "squirrel_filtered" "actor" "roman_empire" "questions" "tolokers" "amazon_ratings" "flickr")
PARKEY="num_hops"
PARVALS=(2 4 6 8 10 12 14 16 18 20)

# ==========
for data in ${DATAS[@]}; do
    for parval in ${PARVALS[@]}; do
        ARGS_S=(
            "--dev" "$DEV"
            "--seed" "$SEED_S"
            "--loglevel" "25"
            "--in_layers" "1"
            "--out_layers" "1"
            "--hidden_channels" "128"
            "--epoch" "500"
            "--patience" "-1"
            "--suffix" "eval_acc_${PARKEY}"
            "-quiet"
        )

        # MLP
        python run_single.py --data $data --model MLP --conv PrecomputedFixed \
            --"$PARKEY" $parval --param $PARKEY "${ARGS_S[@]}" \
            --theta_scheme ones

        # Linear
        model=DecoupledFixed
        python run_single.py --data $data --model $model --conv AdjiConv \
            --"$PARKEY" $parval --param $PARKEY "${ARGS_S[@]}" \
            --theta_scheme ones --beta 1.0

        conv=AdjConv
        SCHEMES=("impulse" "mono" "appr" "hk" "gaussian")
        for scheme in ${SCHEMES[@]}; do
            python run_single.py --data $data --model $model --conv $conv \
                --"$PARKEY" $parval --param $PARKEY "${ARGS_S[@]}" \
                --theta_scheme $scheme
        done

        # ==========
        ARGS_S=("${ARGS_S[@]}"
            "--theta_scheme" "ones"
            "--theta_param" "1.0"
        )
        model=DecoupledVar
        CONVS=("AdjiConv" "AdjConv" "HornerConv" "ChebConv" "ClenshawConv" "ChebIIConv" \
               "BernConv" "LegendreConv" "JacobiConv" "FavardConv" "OptBasisConv")

        for conv in ${CONVS[@]}; do
            python run_single.py --data $data --model $model --conv $conv \
                --"$PARKEY" $parval --param $PARKEY "${ARGS_S[@]}"
        done

    done
done
