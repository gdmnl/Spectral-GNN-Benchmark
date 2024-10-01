# eval: acc vs hop, fullbatch, DecoupledFixed/Var
DEV=${1:--1}
SEED_S="70,71,72"

DATAS=("cora" "citeseer" "pubmed" "flickr" "chameleon_filtered" "squirrel_filtered" "actor" "roman_empire")
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
            "--hidden" "128"
            "--epoch" "500"
            "--patience" "-1"
            "--suffix" "eval_acc_${PARKEY}"
            "-quiet"
        )

        # MLP
        python run_single.py --data $data --model MLP \
            --"$PARKEY" $parval --param $PARKEY "${ARGS_S[@]}" \
            --theta_scheme ones

        # Linear
        python run_single.py --data $data --model $model --conv AdjiConv \
            --"$PARKEY" $parval --param $PARKEY "${ARGS_S[@]}" \
            --theta_scheme ones --beta 1.0

        model=DecoupledFixed
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
