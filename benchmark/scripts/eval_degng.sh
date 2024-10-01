# eval: degree vs normg, fullbatch, DecoupledFixed/Var
DEV=${1:--1}
SEED_S="60,61,62"

DATAS=("cora" "citeseer" "pubmed" "flickr" "chameleon_filtered" "squirrel_filtered" "actor" "roman_empire")
PARKEY="normg"
PARVALS=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)

# ==========
for data in ${DATAS[@]}; do
    for parval in ${PARVALS[@]}; do
        ARGS_S=(
            "--dev" "$DEV"
            "--seed" "$SEED_S"
            "--loglevel" "25"
            "--num_hops" "10"
            "--in_layers" "1"
            "--out_layers" "1"
            "--hidden" "128"
            "--epoch" "500"
            "--patience" "-1"
            "--suffix" "eval_deg_${PARKEY}"
            "-quiet" "--test_deg"
        )
        # MLP
        python run_single.py --data $data --model MLP \
            --"$PARKEY" $parval --param $PARKEY "${ARGS_S[@]}" \
            --theta_scheme ones

        # Linear
        python run_single.py --data $data --model $model --conv AdjiConv \
            --"$PARKEY" $parval --param $PARKEY "${ARGS_S[@]}" \
            --theta_scheme ones -beta 1.0

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
        CONVS=("AdjiConv" "AdjConv" "HornerConv" "ChebConv" "ClenshawConv" "ChebIIConv" "BernConv" "LegendreConv" "JacobiConv" "FavardConv" "OptBasisConv")

        for conv in ${CONVS[@]}; do
            python run_single.py --data $data --model $model --conv $conv \
                --"$PARKEY" $parval --param $PARKEY "${ARGS_S[@]}"
        done

    done
done
