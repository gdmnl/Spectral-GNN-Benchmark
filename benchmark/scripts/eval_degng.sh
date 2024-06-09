# run_eval: degree vs normg, fullbatch, DecoupledFixed/Var
DEV=${1:--1}
SEED_P=1
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
            "--seed_param" "$SEED_P"
            "--loglevel" "25"
            "--num_hops" "10"
            "--in_layers" "1"
            "--out_layers" "1"
            "--hidden" "128"
            "--epoch" "500"
            "--patience" "-1"
            "--eval_name" "deg_${PARKEY}"
            "--suffix" "deg_${PARKEY}"
            "--test_deg"
        )
        # MLP
        python run_eval.py --data $data --model MLP "${ARGS_S[@]}" \
            --param $PARKEY --"$PARKEY" $parval \
            --theta_scheme ones

        # Linear
        python run_eval.py --data $data --model $model --conv AdjiConv \
            --param $PARKEY --"$PARKEY" $parval "${ARGS_S[@]}" \
            --theta_scheme ones -beta 1.0

        model=DecoupledFixed
        conv=AdjConv
        SCHEMES=("impulse" "mono" "appr" "hk" "gaussian")
        for scheme in ${SCHEMES[@]}; do
            python run_eval.py --data $data --model $model --conv $conv \
                --param $PARKEY --"$PARKEY" $parval "${ARGS_S[@]}" \
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
            python run_eval.py --data $data --model $model --conv $conv \
                --param $PARKEY --"$PARKEY" $parval "${ARGS_S[@]}"
        done

    done
done
