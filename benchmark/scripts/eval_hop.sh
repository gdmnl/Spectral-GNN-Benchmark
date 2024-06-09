# run_eval: acc vs hop, fullbatch, DecoupledFixed/Var
source scripts/ck_path.sh
DEV=${1:--1}
SEED_P=1
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
            "--seed_param" "$SEED_P"
            "--loglevel" "25"
            "--in_layers" "1"
            "--out_layers" "1"
            "--hidden" "128"
            "--epoch" "500"
            "--patience" "-1"
            "--eval_name" "acc_${PARKEY}"
            "--suffix" "acc_${PARKEY}"
        )

        # MLP
        python run_eval.py --data $data --model MLP \
            --param $PARKEY --"$PARKEY" $parval "${ARGS_S[@]}" \
            --theta_scheme ones

        # Linear
        python run_eval.py --data $data --model $model --conv AdjiConv \
            --param $PARKEY --"$PARKEY" $parval "${ARGS_S[@]}" \
            --theta_scheme ones --beta 1.0

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
        CONVS=("AdjiConv" "AdjConv" "HornerConv" "ChebConv" "ClenshawConv" "ChebIIConv" \
               "BernConv" "LegendreConv" "JacobiConv" "FavardConv" "OptBasisConv")

        for conv in ${CONVS[@]}; do
            python run_eval.py --data $data --model $model --conv $conv \
                --param $PARKEY --"$PARKEY" $parval "${ARGS_S[@]}"
        done

    done
done
