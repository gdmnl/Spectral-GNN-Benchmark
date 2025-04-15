# search param + best, fullbatch+Decoupled
DEV=${1:-0}
SEED_P="1,2"
SEED_S="20,21,22"
ARGS_ALL=(
    "--dev" "$DEV"
    "--num_hops" "10"
    "--in_layers" "1"
    "--out_layers" "0"
    "--hidden_channels" "128"
    "--suffix" "lpfbdc"
)
# run_param args
ARGS_P=(${ARGS_ALL[@]}
    "--seed" "$SEED_P"
    "--n_trials" "100"
    "--loglevel" "30"
    "--epoch" "200"
    "--patience" "50"
)
# run_single args
ARGS_S=(${ARGS_ALL[@]}
    "--seed" "$SEED_S"
    "--loglevel" "25"
    "--epoch" "500"
    "--patience" "-1"
    "--param"
)

DATAS=("ogbl-collab")

for data in ${DATAS[@]}; do
# ========== fix
model=DecoupledFixedLP
PARLIST="dropout_lin,lr_lin,wd_lin"
PARLIST="normg,dropout_conv,$PARLIST"
    # Linear
    python run_param.py  --data $data --model $model --conv AdjiConv --param $PARLIST "${ARGS_P[@]}" \
        --theta_scheme ones --beta 1.0
    python run_single.py --data $data --model $model --conv AdjiConv "${ARGS_S[@]}" \
        --theta_scheme ones --beta 1.0

PARLIST="theta_param,$PARLIST"
    conv=AdjConv
    SCHEMES=("impulse" "mono" "appr" "hk" "gaussian")
    for scheme in ${SCHEMES[@]}; do
        python run_param.py  --data $data --model $model --conv $conv --param $PARLIST "${ARGS_P[@]}" \
            --theta_scheme $scheme
        python run_single.py --data $data --model $model --conv $conv "${ARGS_S[@]}" \
            --theta_scheme $scheme
    done

# ========== var
ARGS_P=("${ARGS_P[@]}"
    "--theta_scheme" "ones"
    "--theta_param" "1.0")
ARGS_S=("${ARGS_S[@]}"
    "--theta_scheme" "ones"
    "--theta_param" "1.0")
    model="DecoupledVarLP"
    CONVS=("AdjiConv" "AdjConv" "HornerConv" "ChebConv" "ClenshawConv" "ChebIIConv" \
           "BernConv" "LegendreConv" "JacobiConv" "FavardConv" "OptBasisConv")
    for conv in ${CONVS[@]}; do
        PARLIST="normg,dropout_lin,dropout_conv,lr_lin,lr_conv,wd_lin,wd_conv"
        # Add model/conv-specific args/params here
        if [[ "$conv" == "HornerConv" || "$conv" == "ClenshawConv" ]]; then
            PARLIST="$PARLIST,alpha"
        elif [[ "$conv" == "JacobiConv" ]]; then
            PARLIST="$PARLIST,alpha,beta"
        fi

        python run_param.py  --data $data --model $model --conv $conv --param $PARLIST "${ARGS_P[@]}"
        python run_single.py --data $data --model $model --conv $conv "${ARGS_S[@]}"
    done

done
