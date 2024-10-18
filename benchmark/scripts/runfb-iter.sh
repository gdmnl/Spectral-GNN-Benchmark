# search param + best, fullbatch+Iterative
DEV=${1:-0}
SEED_P="1,2"
SEED_S="20,21,22,23,24,25,26,27,28,29"
ARGS_ALL=(
    "--dev" "$DEV"
    "--num_hops" "8"
    "--in_layers" "1"
    "--out_layers" "1"
    "--hidden_channels" "128"
    "--suffix" "fbit"
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

DATAS=("cora" "citeseer" "pubmed" "flickr" "chameleon_filtered" "squirrel_filtered" "actor" "roman_empire" \
       "amazon_ratings" "minesweeper" "tolokers" "questions" "penn94" "ogbn-arxiv" "arxiv-year" "genius" \
       "reddit" "twitch-gamer" "ogbn-mag" "pokec")

for data in ${DATAS[@]}; do
# ========== fix
model=IterativeFixed
PARLIST="dropout_lin,lr_lin,wd_lin"
    # MLP
    # Run hyperparameter search
    python run_param.py  --data $data --model MLP --conv $model --param $PARLIST "${ARGS_P[@]}"
    # Run repeatative with best hyperparameters
    python run_single.py --data $data --model MLP --conv $model "${ARGS_S[@]}"

PARLIST="normg,dropout_conv,$PARLIST"
    # Linear
    python run_param.py  --data $data --model $model --conv AdjiConv --param $PARLIST "${ARGS_P[@]}" \
        --beta 1.0
    python run_single.py --data $data --model $model --conv AdjiConv "${ARGS_S[@]}" \
        --beta 1.0

PARLIST="beta,$PARLIST"
    # PPR
    python run_param.py  --data $data --model $model --conv AdjResConv --param $PARLIST "${ARGS_P[@]}"
    python run_single.py --data $data --model $model --conv AdjResConv "${ARGS_S[@]}"

: '
# ========== PyG
PARLIST="dropout_lin,lr_lin,wd_lin"
    MODELS=("GCN" "GraphSAGE")
    for model in ${MODELS[@]}; do
        python run_param.py  --data $data --model $model --conv $model --param $PARLIST "${ARGS_P[@]}"
        python run_single.py --data $data --model $model --conv $model "${ARGS_S[@]}"
    done
'

# ========== bank
ARGS_P=("${ARGS_P[@]}"
    "--combine" "sum_weighted")
ARGS_S=("${ARGS_S[@]}"
    "--combine" "sum_weighted")
PARLIST="normg,dropout_lin,dropout_conv,lr_lin,lr_conv,wd_lin,wd_conv"
    # FiGURe
    python run_param.py  --data $data --model IterativeCompose --conv AdjConv,ChebConv,BernConv --param $PARLIST "${ARGS_P[@]}"
    python run_single.py --data $data --model IterativeCompose --conv AdjConv,ChebConv,BernConv "${ARGS_S[@]}"

    # ACMGNN/FBGNN-I/II
    for alpha in 1 2; do
        for theta_scheme in "low-high-id" "low-high"; do
            python run_param.py  --data $data --model ACMGNN --conv ACMConv --param $PARLIST "${ARGS_P[@]}" \
                --alpha $alpha --theta_scheme $theta_scheme
            python run_single.py --data $data --model ACMGNN --conv ACMConv "${ARGS_S[@]}" \
                --alpha $alpha --theta_scheme $theta_scheme
        done
    done

# ========== var
ARGS_P=("${ARGS_P[@]}"
    "--theta_scheme" "ones"
    "--theta_param" "1.0")
ARGS_S=("${ARGS_S[@]}"
    "--theta_scheme" "ones"
    "--theta_param" "1.0")
    model="Iterative"
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
