# search param + best, minibatch+Precomputed
DEV=${1:-0}
SEED_P=1

run_mb() {
    local lseed_s=$1
    shift
    local lbatch=$1
    shift
    local ldatas=("$@")

SEED_S=$lseed_s
ARGS_ALL=(
    "--dev" "$DEV"
    "--num_hops" "10"
    "--in_layers"  "0"
    "--out_layers" "2"
    "--batch" "$lbatch"
    "--hidden" "128"
)
# run_param args
ARGS_P=(${ARGS_ALL[@]}
    "--seed" "$SEED_P"
    "--n_trials" "50"
    "--loglevel" "30"
    "--epoch" "200"
    "--patience" "50"
    "--suffix" "mb"
)
# run_single args
ARGS_S=(${ARGS_ALL[@]}
    "--seed" "$SEED_S"
    "--loglevel" "25"
    "--epoch" "500"
    "--patience" "-1"
    "--suffix" "summary"
    "--param"
)

for data in ${ldatas[@]}; do
# ========== fix
ARGS_P=(${ARGS_P[@]} "--normf" "0")
ARGS_S=(${ARGS_S[@]} "--normf" "0")
PARLIST="dp_lin,lr_lin,wd_lin"
    # MLP
    # Run hyperparameter search
    python run_param.py --data $data --model MLP --param $PARLIST "${ARGS_P[@]}" \
        --theta_scheme impulse
    # Run repeatative with best hyperparameters
    python run_single.py  --data $data --model MLP "${ARGS_S[@]}" \
        --theta_scheme impulse

PARLIST="normg,dp_conv,$PARLIST"
    # Linear
    python run_param.py --data $data --model $model --conv AdjSkipConv --param $PARLIST "${ARGS_P[@]}" \
        --theta_scheme ones --beta 1.0
    python run_single.py  --data $data --model $model --conv AdjSkipConv "${ARGS_S[@]}" \
        --theta_scheme ones --beta 1.0

PARLIST="theta_param,$PARLIST"
    model=PrecomputedFixed
    conv=AdjConv
    SCHEMES=("impulse" "mono" "appr" "hk" "gaussian")
    for scheme in ${SCHEMES[@]}; do
        python run_param.py --data $data --model $model --conv $conv --param $PARLIST "${ARGS_P[@]}" \
            --theta_scheme $scheme
        python run_single.py  --data $data --model $model --conv $conv "${ARGS_S[@]}" \
            --theta_scheme $scheme
    done

# ========== bank
ARGS_P=("${ARGS_P[@]:0:${#ARGS_P[@]}-2}"
    "--combine" "sum_weighted" "--normf")
ARGS_S=("${ARGS_S[@]:0:${#ARGS_S[@]}-2}"
    "--combine" "sum_weighted" "--normf")
PARLIST="normg,dp_lin,dp_conv,lr_lin,lr_conv,wd_lin,wd_conv"
    # FiGURe
    python run_param.py --data $data --model PrecomputedVarCompose --conv AdjConv,ChebConv,BernConv --param $PARLIST "${ARGS_P[@]}"
    python run_single.py  --data $data --model PrecomputedVarCompose --conv AdjConv,ChebConv,BernConv "${ARGS_S[@]}"

PARLIST="$PARLIST,beta"
    # FAGNN
    python run_param.py --data $data --model PrecomputedFixedCompose --conv AdjSkipConv,AdjSkipConv --param $PARLIST "${ARGS_P[@]}" \
        --theta_scheme ones,ones --theta_param 1,1 --alpha 1.0,-1.0
    python run_single.py  --data $data --model PrecomputedFixedCompose --conv AdjSkipConv,AdjSkipConv "${ARGS_S[@]}" \
        --theta_scheme ones,ones --theta_param 1,1 --alpha 1.0,-1.0

PARLIST="$PARLIST,theta_param"
    # G2CN
    python run_param.py --data $data --model PrecomputedFixedCompose --conv AdjSkip2Conv,AdjSkip2Conv --param $PARLIST "${ARGS_P[@]}" \
        --theta_scheme gaussian,gaussian --alpha="-1.0,-1.0"
    python run_single.py  --data $data --model PrecomputedFixedCompose --conv AdjSkip2Conv,AdjSkip2Conv "${ARGS_S[@]}" \
        --theta_scheme gaussian,gaussian --alpha="-1.0,-1.0"

    # GNN-LF/HF
    python run_param.py --data $data --model PrecomputedFixedCompose --conv AdjDiffConv,AdjDiffConv --param $PARLIST "${ARGS_P[@]}" \
        --theta_scheme appr,appr --alpha 1.0,1.0
    python run_single.py  --data $data --model PrecomputedFixedCompose --conv AdjDiffConv,AdjDiffConv "${ARGS_S[@]}" \
        --theta_scheme appr,appr --alpha 1.0,1.0

# ========== var
ARGS_P=("${ARGS_P[@]}"
    "--theta_scheme" "ones"
    "--theta_param" "1.0")
ARGS_S=("${ARGS_S[@]}"
    "--theta_scheme" "ones"
    "--theta_param" "1.0")
    model="PrecomputedVar"
    CONVS=("AdjSkipConv" "AdjConv" "HornerConv" "ChebConv" "ClenshawConv" "ChebIIConv" \
           "BernConv" "LegendreConv" "JacobiConv" "OptBasisConv")
    for conv in ${CONVS[@]}; do
        PARLIST="normg,dp_lin,dp_conv,lr_lin,lr_conv,wd_lin,wd_conv"
        # Add model/conv-specific args/params here
        if [[ "$conv" == "HornerConv" || "$conv" == "ClenshawConv" ]]; then
            PARLIST="$PARLIST,alpha"
        elif [[ "$conv" == "JacobiConv" ]]; then
            PARLIST="$PARLIST,alpha,beta"
        fi

        python run_param.py --data $data --model $model --conv $conv --param $PARLIST "${ARGS_P[@]}"
        python run_single.py  --data $data --model $model --conv $conv "${ARGS_S[@]}"
    done

done
}

SEED_S="20,21,22,23,24,25,26,27,28,29"
BATCH=4096
DATAS=("cora" "citeseer" "pubmed" "flickr" "chameleon_filtered" "squirrel_filtered" "actor" "roman_empire" \
       "amazon_ratings" "minesweeper" "tolokers" "questions" "reddit" "penn94" "ogbn-arxiv" "arxiv-year" "genius" "twitch-gamer")
run_mb $SEED_S $BATCH ${DATAS[@]}

SEED_S="20,21,22,23,24"
BATCH=200000
DATAS=("ogbn-mag" "pokec" "ogbn-products" "snap-patents" "wiki")
run_mb $SEED_S $BATCH ${DATAS[@]}
