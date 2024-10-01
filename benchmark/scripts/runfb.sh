# search param + best, fullbatch+Decoupled
DEV=${1:-0}
SEED_P=1
SEED_S="20,21,22,23,24,25,26,27,28,29"
ARGS_ALL=(
    "--dev" "$DEV"
    "--num_hops" "10"
    "--in_layers" "1"
    "--out_layers" "1"
    "--hidden" "128"
)
# run_param args
ARGS_P=(${ARGS_ALL[@]}
    "--seed" "$SEED_P"
    "--n_trials" "50"
    "--loglevel" "30"
    "--epoch" "200"
    "--patience" "50"
    "--suffix" "fb"
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

DATAS=("cora" "citeseer" "pubmed" "flickr" "chameleon_filtered" "squirrel_filtered" "actor" "roman_empire" \
       "amazon_ratings" "minesweeper" "tolokers" "questions" "reddit" "penn94" "ogbn-arxiv" "arxiv-year" "genius" "twitch-gamer" \
       "ogbn-mag" "pokec")

for data in ${DATAS[@]}; do
# ========== fix
PARLIST="dp_lin,lr_lin,wd_lin"
    # MLP
    # Run hyperparameter search
    python run_param.py  --data $data --model MLP --param $PARLIST "${ARGS_P[@]}" \
        --theta_scheme ones
    # Run repeatative with best hyperparameters
    python run_single.py --data $data --model MLP "${ARGS_S[@]}" \
        --theta_scheme ones

PARLIST="normg,dp_conv,$PARLIST"
    # Linear
    python run_param.py  --data $data --model DecoupledFixed --conv AdjiConv --param $PARLIST "${ARGS_P[@]}" \
        --theta_scheme ones --beta 1.0
    python run_single.py --data $data --model DecoupledFixed --conv AdjiConv "${ARGS_S[@]}" \
        --theta_scheme ones --beta 1.0

PARLIST="theta_param,$PARLIST"
    model=DecoupledFixed
    conv=AdjConv
    SCHEMES=("impulse" "mono" "appr" "hk" "gaussian")
    for scheme in ${SCHEMES[@]}; do
        python run_param.py  --data $data --model $model --conv $conv --param $PARLIST "${ARGS_P[@]}" \
            --theta_scheme $scheme
        python run_single.py --data $data --model $model --conv $conv "${ARGS_S[@]}" \
            --theta_scheme $scheme
    done

# ========== bank
ARGS_P=("${ARGS_P[@]}"
    "--combine" "sum_weighted")
ARGS_S=("${ARGS_S[@]}"
    "--combine" "sum_weighted")
PARLIST="normg,dp_lin,dp_conv,lr_lin,lr_conv,wd_lin,wd_conv"
    # AdaGNN
    for conv in "LapiConv"; do
        python run_param.py  --data $data --model AdaGNN --conv $conv --param $PARLIST "${ARGS_P[@]}" \
            --theta_scheme normal --theta_param 0,1e-7
        python run_single.py --data $data --model AdaGNN --conv $conv "${ARGS_S[@]}" \
            --theta_scheme normal --theta_param 0,1e-7
    done

    # FiGURe
    python run_param.py  --data $data --model DecoupledVarCompose --conv AdjConv,ChebConv,BernConv --param $PARLIST "${ARGS_P[@]}"
    python run_single.py --data $data --model DecoupledVarCompose --conv AdjConv,ChebConv,BernConv "${ARGS_S[@]}"

PARLIST="$PARLIST,beta"
    # FAGNN
    python run_param.py  --data $data --model DecoupledFixedCompose --conv AdjiConv,AdjiConv --param $PARLIST "${ARGS_P[@]}" \
        --theta_scheme ones,ones --theta_param 1,1 --alpha 1.0,-1.0
    python run_single.py --data $data --model DecoupledFixedCompose --conv AdjiConv,AdjiConv "${ARGS_S[@]}" \
        --theta_scheme ones,ones --theta_param 1,1 --alpha 1.0,-1.0

PARLIST="$PARLIST,theta_param"
    # G2CN
    python run_param.py  --data $data --model DecoupledFixedCompose --conv Adji2Conv,Adji2Conv --param $PARLIST "${ARGS_P[@]}" \
        --theta_scheme gaussian,gaussian --alpha="-1.0,-1.0"
    python run_single.py --data $data --model DecoupledFixedCompose --conv Adji2Conv,Adji2Conv "${ARGS_S[@]}" \
        --theta_scheme gaussian,gaussian --alpha="-1.0,-1.0"

    # GNN-LF/HF
    python run_param.py  --data $data --model DecoupledFixedCompose --conv AdjDiffConv,AdjDiffConv --param $PARLIST "${ARGS_P[@]}" \
        --theta_scheme appr,appr --alpha 1.0,1.0
    python run_single.py --data $data --model DecoupledFixedCompose --conv AdjDiffConv,AdjDiffConv "${ARGS_S[@]}" \
        --theta_scheme appr,appr --alpha 1.0,1.0

# ========== var
ARGS_P=("${ARGS_P[@]}"
    "--theta_scheme" "ones"
    "--theta_param" "1.0")
ARGS_S=("${ARGS_S[@]}"
    "--theta_scheme" "ones"
    "--theta_param" "1.0")
    model="DecoupledVar"
    CONVS=("AdjiConv" "AdjConv" "HornerConv" "ChebConv" "ClenshawConv" "ChebIIConv" \
           "BernConv" "LegendreConv" "JacobiConv" "FavardConv" "OptBasisConv")
    for conv in ${CONVS[@]}; do
        PARLIST="normg,dp_lin,dp_conv,lr_lin,lr_conv,wd_lin,wd_conv"
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
