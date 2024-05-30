# run_param (critical params), fullbatch, Filter bank model+conv
source scripts/ck_path.sh
DEV=${1:--1}
SEED_P=0
ARGS_P=(
    "--n_trials" "50"
    "--loglevel" "30"
    "--epoch" "200"
    "--patience" "50"
    "--suffix" "fb_bank"
)
DATAS=("cora" "citeseer" "chameleon_filtered" "actor")
PARLIST="num_hops,in_layers,out_layers,hidden,lr_lin,lr_conv"

for data in ${DATAS[@]}; do
    # AdaGNN
    for conv in "LapiConv"; do
        python run_param.py --dev $DEV --seed $SEED_P --param $PARLIST \
            --data $data --model AdaGNN --conv $conv --theta_scheme normal --theta_param 0,1e-7 \
            "${ARGS_P[@]}"
    done

    # ACMGNN/FBGNN-I/II
    for alpha in 1 2; do
        for theta_scheme in "low-high-id" "low-high"; do
            python run_param.py --dev $DEV --seed $SEED_P --param $PARLIST \
                --data $data --model ACMGNN --conv ACMConv \
                --alpha $alpha --theta_scheme $theta_scheme \
                "${ARGS_P[@]}"
        done
    done

    PARLIST="$PARLIST,combine"
    # FiGURe
    python run_param.py --dev $DEV --seed $SEED_P --param $PARLIST \
        --data $data --model DecoupledVarCompose --conv AdjConv,ChebConv,BernConv \
        "${ARGS_P[@]}"

    PARLIST="$PARLIST,beta"
    # FAGNN
    python run_param.py --dev $DEV --seed $SEED_P --param $PARLIST \
        --data $data --model DecoupledFixedCompose --conv AdjiConv,AdjiConv \
        --theta_scheme ones,ones --theta_param 1,1 --alpha 1.0,-1.0 \
        "${ARGS_P[@]}"

    PARLIST="$PARLIST,theta_param"
    # G2CN
    python run_param.py --dev $DEV --seed $SEED_P --param $PARLIST \
        --data $data --model DecoupledFixedCompose --conv Adji2Conv,Adji2Conv \
        --theta_scheme gaussian,gaussian --alpha="-1.0,-1,0" \
        "${ARGS_P[@]}"

    # GNN-LF/HF
    python run_param.py --dev $DEV --seed $SEED_P --param $PARLIST \
        --data $data --model DecoupledFixedCompose --conv AdjDiffConv,AdjDiffConv \
        --theta_scheme appr,appr --alpha 1.0,1.0 \
        "${ARGS_P[@]}"

done
