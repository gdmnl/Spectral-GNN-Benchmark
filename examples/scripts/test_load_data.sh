for graph in "Texas" "Chameleon" "Squirrel"; do
    python run_single.py --data $graph --model PostMLP --conv VarSumAdj --theta appr --alpha 0.5 -K 20 --epoch 200
done