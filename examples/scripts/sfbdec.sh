# Fullbatch + Decoupled
# GPRGNN
python run_single.py --data Cora --model PostMLP --conv VarSumAdj --theta appr --alpha 0.5 -K 20
# ChebBase
python run_single.py --data Cora --model PostMLP --conv ChebBase --alpha 0 -K 10
