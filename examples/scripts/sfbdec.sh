# Single Fullbatch + Decoupled
# GPRGNN
python run_single.py --data cora --model PostMLP --conv VarSumAdj --theta appr --alpha 0.1 -K 10 --lr 0.01 --wd 0.0005
python run_single.py --data citeseer --model PostMLP --conv VarSumAdj --theta appr --alpha 0.1 -K 10 --lr 0.01 --wd 0.0005
# ChebBase
python run_single.py --data cora --model PostMLP --conv ChebBase --alpha 0 -K 10
