# ChebConv2, BernConv, ClenShaw, Horner
# python run_single.py --data cora --model Iterative --conv HornerConv --alpha 0.0
# python run_single.py --data cora --model DecoupledVar --conv HornerConv --alpha 0.0
python run_single.py --data cora --model DecoupledVar --conv JacobiConv --alpha 0.0 --epoch 2000