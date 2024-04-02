#!/bin/bash

for graph in "Cora" "CiteSeer" "PubMed" "Texas" "Chameleon" "Squirrel"; do
    for conv in "ChebBase" "ChebConv2" "BernConv"; do
        for K in 2 10; do
            for lr in 0.01 0.05; do
                for dp in 0.2 0.5 0.7; do
                    for wd in 0.0005 0.05; do
                        python run_single.py --data $graph --model PostMLP --conv $conv --alpha 0 -K $K --epoch 10 --hidden 32 --patience 200 --lr $lr --wd $wd --dp $dp
                        done
                    done
                done
            done
        done
    done
done