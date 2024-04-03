# Define the list of datasets
# datasets=("Cora" "CiteSeer" "PubMed")
datasets=("texas" "squirrel" "chameleon")

# Loop through each dataset and run the Python script with the parameters
for graph in "${datasets[@]}"; do
    echo "Running grid search for dataset: $graph"
    #fixed-linear
    python run_grid_search.py --model IterGNN --conv FixLinSumAdj --data $graph --theta khop
    #fixed-Impulse 
    python run_grid_search.py --model PreDecMLP --conv FixSumAdj --data $graph --theta khop
    #fixed-Monomial
    python run_grid_search.py --model PreDecMLP --conv FixSumAdj --data $graph --theta mono
    #fixed-ppr
    python run_grid_search.py --model PostMLP --conv FixSumAdj --data $graph --theta appr
    #fixed-heart kernel
    python run_grid_search.py --model PreDecMLP --conv FixSumAdj --data $graph --theta hk
    #fixed-guassian
    python run_grid_search.py --model PreDecMLP --conv FixSumAdj --data $graph --theta gaussian
    
    #var-Linear-unfinished
    # python run_grid_search.py --model IterGNN --conv VarSumAdj --data $graph --theta khop
    #var-Monomial
    python run_grid_search.py --model PostMLP --conv VarSumAdj --data $graph --theta appr
    #var-horner-unfinished
    # python run_grid_search.py --model IterGNN --conv VarSumAdj --data $graph --theta appr
    #var-Chebyshev
    python run_grid_search.py --model PostMLP --conv ChebBase --data $graph
    #var-Chebyshev2
    python run_grid_search.py --model PostMLP --conv ChebConv2 --data $graph
    #var-Bernstein
    python run_grid_search.py --model PostMLP --conv BernConv --data $graph

    #bank-linear
    python run_grid_search.py --model PreMLP --conv Adagnn --data $graph

done

echo "All experiments completed."