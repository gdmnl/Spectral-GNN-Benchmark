# -*- coding:utf-8 -*-
"""Run grid search to obtain the optimal hyperparameters.
"""
import logging
from itertools import product
import os
import json
import argparse
import pandas as pd
from trainer import DatasetLoader, ModelLoader
from pathlib import Path
from utils import (
    setup_argparse,
    setup_args,
    save_args,
    setup_logger,
    setup_logpath,
    clear_logger,
    ResLogger)

# Hyperparameters search space
search_spaces = {
    'lr': [1.0e-2, 1.0e-3, 1.0e-4],
    'wd': [1.0e-3, 1.0e-4, 1.0e-5],
    'alpha': [0.2, 0.4, 0.8],
    'K':[2, 4, 8, 10],
    'dp':[0.2, 0.5, 0.7],
}

# Method, model configurations and their specific hyperparameters
method_model_configs = {
    'PostMLP': {
        'VarSumAdj': ('lr','wd', 'K','dp'),
        'ChebBase': ('lr','wd', 'K','dp'),
        'ChebConv2':('lr','wd', 'K','dp'), 
        'BernConv':('lr','wd', 'K','dp'),
    },
    'IterGNN': {
        'FixLinSumAdj': ('lr','wd', 'K','dp'), 
    },
    'PreDecMLP': {
        'FixSumAdj': ('lr','wd', 'K','dp'), 
    },
    'PreMLP': {
        'Adagnn': ('lr','wd', 'K','dp'), 
    },
}
best_accuracy = 0

def update_results_csv(result_path, model_name, dataset_name, new_result):
    # Load the existing CSV file into a DataFrame or create a new one if it doesn't exist
    try:
        df = pd.read_csv(result_path, index_col=0)
    except FileNotFoundError:
        # If the file does not exist, create an empty DataFrame with model_name as the index
        df = pd.DataFrame(columns=[dataset_name])
        df.index.name = 'Model'

    # Check if the dataset_name column exists, if not, add it
    if dataset_name not in df.columns:
        df[dataset_name] = pd.NA  # Initialize the column with NA values

    # Ensure the model_name row exists
    if model_name not in df.index:
        # If the model doesn't exist, append a new row with NA values
        new_row = pd.DataFrame(index=[model_name], columns=df.columns)
        df = pd.concat([df, new_row])

    # Update the specific entry with the new result
    df.at[model_name, dataset_name] = new_result

    # Save the updated DataFrame back to CSV
    df.to_csv(result_path)

    print(f"Updated CSV file at {result_path} with new results for model '{model_name}' and dataset '{dataset_name}'.")

def search_hyperparameters(args, method, model):
    setattr(args, 'grid_search_flag', True)
    setattr(args, 'degree_flag', False)
    # Ensure method and model are valid
    if method in method_model_configs and model in method_model_configs[method]:
        # Get hyperparameters for the specific method and model
        hyperparams = method_model_configs[method][model]
        # Generate combinations of hyperparameter values
        combinations = product(*(search_spaces[param] for param in hyperparams))
        
        for combo in combinations:
            param_set = dict(zip(hyperparams, combo))
            # Here, integrate your model training and validation logic
            print(f"Model: {method}, conv: {model}, Params: {param_set}")
            for key, value in param_set.items():
                setattr(args, key, value)
            # for param, default in param_set.items():
            #     parser.add_argument(f'--{param}', type=type(default), default=default, help=f'Default: {default}', conflict_handler='resolve')
                
            print(f"Model: {args.model}, conv: {args.conv},changed lr: {args.lr}, changed alpha: {args.alpha}")

            main(args)

    else:
        print("Invalid method or model configuration.")

def get_degree_accuracy(args, method, model):
    setattr(args, 'grid_search_flag', False)
    setattr(args, 'degree_flag', True)
    args.logpath = setup_logpath(
        folder_args=(args.data, args.model, args.flag),
        quiet=args.quiet)
    # Ensure method and model are valid
    if method in method_model_configs and model in method_model_configs[method]:
        optimal_path = os.path.join(args.logpath, args.conv+ "_optimal/config.json")
        hyperparams = method_model_configs[method][model]
        with open(optimal_path, 'r') as file:
            config_dict = json.load(file)

            # Convert the dictionary to a namespace object
            for param in hyperparams:
                setattr(args, param, int(config_dict[param])) if isinstance(config_dict[param], int) else setattr(args, param, float(config_dict[param]))
            # args = argparse.Namespace(**config_dict)
            print('args:',args.data)
            main(args)


    else:
        print("Invalid method or model configuration.")

def main(args):
    # ========== Run configuration
    args.logpath = setup_logpath(
        folder_args=(args.data, args.model, args.flag),
        quiet=args.quiet)
    logger = setup_logger(args.logpath, level=args.loglevel, quiet=args.quiet)
    res_logger = ResLogger(args.logpath.parent.parent, quiet=args.quiet)
    res_logger.concat([('seed', args.seed),])

    # ========== Load data
    data_loader = DatasetLoader(args, res_logger)
    dataset = data_loader(args)

    # ========== Load model
    model_loader = ModelLoader(args, res_logger)
    model, trn = model_loader(args)
    res_logger.suffix = trn.name

    # ========== Run trainer
    trn = trn(
        model=model,
        dataset=dataset,
        args=args,
        res_logger=res_logger,)
    del model, dataset
    trn()

    logger.info(f"[args]: {args}")
    logger.log(logging.LRES, f"[res]: {res_logger}")
    current_accuracy = float(res_logger.get_str("f1micro_test", 0).split(":")[1])
    print("result:", current_accuracy)
    f1micro_high = float(res_logger.get_str("f1micro_high", 0).split(":")[1])
    f1micro_low = float(res_logger.get_str("f1micro_low", 0).split(":")[1])
    print("f1micro_high:", f1micro_high, " f1micro_low", f1micro_low)

    
    res_logger.save()
    clear_logger(logger)
    global best_accuracy
    table_path = Path('../log/sum_table.csv')
    degree_table_path = Path('../log/degree_table.csv')
    if current_accuracy>best_accuracy and args.grid_search_flag:
        setattr(args, 'optimal_accuracy', current_accuracy)
        optimal_path = os.path.join(args.logpath, args.conv+ "_optimal")
        if not os.path.exists(optimal_path):
            os.makedirs(optimal_path)
        save_args(Path(optimal_path), args)
        best_accuracy = current_accuracy
        update_results_csv(table_path, args.model + "-" + args.conv+ "-" +args.theta, args.data, best_accuracy)

    if args.degree_flag:
        #Just output the degree accuracy using the optimal config
        update_results_csv(degree_table_path, args.model + "-" + args.conv+ "-" +args.theta, args.data+'-'+'high', f1micro_high) 
        update_results_csv(degree_table_path, args.model + "-" + args.conv+ "-" +args.theta, args.data+'-'+'low', f1micro_low)

if __name__ == '__main__':
    parser = setup_argparse()
    # Experiment-specific arguments
    # parser.add_argument()
    args = setup_args(parser)
 
    # Example usage for grid search
    # search_hyperparameters(args, args.model, args.conv)

    # Get the degree accuracy when you have an optimal config
    get_degree_accuracy(args, args.model, args.conv)






    