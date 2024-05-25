# -*- coding:utf-8 -*-
"""Run with a single data+model+conv entry using optuna for tuning hyperparameters.
Author: nyLiao
File Created: 2024-04-29
"""
from typing import Iterable
import optuna
from copy import deepcopy

from trainer import (
    SingleGraphLoader_Trial,
    ModelLoader_Trial,
    TrnBase_Trial)
from utils import (
    force_list_str,
    setup_seed,
    setup_argparse,
    setup_args,
    save_args,
    setup_logger,
    setup_logpath,
    clear_logger,
    ResLogger)


class TrnWrapper(object):
    metric_name = 'f1micro_val'

    def __init__(self, data_loader, model_loader, args, res_logger = None):
        self.data_loader = data_loader
        self.model_loader = model_loader
        self.args = args
        self.res_logger = res_logger or ResLogger()
        self.fmt_logger = {}

        self.data, self.model, self.trn_cls = None, None, None

    def _get_suggest(self, trial, key):
        list2str = lambda x: ','.join(map(str, x))
        nofmt = lambda x: x
        # >>>>>>>>>>
        theta_dct = {
            # "impulse": (lambda x, _: x[0], (self.args.num_hops,), {}),
            "ones":     (trial.suggest_float, (0.0, 1.0), {'step': 0.01}, lambda x: round(x, 2)),
            "impulse":  (trial.suggest_float, (0.0, 1.0), {'step': 0.01}, lambda x: round(x, 2)),
            "appr":     (trial.suggest_float, (0.0, 1.0), {'step': 0.01}, lambda x: round(x, 2)),
            "nappr":    (trial.suggest_float, (0.0, 1.0), {'step': 0.01}, lambda x: round(x, 2)),
            "mono":     (trial.suggest_float, (0.0, 1.0), {'step': 0.01}, lambda x: round(x, 2)),
            "hk":       (trial.suggest_float, (1e-2, 10), {'log': True}, lambda x: float(f'{x:.3e}')),
            "gaussian": (trial.suggest_float, (1e-2, 10), {'log': True}, lambda x: float(f'{x:.3e}')),
        }
        suggest_dct = {
            # critical
            'num_hops':     (trial.suggest_int, (2, 30), {'step': 2}, nofmt),
            'in_layers':    (trial.suggest_int, (1, 3), {}, nofmt),
            'out_layers':   (trial.suggest_int, (1, 3), {}, nofmt),
            'hidden':       (trial.suggest_categorical, ([16, 32, 64, 128, 256],), {}, nofmt),
            'combine':      (trial.suggest_categorical, (["sum", "sum_weighted", "cat"],), {}, nofmt),
            # secondary
            'theta_param': theta_dct.get(self.args.theta_scheme, None),
            'normg':        (trial.suggest_float, (0.0, 1.0), {'step': 0.05}, lambda x: round(x, 2)),
            'dp_lin':       (trial.suggest_float, (0.0, 1.0), {'step': 0.1}, lambda x: round(x, 2)),
            'dp_conv':      (trial.suggest_float, (0.0, 1.0), {'step': 0.1}, lambda x: round(x, 2)),
            'lr_lin':       (trial.suggest_float, (1e-5, 5e-1), {'log': True}, lambda x: float(f'{x:.3e}')),
            'lr_conv':      (trial.suggest_float, (1e-5, 5e-1), {'log': True}, lambda x: float(f'{x:.3e}')),
            'wd_lin':       (trial.suggest_float, (1e-7, 1e-3), {'log': True}, lambda x: float(f'{x:.3e}')),
            'wd_conv':      (trial.suggest_float, (1e-7, 1e-3), {'log': True}, lambda x: float(f'{x:.3e}')),
            'alpha':        (trial.suggest_float, (0.01, 1.0), {'step': 0.01}, lambda x: round(x, 2)),
            'beta':         (trial.suggest_float, (0.0, 1.0), {'step': 0.01}, lambda x: round(x, 2)),
        }

        # Model/conv-specific
        # if self.args.model in ['Iterative']:
        #     suggest_dct['in_layers'][1] = (1, 3)
        #     suggest_dct['out_layers'][1] = (1, 3)
        # <<<<<<<<<<

        if 'Compose' in self.args.model:
            convs = self.args.conv.split(',')
            if key == 'theta_param':
                schemes = self.args.theta_scheme.split(',')
                lst = []
                for i,c in enumerate(convs):
                    func, fargs, fkwargs, fmt = theta_dct.get(schemes[i], None)
                    lst.append(func(key+'-'+str(i), *fargs, **fkwargs))
                return lst, fmt
            elif key == 'beta':
                func, fargs, fkwargs, fmt = suggest_dct[key]
                beta_c = {
                    'AdjiConv':     [(0.0, 1.0), (0.0, 1.0)],   # FAGNN
                    'Adji2Conv':    [(1.0, 2.0), (0.0, 1.0)],   # G2CN
                    'AdjDiffConv':  [(0.0, 1.0), (-1.0, 0.0)],  # GNN-LF/HF
                }
                return list2str([func(key+'-'+str(i), *beta_i, **fkwargs) for i,beta_i in enumerate(beta_c[convs[0]])]), str
            else:
                func, fargs, fkwargs, fmt = suggest_dct[key]
                return func(key, *fargs, **fkwargs), fmt
        else:
            func, fargs, fkwargs, fmt = suggest_dct[key]
            return func(key, *fargs, **fkwargs), fmt

    def __call__(self, trial):
        args = deepcopy(self.args)
        args.quiet = True
        fmt_logger = {}
        for key in self.args.param:
            args.__dict__[key], fmt_logger[key] = self._get_suggest(trial, key)
        if args.in_layers == 0 and args.out_layers == 0:
            raise optuna.TrialPruned()

        if self.data is None:
            self.data = self.data_loader(args)
            self.model, trn_cls = self.model_loader(args)
            self.trn_cls = type('Trn_Trial', (trn_cls, TrnBase_Trial), {})
        self.data = self.data_loader.update(args, self.data)
        self.model = self.model_loader.update(args, self.model)
        res_logger = deepcopy(self.res_logger)
        for key in self.args.param:
            val = args.__dict__[key]
            if isinstance(val, Iterable) and not isinstance(val, str):
                vali = ','.join(map(str, val))
                fmt = str
            else:
                vali, fmt = val, fmt_logger[key]
            res_logger.concat([(key, vali, fmt)])
            self.fmt_logger[key] = fmt_logger[key]

        trn = self.trn_cls(
            model=self.model,
            data=self.data,
            args=args,
            res_logger=res_logger,)
        trn.trial = trial
        trn()

        res_logger.save()
        trial.set_user_attr("f1_test", res_logger._get(col='f1micro_test', row=0))
        return res_logger.data.loc[0, self.metric_name]


def main(args):
    # ========== Run configuration
    logger = setup_logger(args.logpath, level_console=args.loglevel, level_file=30, quiet=True)
    res_logger = ResLogger(args.logpath, prefix='param', quiet=args.quiet)
    res_logger.concat([('seed', args.seed),])

    # ========== Study configuration
    study_path = '-'.join(filter(None, ['optuna', args.suffix])) + '.db'
    study_path, _ = setup_logpath(folder_args=(study_path,))
    study = optuna.create_study(
        study_name=args.logid,
        storage=f'sqlite:///{str(study_path)}',
        direction='maximize',
        sampler=optuna.samplers.TPESampler(
            seed=args.seed),
        pruner=optuna.pruners.HyperbandPruner(
            min_resource=2,
            max_resource=args.epoch,
            reduction_factor=3),
        load_if_exists=True)
    optuna.logging.set_verbosity(args.loglevel)

    # ========== Init and run trainer
    args_run = deepcopy(args)
    trn = TrnWrapper(
        data_loader=SingleGraphLoader_Trial(args_run, res_logger),
        model_loader=ModelLoader_Trial(args_run, res_logger),
        args=args_run,
        res_logger=res_logger,)
    study.optimize(
        trn,
        n_trials=args.n_trials,
        gc_after_trial=True,
        show_progress_bar=True,)

    if 'Compose' in args.model:
        best_params = {}
        for k, v in study.best_params.items():
            if '-' in k:
                relk, idx = k.split('-')
                if relk not in best_params:
                    best_params[relk] = [None] * len(args.conv.split(','))
                best_params[relk][int(idx)] = trn.fmt_logger[relk](v)
            else:
                best_params[k] = trn.fmt_logger[k](v)
        for k, v in best_params.items():
            if isinstance(v, list):
                best_params[k] = ','.join(map(str, v))
    else:
        best_params = {k: trn.fmt_logger[k](v) for k, v in study.best_params.items()}
    save_args(args.logpath, best_params)
    clear_logger(logger)


if __name__ == '__main__':
    parser = setup_argparse()
    # Experiment-specific arguments
    parser.add_argument('--param', type=force_list_str, help='List of hyperparameters to search')
    parser.add_argument('--n_trials', type=int, default=5, help='Number of trials')
    args = setup_args(parser)

    seed_lst = args.seed.copy()
    for seed in seed_lst:
        args.seed = setup_seed(seed, args.cuda)
        args.flag = f'param-{args.seed}'
        args.logpath, args.logid = setup_logpath(
            folder_args=(args.data, args.model_repr, args.conv_repr, args.flag),
            quiet=args.quiet)

        main(args)
