# -*- coding:utf-8 -*-
"""Run with a single data+model+conv entry using optuna for tuning hyperparameters.
Author: nyLiao
File Created: 2024-04-29
"""
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

        self.data, self.model, self.trn_cls = None, None, None

    def _loader_get(self, args):
        self.data = self.data_loader(args)
        self.model, trn_cls = self.model_loader(args)
        self.trn_cls = type('Trn_Trial', (trn_cls, TrnBase_Trial), {})

    def _get_suggest(self, trial, key):
        # >>>>>>>>>>
        theta_dct = {
            "appr":  (trial.suggest_float, (0.0, 1.0), {}),
            "nappr": (trial.suggest_float, (0.0, 1.0), {}),
            "mono":  (trial.suggest_float, (0.0, 1.0), {}),
            "hk":    (trial.suggest_float, (1e-2, 10), {'log': True}),
            "impulse": (lambda x, _: x[0], (self.args.num_hops,), {}),
        }
        suggest_dct = {
            # critical
            'num_hops':     (trial.suggest_int, (2, 30), {'step': 2}),
            'in_layers':    (trial.suggest_int, (1, 3), {}),
            'out_layers':   (trial.suggest_int, (1, 3), {}),
            'hidden':       (trial.suggest_categorical, ([16, 32, 64, 128, 256],), {}),
            # secondary
            'theta_param': theta_dct.get(self.args.theta_scheme, None),
            'normg':        (trial.suggest_float, (0.0, 1.0), {'step': 0.05}),
            'dp_lin':       (trial.suggest_float, (0.0, 1.0), {'step': 0.1}),
            'dp_conv':      (trial.suggest_float, (0.0, 1.0), {'step': 0.1}),
            'lr_lin':       (trial.suggest_float, (1e-5, 5e-1), {'log': True}),
            'lr_conv':      (trial.suggest_float, (1e-5, 5e-1), {'log': True}),
            'wd_lin':       (trial.suggest_float, (1e-7, 1e-3), {'log': True}),
            'wd_conv':      (trial.suggest_float, (1e-7, 1e-3), {'log': True}),
        }

        # Model/conv-specific
        # if self.args.model in ['Iterative']:
        #     suggest_dct['in_layers'][1] = (1, 3)
        #     suggest_dct['out_layers'][1] = (1, 3)
        # <<<<<<<<<<

        func, fargs, fkwargs = suggest_dct[key]
        return func(key, *fargs, **fkwargs)

    def __call__(self, trial):
        args = deepcopy(self.args)
        res_logger = deepcopy(self.res_logger)
        args.quiet = True
        for key in self.args.param:
            args.__dict__[key] = self._get_suggest(trial, key)
            res_logger.concat([(
                key, args.__dict__[key], type(args.__dict__[key]))])
        if args.in_layers == 0 and args.out_layers == 0:
            raise optuna.TrialPruned()

        if self.data is None:
            self._loader_get(args)
        self.data = self.data_loader.update(args, self.data)
        self.model = self.model_loader.update(args, self.model)
        trn = self.trn_cls(
            model=self.model,
            data=self.data,
            args=args,
            res_logger=res_logger,)
        trn.trial = trial
        res_logger = trn()

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

    best_params = study.best_params
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
            folder_args=(args.data, args.model, args.conv_repr, args.flag),
            quiet=args.quiet)

        main(args)
