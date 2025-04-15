# -*- coding:utf-8 -*-
"""Run with a single data+model+conv entry using optuna for tuning hyperparameters.
Author: nyLiao
File Created: 2024-04-29
"""
from typing import Iterable
import optuna
import uuid
from copy import deepcopy

from pyg_spectral.nn import get_model_regi, get_conv_subregi
from pyg_spectral.nn.parse_args import compose_param
from pyg_spectral.utils import CallableDict

from trainer import (
    SingleGraphLoader_Trial,
    ModelLoader_Trial,
    TrnFullbatch, TrnFullbatch_Trial,
    TrnLPFullbatch,
    TrnMinibatch, TrnMinibatch_Trial)
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
    def __init__(self, data_loader, model_loader, args, res_logger = None):
        self.data_loader = data_loader
        self.model_loader = model_loader
        self.args = args
        self.res_logger = res_logger or ResLogger()
        self.fmt_logger = {}
        self.metric = None

        self.data, self.model, self.trn = None, None, None
        self.trn_cls = {
            TrnFullbatch: TrnFullbatch_Trial,
            TrnLPFullbatch: type('TrnLPFullbatch_Trial', (TrnLPFullbatch, TrnFullbatch_Trial), {}),
            TrnMinibatch: TrnMinibatch_Trial,
        }[self.model_loader.get_trn(args)]

    def _get_suggest(self, trial, key):
        r"""Get the suggested value and format of the hyperparameter.

        Args:
            trial (optuna.Trial): The trial object.
            key (str): The hyperparameter key in :obj:`args.param`.
        Returns:
            val (Any | list[Any]): The suggested value.
            fmt (Callable): The format of the suggested value.
        """
        def parse_param(val):
            r"""From :class:`ParamTuple` to suggest trial value and format.
            Args:
                val (ParamTuple | list[ParamTuple]): registry entry.
            Returns:
                val (Any | list[Any]): The suggested value.
                fmt (Callable): The format of the suggested value
            """
            if isinstance(val, list):
                fmt = val[0][-1]
                val = [getattr(trial, 'suggest_'+func)(key+'-'+str(i), *fargs, **fkwargs) for i, (func, fargs, fkwargs, _) in enumerate(val)]
                return val, fmt
            func, fargs, fkwargs, fmt = val
            return getattr(trial, 'suggest_'+func)(key, *fargs, **fkwargs), fmt

        # Alias param for compose models
        if (self.args.model in compose_param and
            self.model_loader.conv_repr in compose_param[self.args.model] and
            key in compose_param[self.args.model][self.model_loader.conv_repr]):
            return parse_param(compose_param[self.args.model][self.model_loader.conv_repr](key, self.args))

        # Param of trainer and model level
        single_param = SingleGraphLoader_Trial.param | ModelLoader_Trial.param | self.trn_cls.param
        single_param = CallableDict(single_param)
        single_param |= get_model_regi(self.args.model, 'param')
        if key in single_param:
            return parse_param(single_param(key, self.args))

        # Param of conv level
        return parse_param(get_conv_subregi(self.args.conv, 'param', key, self.args))

    def __call__(self, trial):
        args = deepcopy(self.args)
        args.quiet = True
        fmt_logger = {}
        for key in self.args.param:
            args.__dict__[key], fmt_logger[key] = self._get_suggest(trial, key)
        if args.in_layers == 0 and args.out_layers == 0:
            raise optuna.TrialPruned()

        if self.data is None:
            self.data = self.data_loader.get(args)
            self.model, _ = self.model_loader.get(args)

            for key in SingleGraphLoader_Trial.args_out + ModelLoader_Trial.args_out:
                self.args.__dict__[key] = args.__dict__[key]
            self.metric = args.metric
        else:
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

        if self.trn is None:
            self.trn = self.trn_cls(
                model=self.model,
                data=self.data,
                args=args,
                res_logger=res_logger,)
        else:
            self.trn.update(
                model=self.model,
                data=self.data,
                args=args,
                res_logger=res_logger,)
        self.trn.trial = trial
        self.trn.run()

        res_logger.save()
        trial.set_user_attr("s_test", res_logger._get(col=self.metric+'_test', row=0))
        return res_logger.data.loc[0, self.metric+'_hyperval']


def main(args):
    # ========== Run configuration
    logger = setup_logger(args.logpath, level_console=args.loglevel, level_file=30, quiet=True)
    res_logger = ResLogger(args.logpath, prefix='param', quiet=args.quiet)
    res_logger.concat([('seed', args.seed),])

    # ========== Study configuration
    study_path = '-'.join(filter(None, ['optuna', args.suffix])) + '.db'
    study_path = setup_logpath(folder_args=(study_path,))
    study_name = '/'.join(str(args.logpath).split('/')[-4:-1])
    study_id = '/'.join((study_name, str(args.seed)))
    study_id = uuid.uuid5(uuid.NAMESPACE_DNS, study_id).int % 2**32
    study = optuna.create_study(
        study_name=study_name,
        storage=optuna.storages.RDBStorage(
            url=f'sqlite:///{str(study_path)}',
            heartbeat_interval=3600),
        direction='maximize',
        sampler=optuna.samplers.TPESampler(
            n_startup_trials=min(8, args.n_trials // 5),
            n_ei_candidates=36,
            seed=study_id,
            multivariate=True,
            group=True,
            warn_independent_sampling=False),
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
        gc_after_trial=False,
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
    axes = optuna.visualization.matplotlib.plot_parallel_coordinate(
        study, params=study.best_params.keys())
    axes.get_figure().savefig(args.logpath.joinpath('parallel_coordinate.png'))
    clear_logger(logger)


if __name__ == '__main__':
    parser = setup_argparse()
    # Experiment-specific arguments
    parser.add_argument('--param', type=force_list_str, help='List of hyperparameters to search')
    parser.add_argument('--n_trials', type=int, default=5, help='Number of trials')
    args = setup_args(parser)

    seed_lst = args.seed.copy()
    args.n_trials = args.n_trials // len(seed_lst)
    for seed in seed_lst:
        args.seed = setup_seed(seed, args.cuda)
        args.flag = 'param'
        args.logpath = setup_logpath(
            folder_args=(args.data, *ModelLoader_Trial.get_name(args), args.flag),
            quiet=args.quiet)

        main(args)
