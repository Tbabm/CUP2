# encoding=utf-8
"""
Two stage baseline of CUP2
"""
import json
import os
from functools import partial
from typing import Callable, Optional

import fire
import logging

import torch
from tqdm import tqdm
from multiprocessing import Process

from train import train_from_config, sanity_test_from_config
from common import set_reproducibility
from config import load_yaml_config
from dataset import load_dataset_from_file, Dataset
from eval import build_evaluator_from_config
from infer import CompositeInfer, build_inferer_from_config


CLF_CONFIG_FILE= "configs/OCD.yml"
CLF_LOG_DIR="BCD"
UPD_CONFIG_FILE= "configs/CUP.yml"
UPD_LOG_DIR="CUP"
COM_CONFIG_FILE= "configs/CUP2.yml"
COM_LOG_DIR="CUP2"


class TwoProcessProcedure:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        self.clf_process = None
        self.updater_process = None

    def _start_processes(self, clf_f: Callable, upd_f: Callable):
        self.clf_process = Process(target=clf_f)
        self.updater_process = Process(target=upd_f)
        self.logger.info("Start the classifier process")
        self.clf_process.start()
        self.logger.info("Start the updater process")
        self.updater_process.start()

        self.clf_process.join()
        self.updater_process.join()


class TwoStageProcedure:
    def __init__(self, clf_config_file: str, clf_log_dir: str, upd_config_file: str, upd_log_dir: str):
        self.clf_config_file = clf_config_file
        self.clf_log_dir = clf_log_dir
        self.upd_config_file = upd_config_file
        self.upd_log_dir = upd_log_dir


class TwoStageTrainer(TwoStageProcedure, TwoProcessProcedure):
    def __init__(self, clf_config_file: str, clf_log_dir: str, upd_config_file: str, upd_log_dir: str):
        TwoStageProcedure.__init__(self, clf_config_file, clf_log_dir, upd_config_file, upd_log_dir)
        TwoProcessProcedure.__init__(self)

    def train(self, gpu1: int = 1, gpu2: int = 2):
        clf_target = partial(train_func, self.clf_config_file, self.clf_log_dir, gpu1)
        upd_target = partial(train_func, self.upd_config_file, self.upd_log_dir, gpu2)
        self._start_processes(clf_target, upd_target)

    def sanity_test(self, gpu1: int = 1, gpu2: int = 2):
        # infer separately
        clf_target = partial(sanity_test_func, self.clf_config_file, self.clf_log_dir, gpu1)
        upd_target = partial(sanity_test_func, self.upd_config_file, self.upd_log_dir, gpu2)
        self._start_processes(clf_target, upd_target)


class TwoStageInfer(TwoStageProcedure):
    def __init__(self, clf_config_file: str, clf_log_dir: str, upd_config_file: str, upd_log_dir: str,
                 com_config_file: str, com_log_dir: str):
        super().__init__(clf_config_file, clf_log_dir, upd_config_file, upd_log_dir)
        self.com_config_file = com_config_file
        self.com_log_dir = com_log_dir

    def infer(self):
        clf_configs = load_yaml_config(self.clf_config_file)
        probs = torch.load(os.path.join(self.clf_log_dir, clf_configs['infer']['output_file']))

        upd_inferer = build_inferer_from_config(self.upd_config_file, self.upd_log_dir)
        upd_configs = load_yaml_config(self.upd_config_file)
        with open(os.path.join(self.upd_log_dir, upd_configs['infer']['output_file']), 'r') as f:
            hypos = json.load(f)

        com_configs = load_yaml_config(self.com_config_file)
        output_prefix = os.path.join(self.com_log_dir, com_configs['infer']['output_file'])
        if not os.path.exists(self.com_log_dir):
            os.makedirs(self.com_log_dir)
        torch.save(probs, CompositeInfer.get_probs_file(output_prefix))

        # infer according to the probs
        set_reproducibility(seed=com_configs['infer'].get('seed', 0))
        clf_arg_dict = load_yaml_config(self.clf_config_file)['infer']
        test_set = load_dataset_from_file(clf_arg_dict['test_set_file'])
        ts_hypos = []
        upd_inferer.pbar = False
        for prob, example in tqdm(zip(probs, test_set)):
            pred_label = prob.argmax(0).item()
            if pred_label != 1:
                ts_hypos.append([])
                continue
            # perform update
            hypo = upd_inferer.beam_search(Dataset([example]))[0]
            ts_hypos.append(hypo)
        output_file = CompositeInfer.get_hypos_file(output_prefix)
        with open(output_file, 'w') as f:
            json.dump(ts_hypos, f)

        return probs, hypos, ts_hypos


class TwoStageEvaluator(TwoStageProcedure):
    def __init__(self, clf_config_file: str, clf_log_dir: str, upd_config_file: str, upd_log_dir: str,
                 com_config_file: str, com_log_dir: str):
        super().__init__(clf_config_file, clf_log_dir, upd_config_file, upd_log_dir)
        self.com_config_file = com_config_file
        self.com_log_dir = com_log_dir

    def evaluate(self):
        clf_evaluator = build_evaluator_from_config(self.clf_config_file, self.clf_log_dir)
        clf_result = clf_evaluator.evaluate()
        print("Detector's result:\n{}".format(clf_result))

        upd_evaluator = build_evaluator_from_config(self.upd_config_file, self.upd_log_dir)
        upd_result = upd_evaluator.evaluate()
        print("Updater's result:\n{}".format(upd_result))

        # composite result
        com_evaluator = build_evaluator_from_config(self.com_config_file, self.com_log_dir)
        result = com_evaluator.evaluate()
        print("Composite's result: {}".format(result))
        return result


def train_func(config_file: str, log_dir: str, gpu_id: Optional[int], **kwargs):
    train_from_config(config_file, log_dir, gpu_id=gpu_id, **kwargs)


def sanity_test_func(config_file: str, log_dir: str, gpu_id: Optional[int], **kwargs):
    sanity_test_from_config(config_file, log_dir, gpu_id=gpu_id, **kwargs)


def train(clf_config=CLF_CONFIG_FILE,
          clf_log_dir=CLF_LOG_DIR,
          upd_config=UPD_CONFIG_FILE,
          upd_log_dir=UPD_LOG_DIR,
          gpu1=1,
          gpu2=2):
    ts_trainer = TwoStageTrainer(clf_config, clf_log_dir, upd_config, upd_log_dir)
    ts_trainer.train(gpu1, gpu2)


def sanity_test(clf_config=CLF_CONFIG_FILE,
                clf_log_dir=CLF_LOG_DIR,
                upd_config=UPD_CONFIG_FILE,
                upd_log_dir=UPD_LOG_DIR,
                gpu1=1,
                gpu2=2):
    ts_trainer = TwoStageTrainer(clf_config, clf_log_dir, upd_config, upd_log_dir)
    ts_trainer.sanity_test(gpu1, gpu2)


def infer(clf_config=CLF_CONFIG_FILE,
          clf_log_dir=CLF_LOG_DIR,
          upd_config=UPD_CONFIG_FILE,
          upd_log_dir=UPD_LOG_DIR,
          com_config=COM_CONFIG_FILE,
          com_log_dir=COM_LOG_DIR):
    ts_inferer = TwoStageInfer(clf_config, clf_log_dir, upd_config, upd_log_dir, com_config, com_log_dir)
    ts_inferer.infer()


def eval(clf_config=CLF_CONFIG_FILE,
         clf_log_dir=CLF_LOG_DIR,
         upd_config=UPD_CONFIG_FILE,
         upd_log_dir=UPD_LOG_DIR,
         com_config=COM_CONFIG_FILE,
         com_log_dir=COM_LOG_DIR):
    ts_evaluator = TwoStageEvaluator(clf_config, clf_log_dir, upd_config, upd_log_dir, com_config, com_log_dir)
    return ts_evaluator.evaluate()


if __name__ == '__main__':
    fire.Fire({
        'train': train,
        'sanity_test': sanity_test,
        'infer': infer,
        'eval': eval
    })
