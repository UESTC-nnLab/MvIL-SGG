#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from genericpath import isfile
import json
import os
if __name__ == '__main__':
    os.sys.path.append('./src')
from src.model.model import MMGNet
from src.utils.config import Config
from utils import util
import torch
import argparse

def main():
    config = load_config()
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
    util.set_random_seed(config.SEED)

    if config.VERBOSE:
        print(config)
    
    model = MMGNet(config)

    save_path = os.path.join(config.PATH,'config', model.model_name, model.exp)
    os.makedirs(save_path, exist_ok=True)
    save_path = os.path.join(save_path, 'config.json')
    config.DEVICE = 'cuda'
    if not os.path.exists(save_path):
        with open(save_path, 'w') as f:
            json.dump(config, f)

    print('\nstart training...\n')
    model.train()

def test_cluster():
    config = load_config()
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
    util.set_random_seed(config.SEED)

    if config.VERBOSE:
        print(config)

    model = MMGNet(config)
    model.validation()


def load_config():
    r"""loads model config

    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, default='config_example.json', help='configuration file name. Relative path under given path (default: config.yml)')
    parser.add_argument('--loadbest', type=int, default=0, choices=[0,1], help='1: load best model or 0: load checkpoints. Only works in non training mode.')
    parser.add_argument('--mode', type=str, choices=['train','trace','eval'], help='mode. can be [train,trace,eval]',required=True)
    parser.add_argument('--exp', type=str)

    args = parser.parse_args(['--config', '/home/honsen/honsen/SceneGraph/ourPretrained_SGG/config/mmgnet.json', '--mode',
                              'train','--exp','/home/honsen/tartan/msg_data/pretrained_param_trans'])

    # args = parser.parse_args()
    config_path = os.path.abspath(args.config)



    if not os.path.exists(config_path):
        raise RuntimeError('Targer config file does not exist. {}' & config_path)
    
    # load config file
    config = Config(config_path)
    
    if 'NAME' not in config:
        config_name = os.path.basename(args.config)
        if len(config_name) > len('config_'):
            name = config_name[len('config_'):]
            name = os.path.splitext(name)[0]
            translation_table = dict.fromkeys(map(ord, '!@#$'), None)
            name = name.translate(translation_table)
            config['NAME'] = name            
    config.LOADBEST = args.loadbest
    config.MODE = args.mode
    config.exp = args.exp

    return config

if __name__ == '__main__':
    import torch.multiprocessing as mp
    # mp.set_start_method('spawn',force=True)
    # test_cluster()
    main()
