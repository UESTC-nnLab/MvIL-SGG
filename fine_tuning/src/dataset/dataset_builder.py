#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
sys.path.append('/home/honsen/honsen/ourPretrained_SGG')
from xml.dom.minidom import Node
from src.dataset.dataset_pre_3ds import SSGDatasetGraph1
from src.dataset.dataset_3dssg import SSGDatasetGraph
from src.dataset.dataset_honsen import HonsenDatasetGraph

def build_dataset1(config, split_type, shuffle_objs, multi_rel_outputs,
                  use_rgb, use_normal):
    if split_type != 'train_scans' and split_type != 'validation_scans' and split_type != 'test_scans':
        raise RuntimeError(split_type)
    
    dataset = SSGDatasetGraph1(
        config,
        split=split_type,
        multi_rel_outputs=multi_rel_outputs,
        shuffle_objs=shuffle_objs,
        use_rgb=use_rgb,
        use_normal=use_normal,
        label_type='3RScan160',
        for_train=split_type == 'train_scans',
        max_edges = config.dataset.max_edges
    )
    return dataset

def build_dataset(config, split_type, shuffle_objs, multi_rel_outputs,
                  use_rgb, use_normal):
    if split_type != 'train_scans' and split_type != 'validation_scans' and split_type != 'test_scans':
        raise RuntimeError(split_type)
    
    dataset = SSGDatasetGraph(
        config,
        split=split_type,
        multi_rel_outputs=multi_rel_outputs,
        shuffle_objs=shuffle_objs,
        use_rgb=use_rgb,
        use_normal=use_normal,
        label_type='3RScan160',
        for_train= split_type == 'train_scans',
        max_edges = config.dataset.max_edges
    )
    return dataset

def build_honsenDataset():
    dataset = HonsenDatasetGraph(split='train_scannet',shuffle_objs=True, for_train=True)
    dataset[2]
    return dataset


if __name__ == '__main__':
    import numpy as np

    build_honsenDataset()

    # from config import Config
    # config = Config('../config_example.json')
    # config.dataset.root = '../data/example_data'
    # build_dataset(config, split_type = 'train_scans', shuffle_objs=True, multi_rel_outputs=False,use_rgb=True,use_normal=True)