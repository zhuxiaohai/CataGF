# coding: utf-8

import argparse
import numpy as np
import random
import os
import pickle
import yaml
from easydict import EasyDict

import torch
from torch_geometric.transforms import Compose
from confgf import models, dataset, runner, utils

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='confgf')
    parser.add_argument('--config_path', type=str, help='path of dataset',
                        default='/home/zhuxiaohai/PycharmProjects/CataGF/config/qm9_default.yml')
    parser.add_argument('--seed', type=int, default=2021, help='overwrite config seed')
    parser.add_argument('--val_file', type=str, default='val_data_5k.pkl')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--tot_num', type=int, default=500)

    args = parser.parse_args()
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)
    config = EasyDict(config)

    if args.seed != 2021:
        config.train.seed = args.seed

    # check device
    gpus = list(filter(lambda x: x is not None, config.train.gpus))
    assert torch.cuda.device_count() >= len(gpus), 'do you set the gpus in config correctly?'
    device = torch.device(gpus[0]) if len(gpus) > 0 else torch.device('cpu')
    print("Let's use", len(gpus), "GPUs!")
    print("Using device %s as main device" % device)
    config.train.device = device
    config.train.gpus = gpus

    print(config)

    # set random seed
    np.random.seed(config.train.seed)
    random.seed(config.train.seed)
    torch.manual_seed(config.train.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.train.seed)
        torch.cuda.manual_seed_all(config.train.seed)
    torch.backends.cudnn.benchmark = True
    print('set seed for random, numpy and torch')

    load_path = os.path.join(config.data.base_path, '%s_processed' % config.data.dataset)
    print('loading data from %s' % load_path)

    with open(os.path.join(load_path, args.val_file), "rb") as fin:
        val_data = pickle.load(fin)
    print('val size: %d' % (len(val_data)))
    print('loading data done!')
    val_data = val_data[:args.tot_num]

    transform = Compose([
        utils.AddHigherOrderEdges(order=config.model.order),
        utils.AddEdgeLength(),
        utils.AddPlaceHolder(),
        utils.AddEdgeName()
    ])
    # train_data = dataset.GEOMDataset(data=train_data, transform=transform)
    # val_data = dataset.GEOMDataset(data=val_data, transform=transform)
    # test_data = dataset.GEOMDataset_PackedConf(data=test_data, transform=transform)
    train_data = None
    val_data = dataset.GEOMDataset_PackedConf(data=val_data, transform=transform)
    test_data = None

    model = models.DistanceScoreMatch(config)
    optimizer = None
    scheduler = None

    solver = runner.DefaultRunner(train_data, val_data, test_data, model, optimizer, scheduler, gpus, config)
    assert config.test.init_checkpoint is not None
    solver.load(config.test.init_checkpoint, epoch=config.test.epoch)
    solver.analyze('val', args.batch_size, os.path.join(load_path, args.val_file.strip('pkl')+'csv'))


