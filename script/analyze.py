# coding: utf-8
from time import time
import os
import argparse
from easydict import EasyDict
import yaml
import pickle

import torch
from torch_geometric.data import DataLoader
from torch_geometric.transforms import Compose

from confgf import utils, dataset


class Analyzer(object):
    def __init__(self, val_set, test_set, gpus, config):
        self.val_set = val_set
        self.test_set = test_set
        self.gpus = gpus
        self.device = torch.device(gpus[0]) if len(gpus) > 0 else torch.device('cpu')
        self.config = config
        self.batch_size = self.config.train.batch_size

    @torch.no_grad()
    def evaluate(self, split, verbose=0):
        """
        Evaluate the model.
        Parameters:
            split (str): split to evaluate. Can be ``train``, ``val`` or ``test``.
        """
        if split not in ['train', 'val', 'test']:
            raise ValueError('split should be either train, val, or test.')

        test_set = getattr(self, "%s_set" % split)
        dataloader = DataLoader(test_set, batch_size=self.config.train.batch_size, \
                                shuffle=False, num_workers=self.config.train.num_workers)
        # code here
        eval_start = time()
        eval_losses_mean = []
        eval_losses_max = []
        eval_losses_min = []
        eval_losses_std = []
        eval_pos_mean = []
        eval_pos_max = []
        eval_pos_min = []
        eval_pos_std = []
        for batch in dataloader:
            if self.device.type == "cuda":
                batch = batch.to(self.device)
            eval_losses_mean.append(batch.edge_length.mean().item())
            eval_losses_max.append(batch.edge_length.max().item())
            eval_losses_min.append(batch.edge_length.min().item())
            eval_losses_std.append(batch.edge_length.std().item())
            eval_pos_mean.append(batch.pos.mean(axis=0))
            eval_pos_max.append(batch.pos.max(axis=0).values)
            eval_pos_min.append(batch.pos.min(axis=0).values)
            eval_pos_std.append(batch.pos.std(axis=0))
        average_loss_mean = sum(eval_losses_mean) / len(eval_losses_mean)
        average_loss_max = sum(eval_losses_max) / len(eval_losses_max)
        average_loss_min = sum(eval_losses_min) / len(eval_losses_min)
        average_loss_std = sum(eval_losses_std) / len(eval_losses_std)
        average_pos_mean = torch.stack(eval_pos_mean).mean(axis=0).cpu().numpy()
        average_pos_max = torch.stack(eval_pos_max).mean(axis=0).cpu().numpy()
        average_pos_min = torch.stack(eval_pos_min).mean(axis=0).cpu().numpy()
        average_pos_std = torch.stack(eval_pos_std).mean(axis=0).cpu().numpy()
        if verbose:
            print('Evaluate %s | Time: %.5f' % (split, time() - eval_start))
            print('edge_mean %.4f, max %.4f, min %.4f, std %.4f' % (average_loss_mean, average_loss_max, 
                                                                    average_loss_min, average_loss_std))
            print('pos_mean', average_pos_mean)
            print('pos_max', average_pos_max)
            print('pos_min', average_pos_min)
            print('pos_std', average_pos_std)
        return average_loss_mean

    def run(self):
        _ = self.evaluate('val', verbose=1)
        _ = self.evaluate('test', verbose=1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='confgf')
    parser.add_argument('--config_path', type=str, help='path of dataset',
                        default='/home/zhuxiaohai/PycharmProjects/CataGF/config/qm9_default.yml')
    parser.add_argument('--seed', type=int, default=2021, help='overwrite config seed')

    args = parser.parse_args()
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)
    config = EasyDict(config)

    # check device
    gpus = list(filter(lambda x: x is not None, config.train.gpus))
    assert torch.cuda.device_count() >= len(gpus), 'do you set the gpus in config correctly?'
    device = torch.device(gpus[0]) if len(gpus) > 0 else torch.device('cpu')
    print("Let's use", len(gpus), "GPUs!")
    print("Using device %s as main device" % device)
    config.train.device = device
    config.train.gpus = gpus
    print(config)

    load_path = os.path.join(config.data.base_path, '%s_processed' % config.data.dataset)
    print('loading data from %s' % load_path)

    train_start = time()

    if config.data.val_set is not None:
        with open(os.path.join(load_path, config.data.val_set), "rb") as fin:
            val_data = pickle.load(fin)
    if config.data.test_set is not None:
        with open(os.path.join(load_path, config.data.test_set), "rb") as fin:
            test_data = pickle.load(fin)

    transform = Compose([
        utils.AddHigherOrderEdgesBasic(order=config.model.order),
        utils.AddEdgeLength(),
        utils.AddPlaceHolder(),
        utils.AddEdgeName()
    ])
    val_data = dataset.GEOMDataset(data=val_data, transform=transform)
    test_data = dataset.GEOMDataset_PackedConf(data=test_data, transform=transform)

    solver = Analyzer(val_data, test_data, gpus, config)
    solver.run()
