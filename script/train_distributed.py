# coding: utf-8

import argparse
import numpy as np
import random
import os
import pickle
import yaml
from easydict import EasyDict

import torch
import torch.distributed as dist

from confgf import models, dataset, runner, utils


def worker(local_rank, local_world_size, config):
    # setup devices for this process. For example:
    # local_world_size = 2, num_gpus = 8,
    # process rank 0 uses GPUs [0, 1, 2, 3] and
    # process rank 1 uses GPUs [4, 5, 6, 7].
    n = torch.cuda.device_count() // local_world_size  # the number of devices this process can operate
    device_ids = list(range(local_rank * n, (local_rank + 1) * n))  # corresponding device ids

    print(
        f"[{os.getpid()}] rank = {dist.get_rank()}, local_rank = {local_rank}, "
        + f"world_size = {dist.get_world_size()}, devices_num = {n}, device_ids = {device_ids}"
    )

    config.train.batch_size = int(config.train.batch_size / local_world_size)

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

    train_data = []
    val_data = []
    test_data = []

    if config.data.train_set is not None:
        with open(os.path.join(load_path, config.data.train_set), "rb") as fin:
            train_data = pickle.load(fin)
    if config.data.val_set is not None:
        with open(os.path.join(load_path, config.data.val_set), "rb") as fin:
            val_data = pickle.load(fin)
    print('train size : %d  ||  val size: %d  ||  test size: %d ' % (len(train_data), len(val_data), len(test_data)))
    print('loading data done!')

    transform = None
    # train_data = dataset.GEOMDataset(data=train_data, transform=transform)
    # val_data = dataset.GEOMDataset(data=val_data, transform=transform)
    # test_data = dataset.GEOMDataset_PackedConf(data=test_data, transform=transform)
    train_data = dataset.CATADataset(data=train_data, transform=transform)
    val_data = dataset.CATADataset(data=val_data, transform=transform)
    test_data = dataset.CATADataset(data=test_data, transform=transform)

    model = models.DistanceScoreMatch(config)

    optimizer = utils.get_optimizer(config.train.optimizer, model)
    scheduler = utils.get_scheduler(config.train.scheduler, optimizer)

    solver = runner.DefaultRunner(train_data, val_data, test_data, model, optimizer, scheduler, device_ids, config, device_ids[0])
    if config.train.resume_train:
        # Use a barrier() to make sure that process 1 loads the model after process
        # 0 saves it.
        dist.barrier()
        # configure map_location properly
        map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank}
        solver.load(config.train.resume_checkpoint, epoch=config.train.resume_epoch, load_optimizer=True,
                    load_scheduler=True, map_location=map_location)
    solver.train()


def spmd_main(local_world_size, local_rank, config):
    # These are the parameters used to initialize the process group
    env_dict = {
        key: os.environ[key]
        for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE")
    }
    print(f"[{os.getpid()}] Initializing process group with: {env_dict}")
    dist.init_process_group(backend="nccl")
    print(
        f"[{os.getpid()}] world_size = {dist.get_world_size()}, "
        + f"rank = {dist.get_rank()}, local_rank = {local_rank}, backend={dist.get_backend()}"
    )
    worker(local_rank, local_world_size, config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='confgf')
    parser.add_argument('--config_path', type=str, help='path of dataset',
                        default='/home/zhuxiaohai/PycharmProjects/CataGF/config/cata_default.yml')
    parser.add_argument('--seed', type=int, default=2021, help='overwrite config seed')
    parser.add_argument("--local_rank", type=int, default=0, help='this value is automatically added by distributed.launch.py')
    parser.add_argument("--local_world_size", type=int, default=1)
    args = parser.parse_args()

    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)
    config = EasyDict(config)

    if args.seed != 2021:
        config.train.seed = args.seed

    if config.train.save and config.train.save_path is not None:
        config.train.save_path = os.path.join(config.train.save_path, config.model.name)
        if not os.path.exists(config.train.save_path):
            os.makedirs(config.train.save_path)
    spmd_main(args.local_world_size, args.local_rank, config)

