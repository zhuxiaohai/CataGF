import argparse
from urllib.parse import urlparse
import os
import sys
import tempfile
import torch
from torch.utils.data import DataLoader, Subset
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP


class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(1, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


def demo_basic(local_world_size, local_rank):
    # setup devices for this process. For local_world_size = 2, num_gpus = 8,
    # rank 0 uses GPUs [0, 1, 2, 3] and
    # rank 1 uses GPUs [4, 5, 6, 7].
    n = torch.cuda.device_count() // local_world_size
    device_ids = list(range(local_rank * n, (local_rank + 1) * n))

    print(
        f"[{os.getpid()}] rank = {dist.get_rank()}, "
        + f"world_size = {dist.get_world_size()}, n = {n}, device_ids = {device_ids} \n", end=''
    )

    model = ToyModel().cuda(device_ids[0])
    ddp_model = DDP(model, device_ids)

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    dataset = torch.Tensor(range(9)).unsqueeze(-1)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False, drop_last=True)
    batch_size = 2
    dl = DataLoader(dataset, batch_size=batch_size, sampler=sampler, shuffle=False)
    print(
        'rank {} with loader len {} sampler len {} dataset len {}'.format(local_rank, len(dl), len(sampler), len(dl.dataset)))
    batch_num = 0
    for batch in dl:
        batch_num += 1
        batch = batch.cuda(device_ids[0])
        optimizer.zero_grad()
        outputs = ddp_model(batch)
        labels = torch.randn(batch.shape[0], 5).to(device_ids[0])
        loss_fn(outputs, labels).backward()
        optimizer.step()
        print('rank {} with batch {} batch_num {}'.format(local_rank, batch, batch_num))
    if (len(dl.sampler) * dist.get_world_size() < len(dl.dataset)) and (device_ids[0] == 0) and (dist.get_rank() == 0):
        aux_val_dataset = Subset(dl.dataset,
                                 range(len(dl.sampler) * dist.get_world_size(), len(dl.dataset)))
        aux_val_loader = DataLoader(
            aux_val_dataset, batch_size=batch_size, shuffle=False)
        for batch in aux_val_loader:
            batch_num += 1
            batch = batch.cuda(device_ids[0])
            optimizer.zero_grad()
            outputs = ddp_model(batch)
            labels = torch.randn(batch.shape[0], 5).to(device_ids[0])
            print('rank {} with batch {} batch_num {}'.format(local_rank, batch, batch_num))


def spmd_main(local_world_size, local_rank):
    # These are the parameters used to initialize the process group
    env_dict = {
        key: os.environ[key]
        for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE")
    }

    if sys.platform == "win32":
        # Distributed package only covers collective communications with Gloo
        # backend and FileStore on Windows platform. Set init_method parameter
        # in init_process_group to a local file.
        if "INIT_METHOD" in os.environ.keys():
            print(f"init_method is {os.environ['INIT_METHOD']}")
            url_obj = urlparse(os.environ["INIT_METHOD"])
            if url_obj.scheme.lower() != "file":
                raise ValueError("Windows only supports FileStore")
            else:
                init_method = os.environ["INIT_METHOD"]
        else:
            # It is a example application, For convience, we create a file in temp dir.
            temp_dir = tempfile.gettempdir()
            init_method = f"file:///{os.path.join(temp_dir, 'ddp_example')}"
        dist.init_process_group(backend="gloo", init_method=init_method, rank=int(env_dict["RANK"]),
                                world_size=int(env_dict["WORLD_SIZE"]))
    else:
        print(f"[{os.getpid()}] Initializing process group with: {env_dict}")
        dist.init_process_group(backend="nccl")

    print(
        f"[{os.getpid()}]: world_size = {dist.get_world_size()}, "
        + f"rank = {dist.get_rank()}, backend={dist.get_backend()} \n", end=''
    )

    demo_basic(local_world_size, local_rank)

    # Tear down the process group
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # This is passed in via launch.py
    parser.add_argument("--local_rank", type=int, default=0)
    # This needs to be explicitly passed in
    parser.add_argument("--local_world_size", type=int, default=1)
    args = parser.parse_args()
    # The main entry point is called directly without using subprocess
    spmd_main(args.local_world_size, args.local_rank)