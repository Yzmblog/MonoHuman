import imp

from configs import cfg
import logging

import random
import numpy as np
import torch
import torch.distributed as dist
import math
import os
from torch.nn.parallel import DistributedDataParallel as DDP
import subprocess

def worker_init_fn(worker_id):
    random.seed(worker_id+100)
    np.random.seed(worker_id+100)
    torch.manual_seed(worker_id+100)

def ddp_init(args):

    local_rank = args.local_rank
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend = 'nccl')  # 'nccl' for GPU, 'gloo/mpi' for CPU

    rank = torch.distributed.get_rank()

    random.seed(rank)
    np.random.seed(rank)
    torch.manual_seed(rank)
    print(f"local_rank {local_rank} rank {rank} launched...")

    return rank, local_rank


def setup_distributed(backend="nccl", port=None):
    """Initialize distributed training environment.
    support both slurm and torch.distributed.launch
    see torch.distributed.init_process_group() for more details
    """
    num_gpus = torch.cuda.device_count()

    if "SLURM_JOB_ID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        node_list = os.environ["SLURM_NODELIST"]
        addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
        # specify master port
        if port is not None:
            os.environ["MASTER_PORT"] = str(port)
        elif "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "29500"
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = addr
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(rank % num_gpus)
        os.environ["RANK"] = str(rank)
    else:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(rank % num_gpus)

    dist.init_process_group(
        backend=backend,
        world_size=world_size,
        rank=rank,
    )

def _query_network():
    module = cfg.network_module
    module_path = module.replace(".", "/") + ".py"
    network = imp.load_source(module, module_path).Network
    return network


def create_network(local_rank=0):
    network = _query_network()
    network = network()

    # DDP: load parameters first (only on master node), then make ddp model
    if cfg.ddp:
        logging.info("use Distributed Data Parallel...")
        local_rank = int(os.environ["LOCAL_RANK"])
        net = network.to(local_rank) 
        
        net = DDP(net, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    # DP: make dp model, then load parameters to all devices 
    else:
        net = network
    return net

def synchronize():
    if dist.get_world_size() > 1:
        dist.barrier()
    return

class ddpSampler:
    """
    ddp sampler for inference
    """
    def __init__(self, dataset, rank=None, num_replicas=None):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def indices(self):
        indices = list(range(len(self.dataset)))
        # add extra samples to make it evenly divisible
        indices += [indices[-1]] * (self.total_size - len(indices))
        # subsample
        indices = indices[self.rank * self.num_samples : (self.rank + 1) * self.num_samples]
        return indices

    def len(self):
        return self.num_samples

    def distributed_concat(self, tensor, num_total_examples):
        output_tensors = [tensor.clone() for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(output_tensors, tensor)
        concat = torch.cat(output_tensors, dim=0)
        # truncate the dummy elements added by SequentialDistributedSampler
        return concat[:num_total_examples]