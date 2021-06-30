# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Distributed helpers."""

import functools
import logging
import pickle
import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler
from torch.distributed import group
from torch.distributed.distributed_c10d import _rank_not_in_group

_LOCAL_PROCESS_GROUP = None


def all_gather(tensors):
    """
    All gathers the provided tensors from all processes across machines.
    Args:
        tensors (list): tensors to perform all gather across all processes in
        all machines.
    """

    gather_list = []
    output_tensor = []
    world_size = dist.get_world_size()
    for tensor in tensors:
        tensor_placeholder = [
            torch.ones_like(tensor) for _ in range(world_size)
        ]
        dist.all_gather(tensor_placeholder, tensor, async_op=False)
        gather_list.append(tensor_placeholder)
    for gathered_tensor in gather_list:
        output_tensor.append(torch.cat(gathered_tensor, dim=0))
    return output_tensor


def all_reduce(tensors, average=True):
    """
    All reduce the provided tensors from all processes across machines.
    Args:
        tensors (list): tensors to perform all reduce across all processes in
        all machines.
        average (bool): scales the reduced tensor by the number of overall
        processes across all machines.
    """

    for tensor in tensors:
        dist.all_reduce(tensor, async_op=False)
    if average:
        world_size = dist.get_world_size()
        for tensor in tensors:
            tensor.mul_(1.0 / world_size)
    return tensors


def init_process_group(
    local_rank,
    local_world_size,
    shard_id,
    num_shards,
    init_method,
    dist_backend="nccl",
):
    """
    Initializes the default process group.
    Args:
        local_rank (int): the rank on the current local machine.
        local_world_size (int): the world size (number of processes running) on
        the current local machine.
        shard_id (int): the shard index (machine rank) of the current machine.
        num_shards (int): number of shards for distributed training.
        init_method (string): supporting three different methods for
            initializing process groups:
            "file": use shared file system to initialize the groups across
            different processes.
            "tcp": use tcp address to initialize the groups across different
        dist_backend (string): backend to use for distributed training. Options
            includes gloo, mpi and nccl, the details can be found here:
            https://pytorch.org/docs/stable/distributed.html
    """
    # Sets the GPU to use.
    torch.cuda.set_device(local_rank)
    # Initialize the process group.
    proc_rank = local_rank + shard_id * local_world_size
    world_size = local_world_size * num_shards
    dist.init_process_group(
        backend=dist_backend,
        init_method=init_method,
        world_size=world_size,
        rank=proc_rank,
    )


def is_master_proc(num_gpus=8):
    """
    Determines if the current process is the master process.
    """
    if torch.distributed.is_initialized():
        return dist.get_rank() % num_gpus == 0
    else:
        return True


def is_root_proc():
    """
    Determines if the current process is the root process.
    """
    if torch.distributed.is_initialized():
        return dist.get_rank() == 0
    else:
        return True


def get_world_size():
    """
    Get the size of the world.
    """
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    """
    Get the rank of the current process.
    """
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


@functools.lru_cache()
def _get_global_gloo_group():
    """
    Return a process group based on gloo backend, containing all the ranks
    The result is cached.
    Returns:
        (group): pytorch dist group.
    """
    if dist.get_backend() == "nccl":
        return dist.new_group(backend="gloo")
    else:
        return dist.group.WORLD


def _serialize_to_tensor(data, group):
    """
    Seriialize the tensor to ByteTensor. Note that only `gloo` and `nccl`
        backend is supported.
    Args:
        data (data): data to be serialized.
        group (group): pytorch dist group.
    Returns:
        tensor (ByteTensor): tensor that serialized.
    """

    backend = dist.get_backend(group)
    assert backend in ["gloo", "nccl"]
    device = torch.device("cpu" if backend == "gloo" else "cuda")

    buffer = pickle.dumps(data)
    if len(buffer) > 1024 ** 3:
        logger = logging.getLogger(__name__)
        logger.warning(
            "Rank {} trying to all-gather {:.2f} GB of data on device {}".format(
                get_rank(), len(buffer) / (1024 ** 3), device
            )
        )
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to(device=device)
    return tensor


def _pad_to_largest_tensor(tensor, group):
    """
    Padding all the tensors from different GPUs to the largest ones.
    Args:
        tensor (tensor): tensor to pad.
        group (group): pytorch dist group.
    Returns:
        list[int]: size of the tensor, on each rank
        Tensor: padded tensor that has the max size
    """
    world_size = dist.get_world_size(group=group)
    assert (
        world_size >= 1
    ), "comm.gather/all_gather must be called from ranks within the given group!"
    local_size = torch.tensor(
        [tensor.numel()], dtype=torch.int64, device=tensor.device
    )
    size_list = [
        torch.zeros([1], dtype=torch.int64, device=tensor.device)
        for _ in range(world_size)
    ]
    dist.all_gather(size_list, local_size, group=group)
    size_list = [int(size.item()) for size in size_list]

    max_size = max(size_list)

    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    if local_size != max_size:
        padding = torch.zeros(
            (max_size - local_size,), dtype=torch.uint8, device=tensor.device
        )
        tensor = torch.cat((tensor, padding), dim=0)
    return size_list, tensor


def all_gather_unaligned(data, group=None):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors).
    Args:
        data: any picklable object
        group: a torch process group. By default, will use a group which
            contains all ranks on gloo backend.
    Returns:
        list[data]: list of data gathered from each rank
    """
    if get_world_size() == 1:
        return [data]
    if group is None:
        group = _get_global_gloo_group()
    if dist.get_world_size(group) == 1:
        return [data]

    tensor = _serialize_to_tensor(data, group)

    size_list, tensor = _pad_to_largest_tensor(tensor, group)
    max_size = max(size_list)

    # receiving Tensor from all ranks
    tensor_list = [
        torch.empty((max_size,), dtype=torch.uint8, device=tensor.device)
        for _ in size_list
    ]
    dist.all_gather(tensor_list, tensor, group=group)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def init_distributed_training(args):
    """
    Initialize variables needed for distributed training.
    """
    if args.dist.n_gpus <= 1:
        return
    num_gpus_per_machine = args.dist.n_gpus
    num_machines = dist.get_world_size() // num_gpus_per_machine
    for i in range(num_machines):
        ranks_on_i = list(
            range(i * num_gpus_per_machine, (i + 1) * num_gpus_per_machine)
        )
        pg = dist.new_group(ranks_on_i)
        if i == args.dist.shard_id:
            global _LOCAL_PROCESS_GROUP
            _LOCAL_PROCESS_GROUP = pg


def get_local_size() -> int:
    """
    Returns:
        The size of the per-machine process group,
        i.e. the number of processes per machine.
    """
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size(group=_LOCAL_PROCESS_GROUP)


def get_local_rank() -> int:
    """
    Returns:
        The rank of the current process within the local (per-machine) process group.
    """
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    assert _LOCAL_PROCESS_GROUP is not None
    return dist.get_rank(group=_LOCAL_PROCESS_GROUP)


def shuffle_dataset(loader, cur_epoch):
    """ "
    Shuffles the data.
    Args:
        loader (loader): data loader to perform shuffle.
        cur_epoch (int): number of the current epoch.
    """
    sampler = loader.sampler
    if sampler is None:
        return
    assert isinstance(
        sampler, (RandomSampler, DistributedSampler, torch.utils.data.sampler.SequentialSampler)
    ), "Sampler type '{}' not supported".format(type(sampler))
    # RandomSampler handles shuffling automatically
    if isinstance(sampler, DistributedSampler):
        # DistributedSampler shuffles data based on epoch
        sampler.set_epoch(cur_epoch)

def _object_to_tensor(obj):
    """
    Helper function to convert an arbitrary picklable python object to a tensor
    Returns a torch.ByteTensor representing the pickled object.
    """
    buf = pickle.dumps(obj)
    storage = torch.ByteStorage.from_buffer(buf)
    byte_tensor = torch.ByteTensor(storage).cuda()
    size = torch.LongTensor([byte_tensor.numel()]).cuda()
    return byte_tensor, size

def _tensor_to_object(tensor, size):
    """
    Helper function to convert a ByteTensor created by _object_to_tensor back to
    the picklable python object.
    """
    buf = tensor.cpu().numpy().tobytes()[:size]
    return pickle.loads(buf)


def all_gather_object(object, group=group.WORLD, async_op=False):
    """
    Performs all_gather on a picklable object.
    """
    if _rank_not_in_group(group):
        return


    byte_tensor, size = _object_to_tensor(object)

    group_size = torch.distributed.get_world_size(group=group)
    object_size_list = [torch.LongTensor([0]).cuda() for _ in range(group_size)]

    # collect the object sizes from all the tensors
    dist.all_gather(object_size_list, size, group=group)

    # create the list of output tensors needed for all_gather
    max_object_size = max(object_size_list)
    tensor_list = [torch.ByteTensor(size=(max_object_size,)).cuda() for _ in range(group_size)]

    if size != max_object_size:
        # all tensors input into gather should be of the same size, so pad the local tensor if it is not the max size.
        padding = torch.ByteTensor(size=(max_object_size - size,)).cuda()
        byte_tensor = torch.cat((byte_tensor, padding), dim=0).cuda()
    dist.all_gather(tensor_list, byte_tensor, group=group, async_op=async_op)
    # unpickle the tensors back into objects.
    objects = [_tensor_to_object(tensor, object_size) for object_size, tensor in zip(object_size_list, tensor_list)]
    object_list = []
    for object in objects:
        object_list += object
    return object_list