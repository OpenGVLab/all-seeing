import os
import time
import subprocess

import torch
import torch.distributed as dist


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            curr_time = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
            builtin_print(f'[{curr_time}]', *args, **kwargs)

    __builtin__.print = print


def init_distributed_mode():
    if 'SLURM_PROCID' in os.environ:
        rank = int(os.environ['SLURM_PROCID'])
        local_rank = rank % torch.cuda.device_count()

        world_size = int(os.environ["SLURM_NTASKS"])
        local_size = int(os.environ["SLURM_NTASKS_PER_NODE"])

        if "MASTER_PORT" not in os.environ:
            port = 22222
            # for i in range(22222, 65535):
            #     cmd = f'netstat -aon|grep {i}'
            #     with os.popen(cmd, 'r') as file:
            #         if '' == file.read():
            #             port = i
            #             break

            print(f'MASTER_PORT = {port}')
            os.environ["MASTER_PORT"] = str(port)

            time.sleep(3)

        node_list = os.environ["SLURM_NODELIST"]
        addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = addr

        os.environ['RANK'] = str(rank)
        os.environ['LOCAL_RANK'] = str(local_rank)
        os.environ['LOCAL_WORLD_SIZE'] = str(local_size)
        os.environ['WORLD_SIZE'] = str(world_size)

        # setup_for_distributed(rank == 0)


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()
