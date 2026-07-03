"""Minimal 2-GPU NCCL connectivity test — isolates NCCL transport from Lightning.

Fails fast (120s PG timeout) instead of the 30-min default, so a transport
problem (P2P/SHM blocked by Condor cgroups) surfaces quickly. Run with:
    torchrun --standalone --nproc_per_node=<N> nccl_test.py
"""
import datetime
import os

import torch
import torch.distributed as dist

dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=120))
rank = dist.get_rank()
world = dist.get_world_size()
local = int(os.environ.get("LOCAL_RANK", rank))
torch.cuda.set_device(local)
print(f"[rank {rank}/{world}] local={local} dev={torch.cuda.current_device()} "
      f"{torch.cuda.get_device_name(local)}", flush=True)

t = torch.ones(1, device="cuda") * (rank + 1)
dist.all_reduce(t)                       # first real collective
print(f"[rank {rank}] all_reduce -> {t.item()} (expect {world*(world+1)//2})", flush=True)
dist.barrier()
if rank == 0:
    print("NCCL TEST PASSED", flush=True)
dist.destroy_process_group()
