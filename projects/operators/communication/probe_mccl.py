#!/usr/bin/env python3
import os
import torch
import torch_musa  # noqa: F401
import torch.distributed as dist


def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    torch.musa.set_device(local_rank)
    dist.init_process_group(backend="mccl", rank=rank, world_size=world_size)

    x = torch.tensor([rank + 1.0], device=f"musa:{local_rank}")
    dist.all_reduce(x)
    print(f"rank={rank} local_rank={local_rank} value={x.item()}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
