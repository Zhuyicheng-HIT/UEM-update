"""Two-rank DDP smoke test for all Global/Local fusion topologies."""

import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from model.uniegomotion import GlobalLocalOutputHead


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")

    for mode in sorted(GlobalLocalOutputHead.MODES):
        torch.manual_seed(62)
        head = GlobalLocalOutputHead(
            latent_dim=32,
            output_dim=243,
            global_start=198,
            global_end=207,
            mode=mode,
            dropout=0.0,
            stop_gradient=True,
        ).cuda(local_rank)
        model = DistributedDataParallel(
            head,
            device_ids=[local_rank],
            find_unused_parameters=False,
            static_graph=True,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4)

        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            shared = torch.randn(2, 4, 32, device=local_rank)
            output = model(shared)
            if output.shape != (2, 4, 243) or not torch.isfinite(output).all():
                raise AssertionError(f"Invalid {mode} output: {tuple(output.shape)}")
            output.square().mean().backward()
            missing = [name for name, parameter in model.named_parameters() if parameter.grad is None]
            if missing:
                raise AssertionError(f"{mode} has unused DDP parameters: {missing}")
            optimizer.step()

        if dist.get_rank() == 0:
            print(f"DDP smoke passed: {mode}", flush=True)
        dist.barrier()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
