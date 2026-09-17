"""Run with torchrun --standalone --nproc_per_node=2; small-model DDP integration check."""
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from trainer.modeling.mage_flow import MageFlow, MageFlowParams
from trainer.training.flow import FlowConfig, prepare_training_flow_batch, flow_loss
from trainer.training.params import AdapterConfig, apply_adapter


def main():
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl')
    try:
        for adapter in (False, True):
            torch.manual_seed(42)
            model = MageFlow(MageFlowParams(128,128,24,32,4,2,[2,2,4],True,
                                            modulation_rank=8)).to(rank)
            if adapter:
                apply_adapter(model, AdapterConfig(kind='lycoris_lora', lycoris_algo='lokr',
                                                   rank=10000, alpha=10000, lokr_factor=1,
                                                   dtype='float32'))
            model.configure_execution(True, attention_backend='torch_varlen')
            # varlen requires half precision; compressed factors can stay FP32.
            model.to(dtype=torch.bfloat16)
            ddp = DistributedDataParallel(model,device_ids=[rank],find_unused_parameters=True)
            opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.01)
            torch.manual_seed(100+rank)
            for _ in range(2):
                clean = torch.randn(1,128,1,3,4,device=rank)
                noisy,t,target,s,mask = prepare_training_flow_batch(clean,FlowConfig(dual_timestep=True))
                ctx = (torch.randn(1,5,24,device=rank,dtype=torch.bfloat16),
                       torch.ones(1,5,device=rank,dtype=torch.bool))
                pred = ddp([noisy.to(torch.bfloat16)],t.to(torch.bfloat16),ctx,
                           second_timestep=s.to(torch.bfloat16),timestep_mask=[mask])[0][0]
                loss = flow_loss(pred,target)
                loss.backward()
                assert torch.isfinite(loss)
                opt.step()
                opt.zero_grad(set_to_none=True)
            for p in model.parameters():
                if p.requires_grad:
                    reference = p.detach().clone()
                    dist.broadcast(reference,src=0)
                    torch.testing.assert_close(p,reference,rtol=0,atol=0)
            if rank == 0:
                print(f'DDP dual timestep passed: {"LoKr" if adapter else "finetune"}',flush=True)
            del ddp, model, opt
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
