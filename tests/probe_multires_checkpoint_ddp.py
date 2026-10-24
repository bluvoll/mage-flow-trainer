"""CPU DDP save probe: torchrun --standalone --nproc_per_node=2 THIS_FILE OUTPUT."""
import sys

import torch
import torch.distributed as dist
from accelerate import Accelerator

from test_moss_image import inputs
from test_multires_checkpoint_save import make_trainer
from trainer.tools.inspect_checkpoint import read_metadata


def main():
    acc = Accelerator(cpu=True)
    try:
        assert acc.num_processes == 2
        trainer = make_trainer(sys.argv[1], {512: 24, 768: 12, 1024: 8})
        trainer.accelerator = acc
        trainer.transformer, trainer.optimizer = acc.prepare(trainer.transformer, trainer.optimizer)
        xs, times, text, mask = inputs()
        prediction = trainer.transformer(xs, times, (text, mask))[0]
        acc.backward(sum(y.square().mean() for y in prediction))
        trainer.optimizer.step()
        trainer.optimizer.zero_grad(set_to_none=True)
        for tag in ('epoch001', 'step000005', 'final'):
            trainer.save(tag)
            acc.wait_for_everyone()
            stem = 'multires' if tag == 'final' else 'multires-' + tag
            metadata = read_metadata(trainer.out_dir / (stem + '.safetensors'))
            assert metadata['training_state']['effective_batch_size'] == {'512': 48, '768': 24, '1024': 16}
            assert metadata['training_state']['world_size'] == 2
        acc.print('PASS: both DDP ranks completed epoch, step and final saves; tiered batch metadata round-trips.')
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
