import copy
from dataclasses import asdict
import json
import unittest

import torch

from trainer.experimental.rti import (
    BudgetSchedule, RegionInterface, RTIMageFlow, partition_regions,
    rectangular_hilbert_order,
)
from trainer.modeling.mage_flow import MageFlow, MageFlowParams


class RTITests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)

    def test_rectangles_cover_every_token_and_regions_are_connected(self):
        for h,w in ((1,1),(1,13),(13,1),(3,5),(84,48),(48,84),(7,8)):
            n=h*w
            self.assertEqual(sorted(rectangular_hilbert_order(h,w)),list(range(n)))
            features=torch.randn(n,4)
            for requested in set((1,max(1,n//4),n)):
                p=partition_regions(features,h,w,requested)
                self.assertEqual(p.regions,max(requested,p.minimum_regions))
                self.assertTrue((p.counts>0).all())
                self.assertEqual(p.counts.sum().item(),n)
                a,b=p.order[:-1],p.order[1:]
                distances=(a//w-b//w).abs()+(a%w-b%w).abs()
                self.assertTrue((distances[p.labels[:-1]==p.labels[1:]]==1).all())

    def test_initial_read_write_and_rope(self):
        features=torch.randn(15,8,requires_grad=True)
        frequencies=torch.polar(torch.ones(15,2),torch.randn(15,2))
        p=partition_regions(features,3,5,4)
        interface=RegionInterface(8)
        pooled,rope=interface.read(features,frequencies,p)
        for region in range(p.regions):
            members=p.order[p.labels==region]
            torch.testing.assert_close(pooled[region],features[members].mean(0))
            torch.testing.assert_close(rope[region],frequencies[members].mean(0))
        torch.testing.assert_close(interface.restore(features,torch.zeros_like(pooled),p),features)
        delta=torch.randn_like(pooled)
        output=interface.restore(features,delta,p)
        torch.testing.assert_close(output[p.order],features[p.order]+delta[p.labels])
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(features.grad).all())

    def test_schedule_resume_and_boundaries(self):
        schedule=BudgetSchedule(start_keep=.98,target_keep=.5,anneal_steps=100,warmup_steps=10)
        self.assertAlmostEqual(schedule.keep_fraction(0),.98)
        self.assertAlmostEqual(schedule.keep_fraction(10),.98)
        self.assertAlmostEqual(schedule.keep_fraction(110),.5)
        self.assertAlmostEqual(schedule.keep_fraction(1000),.5)
        recovered=BudgetSchedule(**json.loads(json.dumps(asdict(schedule))))
        original=[schedule.regions(4032,step) for step in range(151)]
        resumed=[recovered.regions(4032,step) for step in range(75,151)]
        self.assertEqual(original[75:],resumed)
        self.assertEqual(original,sorted(original,reverse=True))
        for options in (dict(target_keep=0),dict(target_keep=1),dict(anneal_steps=0)):
            with self.assertRaises(ValueError):
                BudgetSchedule(**options)

    def test_dense_parity_and_interface_gradients(self):
        for rank in (0,8):
            base=MageFlow(MageFlowParams(128,128,24,32,4,4,[2,2,4],True,modulation_rank=rank))
            base.requires_grad_(False)
            wrapped=RTIMageFlow(base,1,3)
            image=torch.randn(1,128,1,3,5)
            t=torch.tensor([.6])
            ctx=(torch.randn(1,5,24),torch.tensor([[1,1,1,0,0]]).bool())
            torch.testing.assert_close(base(image,t,ctx)[0],wrapped(image,t,ctx)[0],atol=0,rtol=0)
            original={n:p.detach().clone() for n,p in base.named_parameters()}
            output=wrapped(image,t,ctx,keep_fraction=.75)[0]
            self.assertEqual(output.shape,image.shape)
            self.assertLess(wrapped.last_budget['actual_regions'],15)
            loss=output.square().mean()
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertGreater(wrapped.interface.write.weight.grad.abs().sum(),0)
            self.assertGreater(wrapped.interface.read_score.weight.grad.abs().sum(),0)
            opt=torch.optim.AdamW(wrapped.interface.parameters(),lr=1e-3)
            opt.step()
            for n,p in base.named_parameters():
                self.assertIsNone(p.grad)
                torch.testing.assert_close(p,original[n],atol=0,rtol=0)

    def test_checkpoint_matches_uncheckpointed_and_state_roundtrip(self):
        base=MageFlow(MageFlowParams(128,128,24,32,4,4,[2,2,4],True))
        a=RTIMageFlow(base,1,3)
        b=copy.deepcopy(a)
        b.backbone.configure_execution(False)
        image=torch.randn(1,128,1,3,5)
        ctx=(torch.randn(1,3,24),torch.ones(1,3,dtype=torch.bool))
        t=torch.tensor([.4])
        y=a(image,t,ctx,keep_fraction=.5)[0]
        z=b(image,t,ctx,keep_fraction=.5)[0]
        torch.testing.assert_close(y,z)
        y.square().mean().backward();z.square().mean().backward()
        for (n,p),(_,q) in zip(a.named_parameters(),b.named_parameters()):
            torch.testing.assert_close(p.grad,q.grad,msg=n)
        # Tensor roundtrip only: production metadata/export remains a future milestone.
        b.load_state_dict(a.state_dict(),strict=True)
        torch.testing.assert_close(b(image,t,ctx,keep_fraction=.5)[0],y)

    def test_conflicts_are_explicit(self):
        base=MageFlow(MageFlowParams(128,128,24,32,4,4,[2,2,4],False))
        with self.assertRaisesRegex(ValueError,'mutually exclusive'):
            RTIMageFlow(base,1,3,dual_timestep=True)
        with self.assertRaises(ValueError):
            RTIMageFlow(base,0,4)
        m=RTIMageFlow(base,1,3)
        with self.assertRaisesRegex(ValueError,'mutually exclusive'):
            m(None,None,None,second_timestep=torch.tensor([.5]))


if __name__=='__main__':
    unittest.main()
