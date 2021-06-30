import torch.nn as nn
import torch.nn.functional as F
import torch
from abc import ABC, abstractmethod

class GANLoss(ABC):
    def __init__(self, args):
        self.args = args

    @ abstractmethod
    def update_g(self, good_dis, fake_good, bad_dis=None, fake_bad=None):
        pass

    @abstractmethod
    def update_d(self, good_dis, real_good, fake_good, bad_dis=None, real_bad=None, fake_bad=None):
        pass


class NaiveGANLoss(GANLoss):
    def __init__(self,  args):
        GANLoss.__init__(self, args)
        self.real_loss = lambda x: F.binary_cross_entropy_with_logits\
            (x, torch.ones_like(x))
        self.fake_loss = lambda x: F.binary_cross_entropy_with_logits\
            (x, torch.zeros_like(x))

    def update_g(self, good_dis, fake_good, bad_dis=None, fake_bad=None):
        fake_good_logits = good_dis(fake_good)
        good_loss = self.real_loss(fake_good_logits)
        if bad_dis is None:
            return good_loss
        fake_bad_logits = bad_dis(fake_bad)
        bad_loss = self.real_loss(fake_bad_logits)
        return good_loss, bad_loss

    def update_d(self, good_dis, real_good, fake_good, bad_dis=None, real_bad=None, fake_bad=None):
        # Train dis_good
        real_good_logits = good_dis(real_good)
        fake_good_logits = good_dis(fake_good.detach())
        real_good_loss = self.real_loss(real_good_logits)
        fake_good_loss = self.fake_loss(fake_good_logits)
        good_dis_loss = (real_good_loss + fake_good_loss) / 2
        if bad_dis is None:
            return good_dis_loss

        # Train dis_bad
        real_bad_logits = bad_dis(real_bad)
        fake_bad_logits = bad_dis(fake_bad.detach())
        real_bad_loss = self.real_loss(real_bad_logits)
        fake_bad_loss = self.fake_loss(fake_bad_logits)
        bad_dis_loss = (real_bad_loss + fake_bad_loss) / 2

        return good_dis_loss, bad_dis_loss


class LSGANLoss(nn.Module):
    def __init__(self, args):
        GANLoss.__init__(self, args)
        self.real_loss = lambda x: F.mse_loss(x, torch.ones_like(x))
        self.fake_loss = lambda x: F.mse_loss(x, torch.zeros_like(x))

    def update_g(self, good_dis, fake_good, bad_dis=None, fake_bad=None):
        fake_good_logits = good_dis(fake_good)
        good_loss = self.real_loss(fake_good_logits)
        if bad_dis is None:
            return good_loss
        fake_bad_logits = bad_dis(fake_bad)
        bad_loss = self.real_loss(fake_bad_logits)
        return good_loss, bad_loss

    def update_d(self, good_dis, real_good, fake_good, bad_dis=None, real_bad=None, fake_bad=None):
        # Train dis_good
        real_good_logits = good_dis(real_good)
        fake_good_logits = good_dis(fake_good.detach())
        real_good_loss = self.real_loss(real_good_logits)
        fake_good_loss = self.fake_loss(fake_good_logits)
        good_dis_loss = (real_good_loss + fake_good_loss) / 2
        if bad_dis is None:
            return good_dis_loss
        # Train dis_bad
        real_bad_logits = bad_dis(real_bad)
        fake_bad_logits = bad_dis(fake_bad.detach())
        real_bad_loss = self.real_loss(real_bad_logits)
        fake_bad_loss = self.fake_loss(fake_bad_logits)
        bad_dis_loss = (real_bad_loss + fake_bad_loss) / 2
        return good_dis_loss, bad_dis_loss


