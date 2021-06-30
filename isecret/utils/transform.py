# -*- coding: utf-8 -*-
import torch

def reshape(imgs):
        return imgs


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        if len(self.mean) > 1:
            for t, m, s in zip(tensor, self.mean, self.std):
                t.mul_(s).add_(m)
                # The normalize code -> t.sub_(m).div_(s)
        else:
            tensor.mul_(self.std[0]).add_(self.mean[0])
        return tensor

class StringTensor(object):
    def __init__(self):
        pass

    def transform(self, input: str):
        tensor = None
        for one_input in input:
            if tensor is None:
                tensor = torch.Tensor([ord(c) for c in one_input]).unsqueeze(dim=0)
            else:
                tensor = torch.cat((tensor, torch.Tensor([ord(c) for c in one_input]).unsqueeze(dim=0)), dim=0)
        return tensor

    def inverse_transform(self, input):
        print(input)
        string_list = []
        for one_input in input:
            string_list += [str(chr(c) for c in one_input.cpu().tolist())]
        return string_list

