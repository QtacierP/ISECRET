import torch 
import torch.nn as nn
from packaging import version

class ICCLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool

    def forward(self, feat_q, feat_k, weight=1.0):
        batchSize = feat_q.shape[0]
        dim = feat_q.shape[1]
        feat_k = feat_k.detach()

        # pos logit
        # Calculate the bmm of the f_q and f_k

        # N x patches X 1 X C
        l_pos = torch.bmm(feat_q.view(batchSize, 1, -1), feat_k.view(batchSize, -1, 1))
        l_pos = l_pos.view(batchSize, 1)

        # neg logit -- current batch
        # reshape features to batch size
        # B x patches x C
        feat_q = feat_q.view(self.args.dist.batch_size_per_gpu, -1, dim)
        feat_k = feat_k.view(self.args.dist.batch_size_per_gpu, -1, dim)
 
        # h x w
        npatches = feat_q.size(1)


        # B x patches x C X B x C x patches -> B x patches x patches
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))
        # diagonal entries are similarity between same features, and hence meaningless.
        # just fill the diagonal with very small number, which is exp(-10) and almost zero
        diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]
        l_neg_curbatch.masked_fill_(diagonal, -10.0)  # Replace the corresponding patch with negative value
        
        # B x patches X patches
        l_neg = l_neg_curbatch.view(-1, npatches)
        out = torch.cat((l_pos, l_neg), dim=1)  / self.args.train.nce_T
        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
                                                        device=feat_q.device)) * weight
  
        return loss