import torch
import torch.nn.functional as F
import torch.nn as nn

class ContrastiveLoss(nn.Module):
    def __init__(self,batch_size, device='cuda:0', temperature=0.5):
        super(ContrastiveLoss, self).__init__()
        self.batch_size = batch_size
        self.device = device
        self.register_buffer("temperature", torch.tensor(temperature).to(device))

    def forward(self, emb_i, emb_j,loss_emb=None):
        self.batch_size = emb_i.shape[0]
        self.negatives_mask = (~torch.eye(self.batch_size * 2, self.batch_size * 2, dtype=bool).to(self.device)).float()
        z_i = F.normalize(emb_i, dim=1)     # (bs, dim)  --->  (bs, dim)
        z_j = F.normalize(emb_j, dim=1)
        # (bs, dim)  --->  (bs, dim)
        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
        sim_ij = torch.diag(similarity_matrix, self.batch_size)         # bs
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)        # bs
        positives = torch.cat([sim_ij, sim_ji], dim=0)                  # 2*bs
        nominator = torch.exp(positives / self.temperature)             # 2*bs
        denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)             # 2*bs, 2*bs
        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))        # 2*bs
        num=torch.sum(loss_partial.detach()) / (2 * self.batch_size)

        if loss_emb!=None:
            w = loss_emb
            loss_w = torch.softmax(-w / 1, dim=0) * w.size(0)
            loss_w = torch.where(loss_w > 1.0, loss_w, torch.ones_like(loss_w))
            loss_partial=loss_partial * loss_w.detach()
            #args = torch.argsort(loss_emb)
            #loss_partial_sort=loss_partial[args]
            # loss = torch.sum(loss_partial_sort) / (torch.sum(loss_w.detach()))
            loss = torch.sum(loss_partial) / (torch.sum(loss_w.detach()))
        else:
            loss = torch.sum(loss_partial) / (2 * self.batch_size)

        return loss,loss_partial,num

class Weighted_mse_loss(nn.Module):
    def __init__(self,opt, device='cuda:0', temperature=1.0):
        super(Weighted_mse_loss, self).__init__()
        self.opt=opt
        self.device = device
        self.temperature=temperature

    def forward(self, H, S, weight, w_0,temperature = 1.0):

        weight_S= torch.softmax(weight / 1.0, dim=-1) * weight.size(0)
        weight_w  = weight_S.unsqueeze(1) * weight_S.unsqueeze(0)  # (bs, bs)
        weight_0 = w_0.unsqueeze(1) * w_0.unsqueeze(0)
        weight = weight_w* weight_0
        weight_matrix = 1 / (1 + torch.exp(-weight.detach()))

        weighted_loss1 = (H - S.detach()*self.opt.scal_l1).pow(2) * weight_matrix   # (bs, bs)
        valid_loss1 = weighted_loss1.sum() / (weight_matrix.sum())  # 归一化
        # weighted_loss1 = (H - S.detach()).pow(2).mean()


        valid_loss2 = (H - S*self.opt.scal_l2).pow(2).mean()

        valid_loss = valid_loss1 * self.opt.mse + valid_loss2 * (1-self.opt.mse)



        return valid_loss
