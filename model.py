import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.backcompat import keepdim_warning

from layers import GraphConvolution,GCNet_IMG,PositionalEncoding
from torch.nn import TransformerEncoder, TransformerEncoderLayer, init
from data.model.clip_model.model import  Transformer


class HashingModel(nn.Module):
    """
    Hashing model
    """
    def __init__(self,opt, clip_info=None):
        super().__init__()
        self.batch_size=opt.batch_size
        self.opt=opt
        num_layers, self.token_size, nhead = 2, 1024, 4
        self.FuseTrans = FuseTransEncoder(num_layers, self.token_size, nhead).to(0)


        self.feat_lens=512
        self.device=0
        self.nbits=opt.k_bits
        #
        # self.ImageMlp = ImageMlp(self.feat_lens, self.nbits).to(self.device)
        # self.TextMlp = TextMlp(self.feat_lens, self.nbits).to(self.device)

        self.ImageMlp =ImgNet(self.feat_lens, self.nbits).to(self.device)
        self.TextMlp = TxtNet(self.feat_lens, self.nbits).to(self.device)

    def forward(self, img_tokens, txt_tokens, img_cls, txt_eos, key_padding_mask):
        output_dict = {}
        img_tokens=img_tokens.transpose(1,0)
        txt_tokens=txt_tokens.transpose(1,0)


        img_tokens_fu,txt_tokens_fu,all_fu=self.FuseTrans(img_tokens,txt_tokens)

        output_dict['img_tokens_fu'] = img_tokens_fu
        output_dict['txt_tokens_fu'] = txt_tokens_fu
        output_dict['img_cls'] = img_cls
        output_dict['txt_eos'] = txt_eos

        img_embedding = img_tokens_fu*(1-self.opt.b) + img_cls*self.opt.b
        text_embedding = txt_tokens_fu*(1-self.opt.b) + txt_eos*self.opt.b

        img_embedding = F.normalize(img_embedding, dim=-1)
        text_embedding = F.normalize(text_embedding, dim=-1)





        S = self.similarity_S(img_cls, txt_eos,img_tokens_fu,txt_tokens_fu,all_fu)
        # S = S*self.opt.scal

        # I = torch.eye(S.size(0)).to(S.device)
        # S = S * (1 - I) + I

        output_dict['S'] = S


        output_dict['img_embedding'] = img_embedding
        output_dict['text_embedding'] = text_embedding

        d_img_token_Mlp = self.ImageMlp(img_embedding)
        d_txt_token_Mlp = self.TextMlp(text_embedding)

        output_dict['d_img_idk_Mlp'] = d_img_token_Mlp
        output_dict['d_txt_idk_Mlp'] = d_txt_token_Mlp

        return output_dict



    def similarity_S(self, img_cls, txt_eos,img_region, txt_region,all_region):
        img_cls = F.normalize(img_cls, dim=1)
        txt_eos = F.normalize(txt_eos, dim=1)
        img_region= F.normalize(img_region, dim=1)
        txt_region= F.normalize(txt_region, dim=1)

        sigma = 1.
        n_img = img_cls.size(0)
        n_cap = txt_eos.size(0)
        with torch.no_grad():
            batch_sim_t2t = torch.matmul(txt_eos, txt_eos.t())
            batch_sim_i2i = torch.matmul(img_cls, img_cls.t())
            batch_sim_i2t = torch.matmul(batch_sim_t2t,batch_sim_i2i.t())
            batch_sim_t2i = torch.matmul(batch_sim_i2i,batch_sim_t2t.t())

            batch_sim_all = torch.matmul(all_region, all_region.t())

            region_i2i= torch.matmul(img_region, img_region.t())
            region_t2t = torch.matmul(txt_region, txt_region.t())

            # # Boolean type matrix returned by logical judgment
            batch_t2t_connect = (batch_sim_t2t - batch_sim_t2t.topk(k=int(n_cap * self.opt.s_intra), dim=1, largest=True)[
                                                     0][:, -1:]) >= 0
            batch_i2i_connect = (batch_sim_i2i - batch_sim_i2i.topk(k=int(n_img * self.opt.s_intra), dim=1, largest=True)[
                                                     0][:, -1:]) >= 0
            k = int(n_img * self.opt.s_inter)
            if k <= 0:
                k = 1
            batch_i2t_connect = (batch_sim_i2t - batch_sim_i2t.topk(k=k, dim=1, largest=True)[0][:, -1:]) >= 0
            batch_t2i_connect = (batch_sim_t2i - batch_sim_t2i.topk(k=k, dim=1, largest=True)[0][:, -1:]) >= 0
            batch_all_connect = (batch_sim_all - batch_sim_all.topk(k=k, dim=1, largest=True)[0][:, -1:]) >= 0

        mask = batch_t2t_connect * batch_i2i_connect
        batch_i2i_relation = torch.exp(-torch.cdist(img_region, img_region) / sigma) *batch_i2i_connect
        batch_t2t_relation = torch.exp(-torch.cdist(txt_region, txt_region) / sigma) * batch_t2t_connect
        batch_i2t_relation = torch.exp(-torch.cdist(region_i2i, region_t2t) / sigma) * batch_i2t_connect
        batch_t2i_relation = torch.exp(-torch.cdist(region_t2t, region_i2i) / sigma) * batch_t2i_connect
        batch_all_relation = torch.exp(-torch.cdist(all_region,all_region) / sigma) * batch_all_connect


        S = batch_i2i_relation * self.opt.i + batch_t2t_relation * self.opt.t+ (batch_i2t_relation * 0.5 + batch_t2i_relation * 0.5) *  self.opt.it + batch_all_relation * self.opt.c
        return  S






class FuseTransEncoder(nn.Module):
    def __init__(self, num_layers, hidden_size, nhead):  # num_layers, self.token_size, nhead = 2, 1024, 4
        super(FuseTransEncoder, self).__init__()
        # encoder_layer = TransformerEncoderLayer(d_model=hidden_size, nhead=nhead, batch_first=False)
        encoder_layer = TransformerEncoderLayer(d_model=hidden_size, nhead=nhead)
        self.transformerEncoder = TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers)
        self.d_model = hidden_size
        self.sigal_d = int(self.d_model / 2)

    def forward(self, img, txt):  # torch.Size([1, 128, 1024])

        img_glo_nor=F.normalize(img.mean(dim=1,keepdim=True),dim=-1)
        txt_glo_nor=F.normalize(txt.mean(dim=1,keepdim=True),dim=-1)

        img_self_attn = (img_glo_nor*img).sum(dim=-1)
        img_cross_attn = (txt_glo_nor * img).sum(dim=-1)

        txt_self_attn = (txt_glo_nor * txt).sum(dim=-1)
        txt_cross_attn = (img_glo_nor * txt).sum(dim=-1)

        img_attn =img_self_attn + img_cross_attn
        img_attn_f=F.normalize(img_attn,dim=-1)

        txt_attn = txt_self_attn  +  txt_cross_attn
        txt_attn_f = F.normalize(txt_attn, dim=-1)

        img_tokens =img * img_attn_f.unsqueeze(-1)
        img_attn_token=img_tokens.mean(dim=1)

        txt_tokens =txt* txt_attn_f.unsqueeze(-1)
        txt_attn_token = txt_tokens.mean(dim=1)


        img_tokens_mean = img.mean(dim=1)
        txt_tokens_mean = txt.mean(dim=1)

        a=0.1
        img=img_tokens_mean*a+ img_attn_token*(1-a)
        txt = txt_tokens_mean*a+ txt_attn_token*(1-a)

        temp_tokens = torch.cat((img, txt), dim=1)  # torch.Size([128, 1024])
        tokens = temp_tokens.unsqueeze(0)  # torch.Size([1, 128, 1024])
        encoder_X = self.transformerEncoder(tokens)  # torch.Size([1, 128, 1024])
        encoder_X_r = encoder_X.reshape(-1, self.d_model)  # torch.Size([128, 1024])
        # encoder_X_r = F.normalize(encoder_X_r, p=2, dim=-1)
        encoder_X_r = F.normalize(encoder_X_r,dim=-1)
        img, txt = encoder_X_r[:, :self.sigal_d], encoder_X_r[:, self.sigal_d:]
        return img, txt,encoder_X_r





class ImageMlp(nn.Module):
    def __init__(self, input_dim, hash_lens, dim_feedforward=[1024, 128, 1024], dropout=0.1):  # input_dim=512
        super(ImageMlp, self).__init__()
        self.fc1 = nn.Linear(input_dim, 4096)
        self.relu = nn.ReLU(inplace=True)
        self.dp = nn.Dropout(0.3)
        self.tohash = nn.Linear(4096, hash_lens)
        self.tanh = nn.Tanh()
    def _ff_block(self, x):
        x = F.normalize(x, p=2, dim=1)
        feat = self.relu(self.fc1(x))
        hid = self.tohash(self.dp(feat))
        out = self.tanh(hid)
        return out

    def forward(self, X):  # torch.Size([128, 512])
        mlp_output = self._ff_block(X)
        mlp_output = F.normalize(mlp_output, p=2, dim=1)
        return mlp_output

class TextMlp(nn.Module):
    def __init__(self, input_dim, hash_lens, dim_feedforward=[1024, 128, 1024], dropout=0.1):
        super(TextMlp, self).__init__()
        self.fc1 = nn.Linear(input_dim, 4096)
        self.relu = nn.ReLU(inplace=True)
        self.dp = nn.Dropout(0.3)
        self.tohash = nn.Linear(4096, hash_lens)
        self.tanh = nn.Tanh()

    def _ff_block(self, x):
        x = F.normalize(x, p=2, dim=1)
        feat = self.relu(self.fc1(x))
        hid = self.tohash(self.dp(feat))
        out = self.tanh(hid)
        return out

    def forward(self, X):
        mlp_output = self._ff_block(X)
        mlp_output = F.normalize(mlp_output, p=2, dim=1)
        return mlp_output






class ImgNet(nn.Module):
    def __init__(self, img_feat_len,code_len):
        super(ImgNet, self).__init__()
        self.fc1 = nn.Linear( img_feat_len, 4096)
        self.fc_encode = nn.Linear(4096, code_len)

        self.alpha = 1.0
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU(inplace=True)
        self.init_weights()
    def init_weights(self):

        init.xavier_normal_(self.fc1.weight)
        init.xavier_normal_(self.fc_encode.weight)

        if self.fc1.bias is not None:
            init.constant_(self.fc1.bias, 0)
        if self.fc_encode.bias is not None:
            init.constant_(self.fc_encode.bias, 0)
    def forward(self, x):

        x = x.view(x.size(0), -1).float()
        feat1 = self.relu(self.fc1(x))
        hid = self.fc_encode(self.dropout(feat1))
        code = torch.tanh(self.alpha * hid)

        return code

    def set_alpha(self, epoch):
        self.alpha = math.pow((1.0 * epoch + 1.0), 0.5)

class TxtNet(nn.Module):
    def __init__(self,  txt_feat_len,code_len):
        super(TxtNet, self).__init__()
        self.fc1 = nn.Linear(txt_feat_len, 4096)
        self.fc_encode = nn.Linear(4096, code_len)

        self.alpha = 1.0
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU(inplace=True)
        torch.nn.init.normal(self.fc_encode.weight, mean=0.0, std=0.3)

        # self.init_weights()   #coco时屏蔽掉  25k nus时保留
    def init_weights(self):

        init.xavier_normal_(self.fc1.weight)
        init.xavier_normal_(self.fc_encode.weight)

        if self.fc1.bias is not None:
            init.constant_(self.fc1.bias, 0)
        if self.fc_encode.bias is not None:
            init.constant_(self.fc_encode.bias, 0)
    def forward(self, x):
        feat = self.relu(self.fc1(x))
        hid = self.fc_encode(self.dropout(feat))
        code = torch.tanh(self.alpha * hid)

        return code

    def set_alpha(self, epoch):
        self.alpha = math.pow((1.0 * epoch + 1.0), 0.5)


def gc2(emb, k):
    S = emb.mm(emb.t())
    if k > S.size(0):
        k = S.size(0)
    topk_values, topk_indices = torch.topk(S, k, dim=1)

    result = torch.zeros_like(S)
 
    result.scatter_(1, topk_indices, topk_values)

    mask = 1 - torch.eye(S.size(0)).to(S.device)
    result = result * mask

    return result



def weighted(img_embedding,text_embedding,k):
    with torch.no_grad():

        gc2_I = gc2(img_embedding, k=k)
        gc2_T = gc2(text_embedding, k=k)
        gc = gc2_I * gc2_T
        w = gc.sum(dim=-1)
        Best = w > 0
        # print( torch.sum(Best))
        w_0 = torch.where(Best, 1, 0)
        w_weight = w
        return w_weight , w_0




