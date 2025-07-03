import time
import os
import scipy.io as scio
import torch
import torch.nn as nn
import torch.nn.functional as F
from data.model.clip_model.model import load_download_clip
from data.load_data import generate_dataset
from torch.utils.data import DataLoader
from model import HashingModel,weighted
from metric import ContrastiveLoss,Weighted_mse_loss
from optimization import BertAdam
from evluation import calc_map_k,get_code,pr_curve


class MITH(nn.Module):
    def __init__(self,opt):
        super(MITH, self).__init__()
        self.clip_path='./cache/ViT-B-32.pt' # help="pretrained clip path."
        self.clip, clip_info = load_download_clip(self.clip_path)
        self.hash = HashingModel(opt,clip_info=clip_info)


    def forward(self, image, text, key_padding_mask):
        # img_tokens(49 bs 512)  img_cls(bs 512)
        img_tokens, _, img_cls = self.clip.encode_image(image)
        # txt_token(32 bs 512)new_key_padding_mask(bs 32) txt_eos(bs 512)
        txt_tokens, _, new_key_padding_mask, txt_eos = self.clip.encode_text(text, key_padding_mask)
        output_dict = self.hash(img_tokens, txt_tokens, img_cls, txt_eos, new_key_padding_mask)
        return output_dict

class MGSAL:
    def __init__(self, opt,):
        self.opt=opt
        dataset = opt.dataset
        index_file = "index.mat"
        caption_file = "caption.mat"
        label_file = "label.mat"
        index_file = os.path.join(opt.dataset_root_path, dataset, index_file)  # './dataset\\flickr25k\\index.mat'
        caption_file = os.path.join(opt.dataset_root_path, dataset, caption_file)  # './dataset\\flickr25k\\caption.mat'
        label_file = os.path.join(opt.dataset_root_path, dataset, label_file)  # './dataset\\flickr25k\\label.mat'
        print(caption_file)

        train_data, query_data, retrieval_data = generate_dataset(captionFile=caption_file,
                                                                  indexFile=index_file,
                                                                  labelFile=label_file,
                                                                  maxWords=32,
                                                                  imageResolution=224,
                                                                  query_num=opt.query_num,
                                                                  train_num=opt.train_num,
                                                                  seed=1)

        self.train_labels = train_data.get_all_label().float()  # (10000,24)
        self.query_labels = query_data.get_all_label().float()  # (5000,24)
        self.retrieval_labels = retrieval_data.get_all_label().float()  # (15015,24)
        retrieval_num = len(self.retrieval_labels)
        self.opt.retrieval_num = retrieval_num

        self.train_loader = DataLoader(
            dataset=train_data,
            batch_size=opt.batch_size,
            num_workers=opt.num_workers,
            pin_memory=False,
            shuffle=True
        )

        self.query_loader = DataLoader(
            dataset=query_data,
            batch_size=opt.batch_size,
            num_workers=opt.num_workers,
            pin_memory=False,
            shuffle=False
        )
        self.retrieval_loader = DataLoader(
            dataset=retrieval_data,
            batch_size=opt.batch_size,
            num_workers=opt.num_workers,
            pin_memory=False,
            shuffle=False
        )


        self.model = MITH(opt).to(opt.device)
        self.model.float()
        clip_lr = 0.000002
        lr = 0.001
        epochs = 30
        warmup_proportion = 0.05  # help="Proportion of training to perform learning rate warmup
        weight_decay = 0.01
        self.optimizer = BertAdam(
            [
                {'params': self.model.clip.parameters(), 'lr': clip_lr},
                {'params': self.model.hash.parameters(), 'lr': lr},
            ],
            lr=lr, warmup=warmup_proportion, schedule='warmup_cosine',
            b1=0.9, b2=0.98, e=1e-6, t_total=len(self.train_loader) * epochs,
            weight_decay=weight_decay, max_grad_norm=1.0
        )
        # self.optimizer = BertAdam(
        #     [
        #         {'params': self.model.clip.parameters(), 'lr': clip_lr},
        #         {'params': self.model.hash.ImageMlp.parameters(), 'lr': lr},
        #         {'params': self.model.hash.TextMlp.parameters(), 'lr': lr},
        #         {'params': self.model.hash.FuseTrans.parameters(), 'lr': 0.0002},
        #     ],
        #     lr=lr, warmup=warmup_proportion, schedule='warmup_cosine',
        #     b1=0.9, b2=0.98, e=1e-6, t_total=len(self.train_loader) * epochs,
        #     weight_decay=weight_decay, max_grad_norm=1.0
        # )
        self.loss_l2 = torch.nn.MSELoss()
        self.closs= ContrastiveLoss(self.opt.batch_size)
        self.weighted_mse_loss=Weighted_mse_loss(self.opt)
        self.max_map = {'i2t': 0, "t2i": 0}
        self.best_epoch = 0
        self.K=False
        self.cra=False
        self.global_step_losses = []  # 保存所有epoch的step loss
        self.global_step_counts = []  # 全局step计数（跨epoch）
        self.iter = 0
    def load_checkpoints(self):
        self.model.load_state_dict(torch.load("path/model.pth", map_location=f"cuda:{self.opt.device}"))
        return self.model

    def save_model(self,datase,epoch):
        # torch.save(self.model.state_dict(), os.path.join(self.opt.save_dir, f"model_{datase}_{epoch}.pth"))
        # timestamp = int(time.time())
        # _{timestamp}
        torch.save(self.model.state_dict(), os.path.join(self.opt.save_dir, f"model_{self.opt.dataset}_{self.opt.k_bits}.pth"))
        # for epoch in range(epochs):


    def train(self,epoch):
            # if epoch >= self.opt.warmup:
            #     print('useMseloss')
            # if epoch >= self.opt.warmup:
            #     print('useCraloss')

            if self.iter >= self.opt.IterWarmup:
                print('useMseloss')
            if self.iter >= self.opt.IterWarmup:
                print('useCraloss')

            self.model.train()
            print("####################### Train epochs: %d #######################" % epoch)
            for image, text, key_padding_mask, label, index in self.train_loader:
                image = image.float().to(self.opt.device, non_blocking=True)
                """
                t=image[1].cpu().permute(1,2,0)
                import matplotlib.pyplot as plt
                plt.imshow(t)
                plt.show()
                """
                label = label.float().to(self.opt.device, non_blocking=True)
                text = text.to(self.opt.device, non_blocking=True)
                key_padding_mask = key_padding_mask.to(self.opt.device, non_blocking=True)
                output_dict = self.model(image, text, key_padding_mask)

                d_img_idk_Mlp = output_dict['d_img_idk_Mlp']
                d_txt_idk_Mlp = output_dict['d_txt_idk_Mlp']

                img_embedding = output_dict['img_embedding']
                text_embedding = output_dict['text_embedding']

                S = output_dict['S']



                F_I = F.normalize(d_img_idk_Mlp, dim=1)
                F_T = F.normalize(d_txt_idk_Mlp, dim=1)

                BI_BI = F_I.mm(F_I.t())
                BT_BT = F_T.mm(F_T.t())
                BI_BT = F_I.mm(F_T.t())
                BT_BI = F_T.mm(F_I.t())

                loss1 ,loss_emb,loss_num1= self.closs(img_embedding, text_embedding)
                loss3=0


                # if (epoch + 1) >= self.opt.warmup:
                if self.iter >= self.opt.IterWarmup:
                    loss2,loss_partial,loss_num = self.closs(d_img_idk_Mlp, d_txt_idk_Mlp,loss_emb)
                    # loss2, loss_partial = self.closs(d_img_idk_Mlp, d_txt_idk_Mlp)
                else:
                    loss2 ,loss_partial,loss_num= self.closs(d_img_idk_Mlp, d_txt_idk_Mlp)
                # if(epoch+1)>=self.opt.warmup:
                if self.iter >= self.opt.IterWarmup:
                    w_weight,w_0=weighted(img_embedding,text_embedding,self.opt.nearK)
                    loss3 = ((self.weighted_mse_loss(BI_BI, S, w_weight, w_0) + self.weighted_mse_loss(BT_BT, S,w_weight, w_0))+
                             (self.weighted_mse_loss(BI_BT, S, w_weight, w_0) + self.weighted_mse_loss(BT_BI, S, w_weight, w_0)))


                B = torch.sign(d_img_idk_Mlp.detach() + d_txt_idk_Mlp.detach())
                loss6 = F.mse_loss(d_img_idk_Mlp, B) / d_img_idk_Mlp.shape[0] / self.opt.k_bits + F.mse_loss(d_txt_idk_Mlp, B) / d_img_idk_Mlp.shape[0] / self.opt.k_bits
                loss =loss1+loss2+loss3+loss6

                # 保存当前step的loss
                self.global_step_losses.append(loss_num.item())
                self.global_step_counts.append(len(self.global_step_losses))
                self.save_loss_realtime()
                self.iter += 1

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if (epoch+1)>=2 and (epoch+1)%self.opt.iter==0:
                self.eval(epoch)

    def save_loss_realtime(self, save_path="log/loss_log.txt"):
        # 每次记录时追加写入文件
        with open(save_path, 'a') as f:
            f.write(f"{len(self.global_step_losses)},{self.global_step_losses[-1]}\n")

    def eval(self,epoch,test=False):
        print("TEST")
        self.model.eval()
        k_bits = self.opt.k_bits

        retrieval_num = self.opt.retrieval_num
        q_i, q_t = get_code(self.model, self.query_loader, k_bits, self.opt.device, self.opt.query_num)
        r_i, r_t = get_code(self.model, self.retrieval_loader, k_bits,self.opt.device, retrieval_num)
        _k_ = None
        mAPi2t = calc_map_k(q_i.to(self.opt.device), r_t.to(self.opt.device), self.query_labels.to(self.opt.device), self.retrieval_labels.to(self.opt.device),
                            _k_).item()
        mAPt2i = calc_map_k(q_t.to(self.opt.device), r_i.to(self.opt.device), self.query_labels.to(self.opt.device), self.retrieval_labels.to(self.opt.device),
                            _k_).item()
        print("mAPi2t :", mAPi2t)
        print("mAPt2i :", mAPt2i)

        if mAPi2t + mAPt2i > self.max_map['i2t'] + self.max_map['t2i'] and not test:
            self.best_epoch = epoch
            self.max_map['i2t'] = mAPi2t
            self.max_map['t2i'] = mAPt2i
            self.save_model(self.opt.dataset,epoch)
            self.save_mat(q_i, q_t, r_i, r_t)
            # print("PR i2t:")
            pr_curve(q_i.to(self.opt.device), r_t.to(self.opt.device), self.query_labels.to(self.opt.device),self.retrieval_labels.to(self.opt.device), self.opt.device)
            # print("PR t2i:")
            pr_curve(q_t.to(self.opt.device), r_i.to(self.opt.device), self.query_labels.to(self.opt.device),self.retrieval_labels.to(self.opt.device), self.opt.device)
        print("best_epoch :",self.best_epoch)
        print("max_mAPi2t :", self.max_map['i2t'])
        print("max_mAPt2i :", self.max_map['t2i'])

    def save_mat(self, query_img, query_txt, retrieval_img, retrieval_txt):

        save_dir = os.path.join(self.opt.save_dir, "PR_curve")
        os.makedirs(save_dir, exist_ok=True)

        query_img = query_img.cpu().detach().numpy()
        query_txt = query_txt.cpu().detach().numpy()
        retrieval_img = retrieval_img.cpu().detach().numpy()
        retrieval_txt = retrieval_txt.cpu().detach().numpy()
        query_labels = self.query_labels.cpu().detach().numpy()
        retrieval_labels = self.retrieval_labels.cpu().detach().numpy()

        result_dict = {
            'q_img': query_img,
            'q_txt': query_txt,
            'r_img': retrieval_img,
            'r_txt': retrieval_txt,
            'q_l': query_labels,
            'r_l': retrieval_labels
        }

        scio.savemat(
            os.path.join(save_dir, f"MGSAL-" + self.opt.dataset + "-" + str(self.opt.k_bits) + ".mat"),
            result_dict)

    def test(self, epoch, test=False):
        from utils.Search_image import search_code, search_calc_map_k

        print("TEST")
        self.model.eval()
        k_bits = self.opt.k_bits
        retrieval_num = self.opt.retrieval_num
        q_i, q_t, q_index = search_code(self.model, self.query_loader, k_bits, self.opt.device, self.opt.query_num)
        r_i, r_t, r_index = search_code(self.model, self.retrieval_loader, k_bits, self.opt.device, retrieval_num)
        _k_ = None
        self.save_mat(q_i, q_t, r_i, r_t)
        mAPi2t = calc_map_k(q_i.to(self.opt.device), r_t.to(self.opt.device), self.query_labels.to(self.opt.device),
                            self.retrieval_labels.to(self.opt.device),
                            _k_).item()
        mAPt2i = calc_map_k(q_t.to(self.opt.device), r_i.to(self.opt.device), self.query_labels.to(self.opt.device),
                            self.retrieval_labels.to(self.opt.device),
                            _k_).item()
        print("mAPi2t :", mAPi2t)
        print("mAPt2i :", mAPt2i)
        mAPi2t = search_calc_map_k(q_i.to(self.opt.device), r_t.to(self.opt.device),
                                   self.query_labels.to(self.opt.device),
                                   self.retrieval_labels.to(self.opt.device), q_index, r_index, _k_).item()
        mAPt2i = search_calc_map_k(q_t.to(self.opt.device), r_i.to(self.opt.device),
                                   self.query_labels.to(self.opt.device),
                                   self.retrieval_labels.to(self.opt.device), q_index, r_index, _k_).item()

        print("mAPi2t :", mAPi2t)
        print("mAPt2i :", mAPt2i)

    def weighted_mse_loss(self,H, S, weight,w_0):
        weight_S = weight.unsqueeze(1) * weight.unsqueeze(0)  # (bs, bs)
        temperature = 1.0  # 可调整
        W_softmax = torch.softmax(weight_S / temperature, dim=-1)

        W_weight = W_softmax * W_softmax.size(0)
        weight_matrix = W_weight * w_0
        weight_matrix=weight_matrix.detach()

        weighted_loss1 =(H - S.detach()).pow(2) * weight_matrix # (bs, bs)
        valid_loss1 = weighted_loss1.sum() / (weight_matrix.sum())  # 归一化
        valid_loss2 = (H- S).pow(2).mean()

        valid_loss=valid_loss1*0.1+valid_loss2*0.9

        # valid_loss=(input - target).pow(2).mean()

        return valid_loss

