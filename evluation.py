import numpy as np
import torch
import matplotlib.pyplot as plt





def calc_neighbor(a: torch.Tensor, b: torch.Tensor): #（10000 24） （bs 24）
    return (a.matmul(b.transpose(0, 1)) > 0).float()


def calc_hamming_dist(B1, B2):
    q = B2.shape[1]
    if len(B1.shape) < 2:
        B1 = B1.unsqueeze(0)
    distH = 0.5 * (q - B1.mm(B2.t()))
    return distH

def calc_map_k(qB, rB, query_label, retrieval_label, k=None):
    num_query = query_label.shape[0]#5000
    map = 0.
    if k is None:
        k = retrieval_label.shape[0]
    for i in range(num_query):
        gnd = (query_label[i].unsqueeze(0).mm(retrieval_label.t()) > 0).type(torch.float).squeeze()
        tsum = torch.sum(gnd)
        if tsum == 0:
            continue
        hamm = calc_hamming_dist(qB[i, :], rB)
        _, ind = torch.sort(hamm)
        ind.squeeze_()
        gnd = gnd[ind]
        total = min(k, int(tsum))
        count = torch.arange(1, total + 1).type(torch.float).to(gnd.device)#1到4729步长为1
        tindex = torch.nonzero(gnd)[:total].squeeze().type(torch.float) + 1.0  #gand中不为0的索引。
        map += torch.mean(count / tindex)
    map = map / num_query
    return map

def pr_curve(query_code, retrieval_code, query_targets, retrieval_targets, device):
    """
    P-R curve.
    Args
        query_code(torch.Tensor): Query hash code.
        retrieval_code(torch.Tensor): Retrieval hash code.
        query_targets(torch.Tensor): Query targets.
        retrieval_targets(torch.Tensor): Retrieval targets.
        device (torch.device): Using CPU or GPU.

    Returns
        P(torch.Tensor): Precision.
        R(torch.Tensor): Recall.
    """
    num_query = query_code.shape[0]
    num_bit = query_code.shape[1]
    P = torch.zeros(num_query, num_bit + 1).to(device)
    R = torch.zeros(num_query, num_bit + 1).to(device)
    for i in range(num_query):
        gnd = (query_targets[i].unsqueeze(0).mm(retrieval_targets.t()) > 0).float().squeeze()
        tsum = torch.sum(gnd)
        if tsum == 0:
            continue
        hamm = 0.5 * (retrieval_code.shape[1] - query_code[i, :] @ retrieval_code.t())
        tmp = (hamm <= torch.arange(0, num_bit + 1).reshape(-1, 1).float().to(device)).float()
        total = tmp.sum(dim=-1)
        total = total + (total == 0).float() * 0.1
        t = gnd * tmp
        count = t.sum(dim=-1)
        #
        # count =count.cpu().numpy()
        # total =total.cpu().numpy()
        # tsum=tsum.cpu().numpy()
        p = count / total
        r = count / tsum
        P[i] = p
        R[i] = r
    mask = (P > 0).float().sum(dim=0)
    mask = mask + (mask == 0).float() * 0.1
    P = P.sum(dim=0) / mask
    R = R.sum(dim=0) / mask
    P =P.cpu().numpy()
    R =R.cpu().numpy()
    # print("P :",P )
    # print("R :",R)
    # print("draw")
    # 画 P-R 曲线
    # fig = plt.figure(figsize=(5, 5))
    # plt.plot(R, P, marker='^', linestyle='-', color='red', markersize=10, markerfacecolor='none', markeredgecolor='black')
    # plt.grid(True)
    # plt.xlim(0, 1)
    # plt.ylim(0, 1)
    # plt.xlabel('recall')
    # plt.ylabel('precision')
    # # plt.legend()
    # plt.show()


def get_code( model,data_loader,k_bits, device,length: int):
    k_bits = k_bits
    rank=0
    img_buffer = torch.empty(length, k_bits, dtype=torch.float).to(device)
    text_buffer = torch.empty(length, k_bits, dtype=torch.float).to(device)

    with torch.no_grad():
        for image, text, key_padding_mask, label, index in data_loader:

            image = image.float().to(rank, non_blocking=True)
            text = text.to(rank, non_blocking=True)
            label= label.to(rank, non_blocking=True)
            key_padding_mask = key_padding_mask.to(rank, non_blocking=True)
            index = index.numpy()
            output_dict = model(image, text, key_padding_mask)


            d_img_idk_Mlp = output_dict['d_img_idk_Mlp']
            d_txt_idk_Mlp = output_dict['d_txt_idk_Mlp']

            img_embedding_Mlp = torch.sign(d_img_idk_Mlp)
            text_embedding_Mlp = torch.sign(d_txt_idk_Mlp)

            # if max(index)>10000:
            #     print(index)

            img_buffer[index, :] = img_embedding_Mlp
            text_buffer[index, :] = text_embedding_Mlp

    return img_buffer, text_buffer