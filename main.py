import argparse
from train import MGSAL
def get_argument_parser():

    parser = argparse.ArgumentParser(description='Ours')
    # /data/s2023028006/data/hashing/dataset
    # D:\ProgramingFiles\python\Cross - Modal Hashing\MITH - main\dataset
    parser.add_argument('--dataset_root_path', default=r"D:\Users\24226\Desktop\papper\2\program\NEW\MLRH-3\dataset", type=str, help='')
    parser.add_argument('--dataset', default="flickr25k", type=str, help='')
    parser.add_argument('--batch_size', default=64, type=int, help='Size of a training mini-batch.')
    parser.add_argument("--k_bits", type=int, default=32, help="length of hash codes.")
    parser.add_argument("--device", type=int, default=0, help="device")
    parser.add_argument("--num_workers", type=int, default=0, help="num_workers")
    parser.add_argument("--save_dir", type=str, default="path", help="Saved model folder")
    parser.add_argument("--TRAIN", type=bool, default="True", help="is TRAIN")
    parser.add_argument("--iter", type=int, default=1, help="Print loss")
    # parser.add_argument("--warmup", type=int, default=2, help="")

    parser.add_argument("--IterWarmup", type=int, default=60  , help="")

    parser.add_argument("--nearK", type=int, default=16, help="low sample k")
    parser.add_argument("--b", type=int, default=0.4, help="f")
    parser.add_argument("--s_intra", type=int, default=0.6, help="similarity matrix S")
    parser.add_argument("--s_inter", type=int, default=0.3, help="similarity matrix S")

    parser.add_argument("--i", type=int, default=0.4, help="similarity matrix S")
    parser.add_argument("--t", type=int, default=0.4, help="similarity matrix S")
    parser.add_argument("--it", type=int, default=0.1, help="similarity matrix S")
    parser.add_argument("--c", type=int, default=0.1, help="similarity matrix S")
    parser.add_argument("--mse", type=int, default=0.1, help="loss_mse")

    parser.add_argument("--scal_l1", type=int, default=2, help="S scal")

    parser.add_argument("--scal_l2", type=int, default=4, help="S scal")

    parser.add_argument("--query_num", type=int, default=2000)
    parser.add_argument("--train_num", type=int, default=5000)

    return parser


def main():
    parser = get_argument_parser()
    opt = parser.parse_args()
    # print("warmup:",opt.warmup)
    print("IterWarmup:", opt.IterWarmup)
    print("UseDevice:", opt.device)
    print("UseBit:", opt.k_bits)
    print("UseBs:", opt.batch_size)
    print("nearK:", opt.nearK)
    print("s_intra:", opt.s_intra)
    print("s_inter:", opt.s_inter)


    Model = MGSAL(opt)

    epoch=30
    for epoch in range(epoch):
        Model.train(epoch)
    if opt.TRAIN == False:
        Model.load_checkpoints()
        Model.eval(epoch)
        # Model.test(epoch, test=True)
        # Model.evalue500()


if __name__ == '__main__':
    main()
