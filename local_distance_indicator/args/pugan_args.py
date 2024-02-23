import argparse
from args.utils import str2bool


def parse_pugan_args():
    parser = argparse.ArgumentParser(prog="pugan")
    parser.add_argument('--exp_name', default='exp', type=str)
    # seed
    parser.add_argument('--seed', default=21, type=float, help='seed')
    # optimizer
    parser.add_argument('--optim', default='adam', type=str, help='optimizer, adam or sgd')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--weight_decay', default=0, type=float, help='weight decay')
    # lr scheduler
    parser.add_argument('--scheduler_type', default='cosine', type=str, help='step or cosine; type of learning rate scheduler')
    parser.add_argument('--warm_up_end', default=250, type=int, help='warm up end epoch of cosine scheduler')
    parser.add_argument('--lr_decay_step', default=20, type=int, help='learning rate decay step size')
    parser.add_argument('--gamma', default=0.5, type=float, help='gamma for scheduler_steplr')
    # dataset
    parser.add_argument('--dataset', default='pugan', type=str, help='pu1k or pugan')
    parser.add_argument('--h5_file_path', default="./data/PU-GAN/train/PUGAN_poisson_256_poisson_1024.h5", type=str, help='the path of train dataset')
    parser.add_argument('--num_points', default=256, type=int, help='the points number of each input patch')
    parser.add_argument('--skip_rate', default=1, type=int, help='used for dataset')
    parser.add_argument('--use_random_input', default=True, type=str2bool, help='whether use random sampling for input generation')
    parser.add_argument('--jitter_sigma', type=float, default=0.01, help="jitter augmentation")
    parser.add_argument('--jitter_max', type=float, default=0.03, help="jitter augmentation")
    # train
    parser.add_argument('--epochs', default=60, type=int, help='training epochs')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--num_workers', default=4, type=int, help='workers number')
    parser.add_argument('--print_rate', default=200, type=int, help='loss print frequency in each epoch')
    parser.add_argument('--save_rate', default=10, type=int, help='model save frequency')
    parser.add_argument('--use_smooth_loss', default=False, type=str2bool, help='whether use smooth L1 loss')
    parser.add_argument('--beta', default=0.01, type=float, help='beta for smooth L1 loss')
    # model
    parser.add_argument('--k', default=16, type=int, help='neighbor number')
    parser.add_argument('--up_rate', default=4, type=int, help='upsampling rate')
    parser.add_argument('--block_num', default=3, type=int, help='dense block number in the feature extractor')
    parser.add_argument('--layer_num', default=3, type=int, help='dense layer number in each dense block')
    parser.add_argument('--feat_dim', default=32, type=int, help='input(output) feature dimension in each dense block' )
    parser.add_argument('--bn_size', default=1, type=int, help='the factor used in the bottleneck layer')
    parser.add_argument('--growth_rate', default=32, type=int, help='output feature dimension in each dense layer')
    # query points
    parser.add_argument('--local_sigma', default=0.02, type=float, help='used for sample points')
    # truncate distance
    parser.add_argument('--truncate_distance', default=False, type=str2bool, help='whether truncate distance')
    parser.add_argument('--max_dist', default=0.2, type=float, help='the maximum point-to-point distance')
    # ouput
    parser.add_argument('--out_path', default='./output', type=str, help='the checkpoint and log save path')
    # test
    parser.add_argument('--num_iterations', default=10, type=int, help='the number of update iterations')
    parser.add_argument('--test_step_size', default=50, type=float, help='predefined test step size')
    parser.add_argument('--test_input_path', default='./data/PU-GAN/test_pointcloud/input_2048_4X/input_2048/', type=str, help='the test input data path')
    parser.add_argument('--ckpt_path', default='./pretrained_model/pugan/ckpt/ckpt-epoch-60.pth', type=str, help='the pretrained model path')
    parser.add_argument('--patch_rate', default=3, type=int, help='used for patch generation')
    parser.add_argument('--save_dir', default='pcd', type=str, help='save upsampled point cloud')
    parser.add_argument('--double_4X', default=False, type=str2bool, help='conduct 4X twice to get 16X')

    # global
    parser.add_argument('--conf', type=str, default='./confs/pugan.conf')
    parser.add_argument('--mode', type=str, default='train or upsample')
    parser.add_argument('--dir', type=str, default='')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--dataname', type=str, default='')
    parser.add_argument('--listname', type=str, default='pugan.txt')

    args = parser.parse_args()

    return args
