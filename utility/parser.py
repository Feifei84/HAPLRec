import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="LightGCN")

    parser.add_argument("--seed", type=int, default=2023, help="random seed for init")

    parser.add_argument("--dataset", nargs="?", default="ml100k", help="[ml100k,yelp,lastfm]")

    parser.add_argument("--data_path", nargs="?", default="./data/", help="Input data path.")

    parser.add_argument('--epochs', type=int, default=2000, help='number of epochs')

    parser.add_argument('--batch_size', type=int, default=2048, help='batch size')

    parser.add_argument('--GCNLayer', type=int, default=3, help="the layer number of GCN")

    parser.add_argument('--layer_size', nargs='?', default='[64,64,64]', help='Output sizes of every layer')

    parser.add_argument('--test_batch_size', type=int, default=100, help='batch size')

    parser.add_argument('--dim', type=int, default=64, help='embedding size')

    parser.add_argument('--l2', type=float, default=1e-4, help='l2 regularization weight')

    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')

    parser.add_argument("--cuda", type=bool, default=True, help="use gpu or not")

    parser.add_argument("--gpu_id", type=int, default=0, help="gpu id")

    parser.add_argument('--topK', nargs='?', default='[20]', help='size of Top-K')

    parser.add_argument('--test_flag', nargs='?', default='part', help='Specify the test type from {part, full}')

    parser.add_argument("--verbose", type=int, default=5, help="Test interval")

    parser.add_argument("--multicore", type=int, default=0, help="use multiprocessing or not in test")

    parser.add_argument("--sparsity", type=bool, default=False)

    return parser.parse_args()
