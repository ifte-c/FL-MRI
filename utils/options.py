# TODO: Early stopping
# TODO: Adjust batch size
# Consider testing Adam optimizer with reset state and loaded state
# Show histograms before and after standardization
# NOTE: Different loss functions for local/global??

import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--rounds', type=int, default=10, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=100, help="number of users: K")
    parser.add_argument('--shard_per_user', type=int, default=2, help="classes per user")
    parser.add_argument('--frac', type=float, default=0.1, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=5, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=10, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=128, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
    parser.add_argument('--split', type=str, default='user', help="train-test split type, user or sample")
    parser.add_argument('--grad_norm', action='store_true', help='use_gradnorm_avging')
    parser.add_argument('--local_ep_pretrain', type=int, default=0, help="the number of pretrain local ep")
    parser.add_argument('--lr_decay', type=float, default=1.0, help="learning rate decay per round")

    # model arguments
    parser.add_argument('--model', type=str, default='unet', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9, help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to use for convolution')
    parser.add_argument('--norm', type=str, default='batch_norm', help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32, help="number of filters for conv nets")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than strided convolutions")
    parser.add_argument('--num_layers_keep', type=int, default=1, help='number layers to keep')

    # other arguments
    parser.add_argument('--dataset', type=str, default='cardiac', help="name of dataset")
    parser.add_argument('--iid', action='store_true', help='whether i.i.d or not')
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=3, help="number of channels of imges")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--stopping_rounds', type=int, default=10, help='rounds of early stopping')
    parser.add_argument('--verbose', action='store_true', help='verbose print')
    parser.add_argument('--print_freq', type=int, default=100, help="print loss frequency during training")
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--test_freq', type=int, default=1, help='how often to test on val set')
    parser.add_argument('--load_fed', type=str, default='', help='define pretrained federated model path')
    parser.add_argument('--results_save', type=str, default='/', help='define fed results save folder')
    parser.add_argument('--start_saving', type=int, default=0, help='when to start saving models')

    # mri arguments
    parser.add_argument('--centres', nargs='*', type=int, default=[0, 1, 2], help="centres to sample MRI from and train on - 0: ACDC, 1: Vall d'Hebron, 2: Sagrada Familia, 4: SantPau")
    parser.add_argument('--target_shape', nargs='*', type=int, default=[176, 176, 6], help="target shape to resample MRI to")
    parser.add_argument('--k', type=int, default=5, help="number of folds - one fold is used for testing and the rest for training")
    parser.add_argument('--use_cache', type=str, default='False', help="Use existing preprocessed data or overwrite it with new data")

    args = parser.parse_args()
    return args
