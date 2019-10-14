import argparse

arg_lists = []
parser = argparse.ArgumentParser(description='IsoNN')


def str2bool(v):
    return v.lower() in ('true', '1')


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


# glimpse network params
IsoNN_arg = add_argument_group('FC layers')


IsoNN_arg.add_argument('--hidden1', type=int, default=1024,
                         help='hidden size of fc1')
IsoNN_arg.add_argument('--hidden2', type=int, default=128,
                         help='hidden size of fc2')

IsoNN_arg.add_argument('--nclass', type=int, default=2,
                         help='number of class')


IsoNN_arg.add_argument('--k', type=int, default=4,
                         help='size of kernel1')
IsoNN_arg.add_argument('--c', type=int, default=1,
                         help='size of channels in layer 1')

IsoNN_arg.add_argument('--num_node', type=int, default=90, #hiv/90 bp/82 mutag/28 nci1/111 nci109/111 ptc/109 protein/620
                         help='scale of successive patches')


IsoNN_arg.add_argument('--M', type=float, default=10,
                           help='Monte Carlo sampling for valid and test sets')


IsoNN_arg.add_argument('--batch_size', type=int, default=128,
                      help='# of images in each batch of data')

IsoNN_arg.add_argument('--shuffle', type=str2bool, default=True,
                      help='Whether to shuffle the train and valid indices')



IsoNN_arg.add_argument('--epochs', type=int, default=150,
                       help='# of epochs to train for')
IsoNN_arg.add_argument('--init_lr', type=float, default=1e-3,
                       help='Initial learning rate value')


IsoNN_arg.add_argument('--random_seed', type=int, default=3, # fold1&2(?)&3(fast?)/2, fold2&3/1 HIV_DTI_3_fold   ### BP_fMRI_3_fold fold2/4  fold1&3/1
                      help='Seed to ensure reproducibility')

IsoNN_arg.add_argument('--data_dir', type=str, default='../../data/kdd17/HIV_fMRI_3_fold',
                      help='Directory in which data is stored')

IsoNN_arg.add_argument('--fold_count', type=str, default='1',
                      help='which fold is chosen')

IsoNN_arg.add_argument('--result_dir', type=str, default='../../result/expRecurrent_Graph_Model/',
                      help='Directory in which result will be stored')



def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed
