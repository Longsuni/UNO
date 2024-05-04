import argparse
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=60,
                        help='number of epochs of training')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='training batch size')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='adam: learning rate')
    parser.add_argument('--b1', type=float, default=0.9,
                        help='adam: decay of first order momentum of gradient')
    parser.add_argument('--b2', type=float, default=0.999,
                        help='adam: decay of second order momentum of gradient')
    parser.add_argument('--n_channels', type=int, default=128,
                        help='number of channels')
    parser.add_argument('--channels', type=int, default=1,
                        help='number of flow image channels')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--use_exf', action='store_true', default=True,
                        help='External influence factors')
    parser.add_argument('--height', type=int, default=32,
                        help='height of the input map')
    parser.add_argument('--width', type=int, default=32,
                        help='weight of the input map')
    parser.add_argument('--scale_factor', type=int, default=4,
                        help='upscaling factor')
    parser.add_argument('--initial_train',type=bool, default=True)
    parser.add_argument('--edsr_channels', type=int, default=64,
                        help='number of edsr channels')
    parser.add_argument('--sample',type=int,default=4,
                        help='number of samples')
    parser.add_argument('--heads', type=int, default=16,
                        help='number of heads')
    parser.add_argument('--model', type=str, default='UNO',
                        help='chose model to use', )
    parser.add_argument('--scaler_X', type=int, default=1,
                        help='scaler of coarse-grained flows')
    parser.add_argument('--scaler_Y', type=int, default=1,
                        help='scaler of fine-grained flows')
    parser.add_argument('--c_map_shape', type=int, default=32)
    parser.add_argument('--f_map_shape', type=int, default=128)
    parser.add_argument('--ext_shape', type=int, default=7)
    parser.add_argument('--dataset', type=str, default='TaxiBJ',
                        help='dataset name')
    parser.add_argument('--sub_region', type=int, default=4,
                        help='sub regions number H\' and W\'')
    opt = parser.parse_args()

    return opt
print(get_args())