from ssp_navigation.utils.training import PolicyEvaluation
import torch
import numpy as np
import argparse
import os
from ssp_navigation.utils.models import MLP, LearnedEncoding, LearnedSSPEncoding
# from utils.encodings import get_ssp_encode_func, encode_trig, encode_hex_trig, encode_random_trig, \
#     encode_projection, get_one_hot_encode_func, get_pc_gauss_encoding_func, get_tile_encoding_func
# from spatial_semantic_pointers.utils import encode_random
# from functools import partial
from ssp_navigation.utils.encodings import get_encoding_function
from sklearn.metrics import r2_score
import nengo.spa as spa
import pandas as pd
import sys

parser = argparse.ArgumentParser(
    'Compute the RMSE of a policy on a dataset'
)


parser.add_argument('--spatial-encoding', type=str, default='ssp',
                    choices=[
                        'ssp', 'hex-ssp', 'periodic-hex-ssp', 'grid-ssp', 'ind-ssp', 'orth-proj-ssp',
                        'learned-ssp',
                        'rec-ssp', 'rec-hex-ssp', 'rec-ind-ssp', 'sub-toroid-ssp', 'proj-ssp', 'var-sub-toroid-ssp',
                        'random', '2d', '2d-normalized', 'one-hot',
                        'hex-trig', 'trig', 'random-trig', 'random-rotated-trig', 'random-proj',
                        'learned', 'learned-normalized', 'frozen-learned', 'frozen-learned-normalized', 'legendre',
                        'pc-gauss', 'pc-dog', 'tile-coding'
                    ],
                    help='coordinate encoding for agent location and goal')
parser.add_argument('--freq-limit', type=float, default=10,
                    help='highest frequency of sine wave for random-trig encodings')
parser.add_argument('--hex-freq-coef', type=float, default=2.5, help='constant to scale frequencies by for hex-trig')
parser.add_argument('--pc-gauss-sigma', type=float, default=0.75, help='sigma for the gaussians')
parser.add_argument('--pc-diff-sigma', type=float, default=1.5, help='sigma for subtracted gaussian in DoG')
parser.add_argument('--hilbert-points', type=int, default=1, choices=[0, 1, 2, 3],
                    help='pc centers. 0: random uniform. 1: hilbert curve. 2: evenly spaced grid. 3: hex grid')
parser.add_argument('--n-tiles', type=int, default=8, help='number of layers for tile coding')
parser.add_argument('--n-bins', type=int, default=0, help='number of bins for tile coding')
parser.add_argument('--ssp-scaling', type=float, default=1.0)
parser.add_argument('--grid-ssp-min', type=float, default=0.25, help='minimum plane wave scale')
parser.add_argument('--grid-ssp-max', type=float, default=2.0, help='maximum plane wave scale')
parser.add_argument('--phi', type=float, default=0.5, help='phi as a fraction of pi for orth-proj-ssp')
parser.add_argument('--n-proj', type=int, default=3, help='projection dimension for sub toroids')
parser.add_argument('--scale-ratio', type=float, default=(1 + 5 ** 0.5) / 2, help='ratio between sub toroid scales')
parser.add_argument('--subsample', type=int, default=1, help='amount to subsample for the visualization validation')
parser.add_argument('--encoding-limit', type=float, default=0.0,
                    help='if set, use this upper limit to define the space that the encoding is optimized over')
parser.add_argument('--maze-id-type', type=str, choices=['ssp', 'one-hot', 'random-sp'], default='random-sp',
                    help='ssp: region corresponding to maze layout.'
                         'one-hot: each maze given a one-hot vector.'
                         'random-sp: each maze given a unique random SP as an ID')
parser.add_argument('--maze-id-dim', type=int, default=256, help='Dimensionality for the Maze ID')
parser.add_argument('--seed', type=int, default=13, help='Seed for training and generating axis SSPs')
parser.add_argument('--dim', type=int, default=512, help='Dimensionality of the SSPs')
parser.add_argument('--hidden-size', type=int, default=512, help='Size of the hidden layer in the model')
parser.add_argument('--n-hidden-layers', type=int, default=1, help='Number of hidden layers in the model')
parser.add_argument('--dropout-fraction', type=float, default=0.0, help='Amount of dropout to apply during training')
parser.add_argument('--dataset-dir', type=str, default='datasets/mixed_style_20mazes_50goals_64res_13size_13seed')
parser.add_argument('--model', type=str, default='', help='Saved model to use')

parser.add_argument('--gpu', type=int, default=-1,
                    help="set to an integer corresponding to the gpu to use. Set to -1 to use the CPU")

parser.add_argument('--out-file', type=str, default="", help='Output file name')

parser.add_argument('--n-train-samples', type=int, default=100000, help='Number of training samples to evaluate')
parser.add_argument('--n-test-samples', type=int, default=100000, help='Number of test set samples to evaluate')

parser.add_argument('--tile-mazes', action='store_true', help='put all mazes into the same space')
parser.add_argument('--connected-tiles', action='store_true', help='all mazes in the same space and connected')

# Tags for the pandas dataframe
parser.add_argument('--dataset', type=str, default='', choices=['', 'blocks', 'maze', 'mixed'])
parser.add_argument('--trained-on', type=str, default='', choices=['', 'blocks', 'maze', 'mixed'])
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--n-mazes', type=int, default=10, help='Number of mazes from the dataset that were trained on')
parser.add_argument('--loss-function', type=str, default='mse', choices=['mse', 'cosine'])
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight-decay', type=float, default=0, help='strength of L2 weight normalization')
parser.add_argument('--optimizer', type=str, default='rmsprop', choices=['rmsprop', 'adam'])

parser.add_argument('--use-cache', type=int, default=1, help='if 1, use cache, if 0, do not')
parser.add_argument('--overwrite-output', type=int, default=1, help='If 0, do not run if output file exists')



args = parser.parse_args()

# an output file must be specified
assert args.out_file != ""
# only support npz or pandas csv
assert '.csv' in args.out_file

if args.overwrite_output == 0:
    if os.path.exists(args.out_file):
        print("Output file already exists, skipping")
        print(args.out_file)
        sys.exit(0)
    else:
        print("Generating data for:")
        print(args.out_file)
else:
    print("Generating data for:")
    print(args.out_file)

if args.connected_tiles:
    dataset_file = os.path.join(args.dataset_dir, 'connected_{}tile_dataset_gen.npz'.format(args.n_mazes))
else:
    dataset_file = os.path.join(args.dataset_dir, 'maze_dataset.npz')

data = np.load(dataset_file)

rng = np.random.RandomState(seed=args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# n_mazes by res by res
fine_mazes = data['fine_mazes']

# n_mazes by res by res by 2
solved_mazes = data['solved_mazes']

# # n_mazes by dim
# maze_sps = data['maze_sps']

# n_mazes by n_goals by 2
goals = data['goals']

n_goals = goals.shape[1]
n_mazes = args.n_mazes  # fine_mazes.shape[0]

if args.gpu == -1:
    device = torch.device('cpu:0')
    pin_memory = False
else:
    device = torch.device('cuda:{}'.format(int(args.gpu)))
    pin_memory = True

if args.maze_id_type == 'ssp':
    id_size = args.maze_id_dim
    maze_sps = data['maze_sps']
elif args.maze_id_type == 'one-hot':
    id_size = n_mazes
    # overwrite data
    maze_sps = np.eye(n_mazes)
elif args.maze_id_type == 'random-sp':
    id_size = args.maze_id_dim
    if id_size > 0:
        maze_sps = np.zeros((n_mazes, args.maze_id_dim))
        # overwrite data
        for mi in range(n_mazes):
            maze_sps[mi, :] = spa.SemanticPointer(args.maze_id_dim).v
    else:
        maze_sps = None
else:
    raise NotImplementedError

limit_low = 0
if args.encoding_limit != 0.0:
    limit_high = args.encoding_limit
else:
    limit_high = data['coarse_mazes'].shape[2]

    # modify the encoding limit to account for all of the environments
    if args.tile_mazes:
        limit_high *= int(np.ceil(np.sqrt(args.n_mazes)))


encoding_func, repr_dim = get_encoding_function(args, limit_low=limit_low, limit_high=limit_high)


# TODO: create a way to cache the validation set, so it doesn't need to be remade every time for multiple runs.
#       have an init flag to use the cache, rather than pickle the entire object
#       check whether the cache exists before generating the object

# if args.use_cache == 1:
#     dataset_name = args.dataset_dir.split('/')[-1]
#     cache_fname = 'validation_set_cache/{}_{}_dim{}_{}mazes_{}goals_id_{}_seed{}.npz'.format(
#         dataset_name,
#         args.spatial_encoding,
#         args.dim,
#         args.n_mazes_tested,
#         args.n_goals_tested,
#         args.maze_id_type,
#         args.seed,
#     )
# else:
#     cache_fname = ''


# input is maze, loc, goal ssps, output is 2D direction to move
if args.spatial_encoding == 'learned-ssp':
    model = LearnedSSPEncoding(
        input_size=repr_dim,
        encoding_size=args.dim,
        maze_id_size=id_size,
        hidden_size=args.hidden_size,
        output_size=2,
        n_layers=args.n_hidden_layers,
        dropout_fraction=args.dropout_fraction,
    )
elif 'learned' in args.spatial_encoding:
    model = LearnedEncoding(
        input_size=repr_dim,
        encoding_size=args.dim,
        maze_id_size=id_size,
        hidden_size=args.hidden_size,
        output_size=2,
        n_layers=args.n_hidden_layers,
        dropout_fraction=args.dropout_fraction,
    )
else:
    model = MLP(
        input_size=id_size + repr_dim * 2,
        hidden_size=args.hidden_size,
        output_size=2,
        n_layers=args.n_hidden_layers,
        dropout_fraction=args.dropout_fraction,
    )

if args.gpu == -1:
    model.load_state_dict(torch.load(args.model, map_location=lambda storage, loc: storage), strict=True)
else:
    model.load_state_dict(torch.load(args.model), strict=True)

model.to(device)

validation_set = PolicyEvaluation(
    data=data, dim=repr_dim, maze_sps=maze_sps,
    encoding_func=encoding_func, device=device,
    n_train_samples=args.n_train_samples,
    n_test_samples=args.n_test_samples,
    spatial_encoding=args.spatial_encoding,
    n_mazes=n_mazes,
    tile_mazes=args.tile_mazes,
    connected_tiles=args.connected_tiles,
)

# Reset seeds here after generating data
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# avg_rmse_train, avg_angle_rmse_train, avg_rmse_test, avg_angle_rmse_test = validation_set.get_rmse(model)
# train_r2, test_r2 = validation_set.get_r2_score(model)

avg_rmse_train, avg_angle_rmse_train, avg_rmse_test, avg_angle_rmse_test, train_r2, test_r2, avg_mse_loss_train, avg_mse_loss_test = validation_set.get_r2_and_rmse(model)

# Keep track of global and local RMSE separately for connected tiles
if args.connected_tiles:
    global_avg_rmse_train, global_avg_angle_rmse_train, global_avg_rmse_test, global_avg_angle_rmse_test = validation_set.get_global_rmse(model)
    df = pd.DataFrame(
        data=[[(avg_rmse_train + global_avg_rmse_train)/2.,
               (avg_angle_rmse_train + global_avg_angle_rmse_train)/2.,
               (avg_rmse_test + global_avg_rmse_test)/2.,
               (avg_angle_rmse_test + global_avg_angle_rmse_test)/2.,
               avg_rmse_train, avg_angle_rmse_train, avg_rmse_test, avg_angle_rmse_test,
               global_avg_rmse_train, global_avg_angle_rmse_train, global_avg_rmse_test, global_avg_angle_rmse_test,
               train_r2, test_r2,
               ]],
        columns=['Train RMSE', 'Train Angular RMSE', 'RMSE', 'Angular RMSE',
                 'Local Train RMSE', 'Local Train Angular RMSE', 'Local RMSE', 'Local Angular RMSE',
                 'Global Train RMSE', 'Global Train Angular RMSE', 'Global RMSE', 'Global Angular RMSE',
                 'Train R2', 'R2'
                 ],
    )
else:
    df = pd.DataFrame(
        data=[[avg_rmse_train, avg_angle_rmse_train, avg_rmse_test, avg_angle_rmse_test, train_r2, test_r2, avg_mse_loss_train, avg_mse_loss_test]],
        columns=['Train RMSE', 'Train Angular RMSE', 'RMSE', 'Angular RMSE', 'Train R2', 'R2', 'Train Loss', 'Test Loss'],
    )

df['Dimensionality'] = args.dim
df['Hidden Layer Size'] = args.hidden_size
df['Hidden Layers'] = args.n_hidden_layers
df['Encoding'] = args.spatial_encoding
df['Seed'] = args.seed
df['Maze ID Type'] = args.maze_id_type
df['Maze ID Dim'] = args.maze_id_dim
df['Tile Mazes'] = args.tile_mazes
df['Dropout Fraction'] = args.dropout_fraction
df['SSP Scaling'] = args.ssp_scaling

if args.spatial_encoding == 'sub-toroid-ssp':
    df['Proj Dim'] = args.n_proj
    df['Scale Ratio'] = args.scale_ratio

if args.spatial_encoding == 'proj-ssp':
    df['Proj Dim'] = args.n_proj

if args.spatial_encoding == 'grid-ssp':
    df['Grid SSP Min'] = args.grid_ssp_min
    df['Grid SSP Max'] = args.grid_ssp_max

if (args.spatial_encoding == 'pc-gauss') or (args.spatial_encoding == 'pc-dog'):
    df['Hilbert Curve'] = args.hilbert_points
    df['Sigma'] = args.pc_gauss_sigma
    if args.spatial_encoding == 'pc-dog':
        df['Diff Sigma'] = args.pc_diff_sigma

if (args.spatial_encoding == 'random-trig') or (args.spatial_encoding == 'random-rotated-trig'):
    df['Freq Limit'] = args.freq_limit

# # Other differentiators
# df['Number of Mazes Tested'] = args.n_mazes_tested
# df['Number of Goals Tested'] = args.n_goals_tested

# Command line supplied tags
df['Dataset'] = args.dataset
df['Trained On'] = args.trained_on
df['Epochs'] = args.epochs
df['Batch Size'] = args.batch_size
df['Number of Mazes'] = args.n_mazes
df['Loss Function'] = args.loss_function
df['Learning Rate'] = args.lr
df['Momentum'] = args.momentum
df['Weight Decay'] = args.weight_decay
df['Optimizer'] = args.optimizer

df.to_csv(args.out_file)
