from ssp_navigation.utils.training import PolicyEvaluation, PolicyValidationSet
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
    'View spatial activations of a policy'
)


parser.add_argument('--spatial-encoding', type=str, default='sub-toroid-ssp',
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
parser.add_argument('--ssp-scaling', type=float, default=0.5)
parser.add_argument('--grid-ssp-min', type=float, default=0.25, help='minimum plane wave scale')
parser.add_argument('--grid-ssp-max', type=float, default=2.0, help='maximum plane wave scale')
parser.add_argument('--phi', type=float, default=0.5, help='phi as a fraction of pi for orth-proj-ssp')
parser.add_argument('--n-proj', type=int, default=3, help='projection dimension for sub toroids')
parser.add_argument('--scale-ratio', type=float, default=0, help='ratio between sub toroid scales')
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
parser.add_argument('--dataset-dir', type=str, default='datasets/mixed_style_100mazes_100goals_64res_13size_13seed')
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
parser.add_argument('--optimizer', type=str, default='rmsprop', choices=['rmsprop', 'adam'])

parser.add_argument('--use-cache', type=int, default=1, help='if 1, use cache, if 0, do not')
parser.add_argument('--overwrite-output', type=int, default=0, help='If 0, do not run if output file exists')



args = parser.parse_args()

# an output file must be specified
assert args.out_file != ""
# only support npz or pandas csv
assert '.npz' in args.out_file

if args.overwrite_output == 0 and os.path.exists(args.out_file):
    print("Output file already exists, loading and viewing")
    print(args.out_file)
    data = np.load(args.out_file)
    feature_maps_null_goal = data['feature_maps_null_goal']
    feature_maps_null_start = data['feature_maps_null_start']
    # sys.exit(0)
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

    res = 32

    xs = np.linspace(limit_low, limit_high, res)
    ys = np.linspace(limit_low, limit_high, res)

    maze_indices = [0, 1, 2, 3, 5]
    goal_indices = [0]

    validation_set = PolicyValidationSet(
        data=data, dim=repr_dim, maze_sps=maze_sps, maze_indices=maze_indices, goal_indices=goal_indices, subsample=args.subsample,
        # spatial_encoding=args.spatial_encoding,
        tile_mazes=args.tile_mazes,
        connected_tiles=args.connected_tiles,  # NOTE: viz set currently does not use this
        encoding_func=encoding_func, device=device
    )

    feature_maps_null_goal, feature_maps_null_start = validation_set.get_spatial_activations(
        model, dim=args.dim, iddim=args.maze_id_dim, hidden_size=args.hidden_size, n_mazes=len(maze_indices), res=64
    )

    np.savez(
        args.out_file,
        feature_maps_null_goal=feature_maps_null_goal,
        feature_maps_null_start=feature_maps_null_start,
    )

# view results

n_mazes = feature_maps_null_goal.shape[0]
res = int(np.sqrt(feature_maps_null_goal.shape[1]))
n_features = feature_maps_null_goal.shape[2]

import matplotlib.pyplot as plt

# create image of the mazes for reference
fig, ax = plt.subplots(1, n_mazes)
if args.connected_tiles:
    dataset_file = os.path.join(args.dataset_dir, 'connected_{}tile_dataset_gen.npz'.format(args.n_mazes))
else:
    dataset_file = os.path.join(args.dataset_dir, 'maze_dataset.npz')
data = np.load(dataset_file)
# n_mazes by res by res
fine_mazes = data['fine_mazes']
for mi in range(n_mazes):
    ax[mi].imshow(fine_mazes[mi, :, :])


# folder_name = 'output/feature_images'
folder_name = 'output/feature_images_debug'
if args.tile_mazes:
    folder_name = 'output/feature_images_tiled'

if not os.path.exists(folder_name):
    os.makedirs(folder_name)

fig.savefig("{}/environments.png".format(folder_name))

fig, ax = plt.subplots(2, n_mazes)


for fi in range(n_features):
    for mi in range(n_mazes):
        # plt.imshow(feature_maps_null_goal[mi, :, fi].reshape(res, res))
        ax[0, mi].imshow(feature_maps_null_goal[mi, :, fi].reshape(res, res))
        ax[1, mi].imshow(feature_maps_null_start[mi, :, fi].reshape(res, res))
    fig.savefig("{}/feature_map_{}.png".format(folder_name, fi))
