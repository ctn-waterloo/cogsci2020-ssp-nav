import matplotlib
import os
# allow code to work on machines without a display or in a screen session
display = os.environ.get('DISPLAY')
if display is None or 'localhost' in display:
    matplotlib.use('agg')

import argparse
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
from tensorboardX import SummaryWriter
import json
from spatial_semantic_pointers.utils import ssp_to_loc_v
from ssp_navigation.utils.training import LocalizationSnapshotDataset, coloured_localization_encoding_train_test_loaders
from ssp_navigation.utils.models import FeedForward, MLP
from ssp_navigation.utils.encodings import get_encoding_function, get_encoding_heatmap_vectors
import nengo.spa as spa
import pandas as pd


parser = argparse.ArgumentParser('Run 2D supervised localization experiment with coloured snapshots using pytorch')

parser.add_argument('--seed', type=int, default=13)
parser.add_argument('--n-epochs', type=int, default=100)
parser.add_argument('--n-samples', type=int, default=10000)
parser.add_argument('--n-train-samples', type=int, default=50000)
parser.add_argument('--n-test-samples', type=int, default=10000)
parser.add_argument('--eval-period', type=int, default=25)
parser.add_argument('--load-saved-model', type=str, default='', help='Saved model to load from')
parser.add_argument('--loss-function', type=str, default='combined',
                    choices=['cosine', 'mse', 'combined', 'alternating', 'scaled'])
parser.add_argument('--n-mazes', type=int, default=10, help='number of mazes to use. Set to 0 to use all in the dataset')
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--lr', type=float, default=1e-3, help='Step size multiplier in the RMSProp algorithm')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum parameter of the RMSProp algorithm')


parser.add_argument('--spatial-encoding', type=str, default='ssp',
                    choices=[
                        'ssp', 'hex-ssp', 'periodic-hex-ssp', 'grid-ssp', 'ind-ssp', 'orth-proj-ssp',
                        'random', '2d', '2d-normalized', 'one-hot', 'sub-toroid-ssp', 'proj-ssp', 'var-sub-toroid-ssp',
                        'hex-trig', 'trig', 'random-trig', 'random-rotated-trig', 'random-proj', 'legendre',
                        'learned', 'frozen-learned',
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
parser.add_argument('--encoding-limit', type=float, default=0.0,
                    help='if set, use this upper limit to define the space that the encoding is optimized over')
parser.add_argument('--dim', type=int, default=512, help='Dimensionality of encoding')

parser.add_argument('--maze-id-type', type=str, choices=['ssp', 'one-hot', 'random-sp'], default='random-sp',
                    help='ssp: region corresponding to maze layout.'
                         'one-hot: each maze given a one-hot vector.'
                         'random-sp: each maze given a unique random SP as an ID')
parser.add_argument('--maze-id-dim', type=int, default=256, help='Dimensionality for the Maze ID')

parser.add_argument('--tile-mazes', action='store_true', help='put all mazes into the same space')
parser.add_argument('--connected-tiles', action='store_true', help='all mazes in the same space and connected')

parser.add_argument('--optimizer', type=str, default='adam', choices=['rmsprop', 'adam', 'sgd'])
parser.add_argument('--dropout-fraction', type=float, default=0.1)

parser.add_argument('--hidden-size', type=int, default=2048)
parser.add_argument('--n-hidden-layers', type=int, default=1, help='Number of hidden layers in the model')
parser.add_argument('--grad-clip-thresh', type=float, default=1e-5, help='Gradient clipping threshold')

parser.add_argument('--dataset-dir', type=str,
                    default='datasets/mixed_style_100mazes_100goals_64res_13size_13seed/coloured_10mazes_36sensors_360fov_0samples')
parser.add_argument('--variant-subfolder', type=str, default='',
                    help='Optional custom subfolder')
parser.add_argument('--gpu', type=int, default=-1,
                    help="set to an integer corresponding to the gpu to use. Set to -1 to use the CPU")
parser.add_argument('--dataset-seed', type=int, default=13)

parser.add_argument('--out-file', type=str, default="", help='Output file name')

parser.add_argument('--heatmap-res', type=int, default=128)

args = parser.parse_args()

dataset_file = os.path.join(args.dataset_dir, 'coloured_sensor_dataset_test.npz')

# variant_folder = '{}_{}dim_{}layer_{}units'.format(
#     args.spatial_encoding, args.dim, args.n_hidden_layers, args.hidden_size,
# )
#
# if args.variant_subfolder != '':
#     variant_folder = os.path.join(args.variant_subfolder, variant_folder)


torch.manual_seed(args.seed)
np.random.seed(args.seed)


if args.gpu == -1:
    device = torch.device('cpu:0')
    pin_memory = False
else:
    device = torch.device('cuda:{}'.format(int(args.gpu)))
    pin_memory = True


data = np.load(dataset_file)


n_mazes = data['fine_mazes'].shape[0]

if args.maze_id_type == 'ssp':
    id_size = args.maze_id_dim
    raise NotImplementedError  # this method is currently not supported by this script
elif args.maze_id_type == 'one-hot':
    id_size = n_mazes
    # overwrite data
    maze_sps = np.eye(n_mazes)
elif args.maze_id_type == 'random-sp':
    id_size = args.maze_id_dim
    maze_sps = np.zeros((n_mazes, args.maze_id_dim))
    # overwrite data
    if id_size > 0:
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

xs = data['xs']
ys = data['ys']

# dist sensors is (n_mazes, res, res, n_sensors)
n_sensors = data['dist_sensors'].shape[3]
n_mazes = data['coarse_mazes'].shape[0]


# Used for visualization of test set performance using pos = ssp_to_loc(sp, heatmap_vectors, xs, ys)
heatmap_vectors = get_encoding_heatmap_vectors(xs, ys, repr_dim, encoding_func, normalize=False)

# n_samples = args.n_samples
batch_size = args.batch_size
n_epochs = args.n_epochs

# TODO: make this a parameter, whether it is specific to a single maze, or handles multiple
# alternatively, should it learn the maze ID from what it sees? that could be an additional output rather than input
# may need more rich sensory environment, such as colour, or information over time.
# Could produce interesting backtracking behaviour if it learns the environment context as it moves
n_maze_dim = id_size  # args.maze_id_dim #0

# Input is the distance sensor measurements
model = MLP(
    input_size=n_sensors*4 + n_maze_dim,
    hidden_size=args.hidden_size,
    output_size=repr_dim,
    n_layers=args.n_hidden_layers,
    dropout_fraction=args.dropout_fraction,
)

model.to(device)

if args.load_saved_model:
    model.load_state_dict(torch.load(args.load_saved_model, map_location=lambda storage, loc: storage), strict=False)
else:
    raise NotImplementedError('Must provide a model to evaluate')

cosine_criterion = nn.CosineEmbeddingLoss()
mse_criterion = nn.MSELoss()


# generate the dataset

# shape is (n_mazes, res, res, n_sensors, 4)
dist_sensors = data['dist_sensors']

fine_mazes = data['fine_mazes']

n_sensors = dist_sensors.shape[3]

# ssps = data['ssps']

n_mazes = data['coarse_mazes'].shape[0]
# dim = x_axis_vec.shape[0]


sensor_inputs = np.zeros((args.n_samples, n_sensors*4))

# these include outputs for every time-step
encoding_outputs = np.zeros((args.n_samples, repr_dim))

# for the 2D encoding method
pos_outputs = np.zeros((args.n_samples, 2))

if maze_sps is not None:
    maze_ids = np.zeros((args.n_samples, maze_sps.shape[1]))
else:
    maze_ids = None

eval_rng = np.random.RandomState(seed=13)

for i in range(args.n_samples):
    # choose random maze and position in maze
    maze_ind = eval_rng.randint(low=0, high=args.n_mazes)
    xi = eval_rng.randint(low=0, high=len(xs))
    yi = eval_rng.randint(low=0, high=len(ys))
    # Keep choosing position until it is not inside a wall
    while fine_mazes[maze_ind, xi, yi] == 1:
        xi = eval_rng.randint(low=0, high=len(xs))
        yi = eval_rng.randint(low=0, high=len(ys))

    sensor_inputs[i, :] = dist_sensors[maze_ind, xi, yi, :].flatten()

    encoding_outputs[i, :] = heatmap_vectors[xi, yi, :]

    if maze_sps is not None:
        # supports both one-hot and random-sp
        maze_ids[i, :] = maze_sps[maze_ind, :]

    # for the 2D encoding method
    pos_outputs[i, :] = np.array([xs[xi], ys[yi]])


dataset = LocalizationSnapshotDataset(
    sensor_inputs=sensor_inputs,
    maze_ids=maze_ids,
    ssp_outputs=encoding_outputs,
)

dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=args.n_samples, shuffle=True, num_workers=0,
)

if args.spatial_encoding == '2d':
    spatial_decoding_type = '2d'
elif args.spatial_encoding == '2d-normalized':
    spatial_decoding_type = '2d-normalized'
else:
    spatial_decoding_type = 'ssp'  # Note: this is actually generic and represents any non-2d encoding

eval_xs = np.linspace(limit_low, limit_high, args.heatmap_res)
eval_ys = np.linspace(limit_low, limit_high, args.heatmap_res)
eval_heatmap_vectors = get_encoding_heatmap_vectors(eval_xs, eval_ys, repr_dim, encoding_func, normalize=True)

with torch.no_grad():
    # Everything is in one batch, so this loop will only happen once
    for i, data in enumerate(dataloader):
        # sensor_inputs, map_ids, ssp_outputs = data
        # sensors and map ID combined
        combined_inputs, ssp_outputs = data

        # ssp_pred = model(sensor_inputs, map_ids)
        ssp_pred = model(combined_inputs.to(device))

        cosine_loss = cosine_criterion(
            ssp_pred,
            ssp_outputs.to(device),
            torch.ones(ssp_pred.shape[0]).to(device)
        )
        mse_loss = mse_criterion(
            ssp_pred,
            ssp_outputs.to(device)
        )

        print("test mse loss", mse_loss.data.item())
        print("test cosine loss", mse_loss.data.item())


    # One prediction and ground truth coord for every element in the batch
    # NOTE: this is assuming the eval set only has one giant batch
    predictions = np.zeros((ssp_pred.shape[0], 2))
    coords = np.zeros((ssp_pred.shape[0], 2))

    if spatial_decoding_type == 'ssp':
        print("computing prediction locations")
        predictions[:, :] = ssp_to_loc_v(
            ssp_pred.detach().cpu().numpy()[:, :],
            eval_heatmap_vectors, eval_xs, eval_ys
        )

        print("computing ground truth locations")
        coords[:, :] = ssp_to_loc_v(
            ssp_outputs.detach().cpu().numpy()[:, :],
            eval_heatmap_vectors, eval_xs, eval_ys
        )

    elif spatial_decoding_type == '2d':
        print("copying prediction locations")
        predictions[:, :] = ssp_pred.detach().cpu().numpy()[:, :]
        print("copying ground truth locations")
        coords[:, :] = ssp_outputs.detach().cpu().numpy()[:, :]

    elif spatial_decoding_type == '2d-normalized':
        print("copying prediction locations")
        predictions[:, :] = ssp_pred.detach().cpu().numpy()[:, :]
        print("copying ground truth locations")
        coords[:, :] = ssp_outputs.detach().cpu().numpy()[:, :]
        # un-normalizing
        coords = (coords + 1) / 2. * (limit_high - limit_low) - limit_low

    # Get a measure of error in the output coordinate space
    coord_rmse = np.sqrt((np.linalg.norm(predictions - coords, axis=1) ** 2).mean())

    print("coord rmse", coord_rmse)

df = pd.DataFrame(
    data=[[coord_rmse, mse_loss.data.item(), cosine_loss.data.item()]],
    columns=['RMSE', 'MSE Loss', 'Cosine Loss'],
)

df['Dimensionality'] = args.dim
df['Hidden Layer Size'] = args.hidden_size
df['Hidden Layers'] = args.n_hidden_layers
df['Encoding'] = args.spatial_encoding
df['Seed'] = args.seed
# df['Maze ID Type'] = args.maze_id_type
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
# df['Dataset'] = args.dataset
# df['Trained On'] = args.trained_on
df['Epochs'] = args.n_epochs
df['Batch Size'] = args.batch_size
df['Number of Mazes'] = args.n_mazes
df['Loss Function'] = args.loss_function
df['Learning Rate'] = args.lr
df['Momentum'] = args.momentum
df['Optimizer'] = args.optimizer

df.to_csv(args.out_file)
