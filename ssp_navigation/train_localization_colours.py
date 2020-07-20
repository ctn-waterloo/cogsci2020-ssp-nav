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
# from spatial_semantic_pointers.utils import get_heatmap_vectors, ssp_to_loc, ssp_to_loc_v
from ssp_navigation.utils.training import SnapshotValidationSet, coloured_localization_encoding_train_test_loaders
from ssp_navigation.utils.models import FeedForward, MLP
from ssp_navigation.utils.encodings import get_encoding_function, get_encoding_heatmap_vectors
import nengo.spa as spa
import pandas as pd
import sys


parser = argparse.ArgumentParser('Run 2D supervised localization experiment with coloured snapshots using pytorch')

parser.add_argument('--seed', type=int, default=13)
parser.add_argument('--n-epochs', type=int, default=100)
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
parser.add_argument('--weight-decay', type=float, default=0, help='strength of L2 weight normalization')


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
parser.add_argument('--scale-ratio', type=float, default=0, help='ratio between sub toroid scales')
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
                    default='datasets/mixed_style_100mazes_100goals_64res_13size_13seed/coloured_10mazes_36sensors_360fov_100000samples')
parser.add_argument('--variant-subfolder', type=str, default='',
                    help='Optional custom subfolder')
parser.add_argument('--gpu', type=int, default=-1,
                    help="set to an integer corresponding to the gpu to use. Set to -1 to use the CPU")
parser.add_argument('--dataset-seed', type=int, default=13)

parser.add_argument('--out-file', type=str, default="", help='Output file name')

args = parser.parse_args()

dataset_file = os.path.join(args.dataset_dir, 'coloured_sensor_dataset.npz')

variant_folder = '{}_{}dim_{}layer_{}units'.format(
    args.spatial_encoding, args.dim, args.n_hidden_layers, args.hidden_size,
)

if args.variant_subfolder != '':
    variant_folder = os.path.join(args.variant_subfolder, variant_folder)

logdir = os.path.join(args.dataset_dir, 'coloured_snapshot_network', variant_folder)

if os.path.exists(logdir):
    print("Skipping, output already exists for:")
    print(logdir)
    sys.exit(0)

torch.manual_seed(args.seed)
np.random.seed(args.seed)


if args.gpu == -1:
    device = torch.device('cpu:0')
    pin_memory = False
else:
    device = torch.device('cuda:{}'.format(int(args.gpu)))
    pin_memory = True


current_time = datetime.now().strftime('%b%d_%H-%M-%S')
save_dir = os.path.join(logdir, current_time)
writer = SummaryWriter(log_dir=save_dir)

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
n_sensors = data['dist_sensors'].shape[2]
n_mazes = data['coarse_mazes'].shape[0]


# Used for visualization of test set performance using pos = ssp_to_loc(sp, heatmap_vectors, xs, ys)
heatmap_vectors = get_encoding_heatmap_vectors(xs, ys, repr_dim, encoding_func)

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
    model.load_state_dict(torch.load(args.load_saved_model), strict=False)

cosine_criterion = nn.CosineEmbeddingLoss()
mse_criterion = nn.MSELoss()

if args.optimizer == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
elif args.optimizer == 'rmsprop':
    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
elif args.optimizer == 'adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

dataset_rng = np.random.RandomState(seed=args.dataset_seed)
# TODO: update these to be more general
trainloader, testloader = coloured_localization_encoding_train_test_loaders(
    data,
    encoding_func=encoding_func,
    encoding_dim=repr_dim,
    maze_sps=maze_sps,
    n_train_samples=args.n_train_samples,
    n_test_samples=args.n_test_samples,
    batch_size=batch_size,
    n_mazes_to_use=args.n_mazes,
    rng=dataset_rng
)

if args.spatial_encoding == '2d':
    spatial_decoding_type = '2d'
elif args.spatial_encoding == '2d-normalized':
    spatial_decoding_type = '2d-normalized'
else:
    spatial_decoding_type = 'ssp'  # Note: this is actually generic and represents any non-2d encoding

validation_set = SnapshotValidationSet(
    dataloader=testloader,
    heatmap_vectors=heatmap_vectors,
    xs=xs,
    ys=ys,
    device=device,
    spatial_encoding=spatial_decoding_type,
)

params = vars(args)
with open(os.path.join(save_dir, "params.json"), "w") as f:
    json.dump(params, f)

# Keep track of running average losses, to adaptively scale the weight between them
running_avg_cosine_loss = 1.
running_avg_mse_loss = 1.

print("Training")
for epoch in range(n_epochs):
    print("Epoch {} of {}".format(epoch + 1, n_epochs))

    # Every 'eval_period' epochs, create a test loss and image
    if epoch % args.eval_period == 0:

        print("Evaluating")
        model.eval()
        validation_set.run_eval(
            model=model,
            writer=writer,
            epoch=epoch,
        )
        model.train()

        torch.save(
            model.state_dict(),
            os.path.join(save_dir, '{}_snapshot_localization_model_epoch{}.pt'.format(args.spatial_encoding, epoch))
        )

    avg_cosine_loss = 0
    avg_mse_loss = 0
    avg_combined_loss = 0
    avg_scaled_loss = 0
    n_batches = 0
    for i, data in enumerate(trainloader):
        # sensor_inputs, maze_ids, ssp_outputs = data
        # sensor and map IDs combined
        combined_inputs, ssp_outputs = data

        if combined_inputs.size()[0] != batch_size:
            continue  # Drop data, not enough for a batch
        optimizer.zero_grad()
        # model.zero_grad()

        # ssp_pred = model(sensor_inputs, maze_ids)
        ssp_pred = model(combined_inputs.to(device))

        cosine_loss = cosine_criterion(
            ssp_pred,
            ssp_outputs.to(device),
            torch.ones(ssp_pred.shape[0]).to(device)
        )
        mse_loss = mse_criterion(ssp_pred, ssp_outputs.to(device))

        loss = cosine_loss + mse_loss

        # adaptive weighted combination of the two loss functions
        c_f = (running_avg_mse_loss / (running_avg_mse_loss + running_avg_cosine_loss))
        m_f = (running_avg_cosine_loss / (running_avg_mse_loss + running_avg_cosine_loss))
        scaled_loss = cosine_loss * c_f + mse_loss * m_f

        if args.loss_function == 'cosine':
            cosine_loss.backward()
        elif args.loss_function == 'mse':
            mse_loss.backward()
        elif args.loss_function == 'combined':
            loss.backward()
        elif args.loss_function == 'alternating':
            if epoch % 2 == 0:
                cosine_loss.backward()
            else:
                mse_loss.backward()
        elif args.loss_function == 'scaled':
            scaled_loss.backward()

        # Gradient Clipping
        # torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_thresh)

        optimizer.step()

        avg_mse_loss += mse_loss.data.item()
        avg_cosine_loss += cosine_loss.data.item()
        avg_combined_loss += (cosine_loss.data.item() + mse_loss.data.item())
        avg_scaled_loss += scaled_loss.data.item()
        n_batches += 1

    avg_mse_loss /= n_batches
    avg_cosine_loss /= n_batches
    avg_combined_loss /= n_batches
    avg_scaled_loss /= n_batches
    print("mse loss:", avg_mse_loss)
    print("cosine loss:", avg_cosine_loss)
    print("combined loss:", avg_combined_loss)
    print("scaled loss:", avg_scaled_loss)
    writer.add_scalar('avg_cosine_loss', avg_cosine_loss, epoch + 1)
    writer.add_scalar('avg_mse_loss', avg_mse_loss, epoch + 1)
    writer.add_scalar('avg_combined_loss', avg_combined_loss, epoch + 1)
    writer.add_scalar('avg_scaled_loss', avg_scaled_loss, epoch + 1)

    running_avg_cosine_loss = 0.9 * running_avg_cosine_loss + 0.1 * avg_cosine_loss
    running_avg_mse_loss = 0.9 * running_avg_mse_loss + 0.1 * avg_mse_loss


print("Testing")
model.eval()
coord_rmse, avg_mse_loss, avg_cosine_loss = validation_set.run_eval(
    model=model,
    writer=writer,
    epoch=n_epochs,
    ret_loss=True
)


torch.save(model.state_dict(), os.path.join(save_dir, '{}_snapshot_localization_model.pt'.format(args.spatial_encoding)))

# save the RMSE from the test set in an easy to access fashion
np.savez(
    os.path.join(save_dir, 'rmse.npz'),
    coord_rmse=coord_rmse,
)

if args.out_file == '':
    print("No output file given for dataframe, skipping")
else:
    print("Generating data for:")
    print(args.out_file)

    df = pd.DataFrame(
        data=[[coord_rmse, avg_mse_loss, avg_cosine_loss]],
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
    df['Weight Decay'] = args.weight_decay
    df['Momentum'] = args.momentum
    df['Optimizer'] = args.optimizer

    df.to_csv(args.out_file)
