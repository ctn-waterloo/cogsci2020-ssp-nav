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
from ssp_navigation.utils.training import SnapshotValidationSet, snapshot_localization_encoding_train_test_loaders
from ssp_navigation.utils.models import FeedForward, MLP
from ssp_navigation.utils.encodings import get_encoding_function, get_encoding_heatmap_vectors
import nengo.spa as spa


parser = argparse.ArgumentParser('Run 2D supervised localization experiment with snapshots using pytorch')

parser.add_argument('--seed', type=int, default=13)
parser.add_argument('--n-epochs', type=int, default=100)
# parser.add_argument('--n-samples', type=int, default=10000)
parser.add_argument('--n-train-samples', type=int, default=50000)
parser.add_argument('--n-test-samples', type=int, default=10000)
# parser.add_argument('--encoding', type=str, default='ssp', choices=['ssp', '2d', 'pc'])
parser.add_argument('--eval-period', type=int, default=25)
parser.add_argument('--load-saved-model', type=str, default='', help='Saved model to load from')
parser.add_argument('--loss-function', type=str, default='combined',
                    choices=['cosine', 'mse', 'combined', 'alternating', 'scaled'])
parser.add_argument('--n-mazes', type=int, default=0, help='number of mazes to use. Set to 0 to use all in the dataset')
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

parser.add_argument('--hidden-size', type=int, default=512)
parser.add_argument('--n-hidden-layers', type=int, default=1, help='Number of hidden layers in the model')
parser.add_argument('--grad-clip-thresh', type=float, default=1e-5, help='Gradient clipping threshold')

parser.add_argument('--dataset-dir', type=str,
                    default='datasets/mixed_style_100mazes_100goals_64res_13size_13seed/36sensors_360fov')
parser.add_argument('--variant-subfolder', type=str, default='',
                    help='Optional custom subfolder')
parser.add_argument('--gpu', type=int, default=-1,
                    help="set to an integer corresponding to the gpu to use. Set to -1 to use the CPU")

args = parser.parse_args()

dataset_file = os.path.join(args.dataset_dir, 'maze_dataset.npz')

variant_folder = '{}_{}dim_{}layer_{}units'.format(
    args.spatial_encoding, args.dim, args.n_hidden_layers, args.hidden_size,
)

if args.variant_subfolder != '':
    variant_folder = os.path.join(args.variant_subfolder, variant_folder)

logdir = os.path.join(args.dataset_dir, 'snapshot_network', variant_folder)

torch.manual_seed(args.seed)
np.random.seed(args.seed)

# use_cosine_loss = False
# if args.loss_function == 'cosine':
#     use_cosine_loss = True
#
# if use_cosine_loss:
#     print("Using Cosine Loss")
# else:
#     print("Using MSE Loss")


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


# x_axis_vec = data['x_axis_sp']
# y_axis_vec = data['y_axis_sp']
#
xs = data['xs']
ys = data['ys']

# dist sensors is (n_mazes, res, res, n_sensors)
n_sensors = data['dist_sensors'].shape[3]
n_mazes = data['coarse_mazes'].shape[0]


# Used for visualization of test set performance using pos = ssp_to_loc(sp, heatmap_vectors, xs, ys)
# TODO: use the more general heatmap vector generation
# heatmap_vectors = get_heatmap_vectors(xs, ys, x_axis_vec, y_axis_vec)
heatmap_vectors = get_encoding_heatmap_vectors(xs, ys, args.dim, encoding_func)

# n_samples = args.n_samples
batch_size = args.batch_size
n_epochs = args.n_epochs

# # Input is the distance sensor measurements
# if args.n_hidden_layers > 1:
#     model = MLP(
#         input_size=n_sensors + n_mazes,
#         hidden_size=args.hidden_size,
#         output_size=args.dim,
#         n_layers=args.n_hidden_layers)
# else:
#     model = FeedForward(
#         input_size=n_sensors + n_mazes,
#         hidden_size=args.hidden_size,
#         output_size=args.dim,
#     )

# TODO: make this a parameter, whether it is specific to a single maze, or handles multiple
# alternatively, should it learn the maze ID from what it sees? that could be an additional output rather than input
# may need more rich sensory environment, such as colour, or information over time.
# Could produce interesting backtracking behaviour if it learns the environment context as it moves
n_maze_dim = id_size  # args.maze_id_dim #0

# Input is the distance sensor measurements
model = MLP(
    input_size=n_sensors + n_maze_dim,
    hidden_size=args.hidden_size,
    output_size=args.dim,
    n_layers=args.n_hidden_layers
)

model.to(device)

if args.load_saved_model:
    model.load_state_dict(torch.load(args.load_saved_model), strict=False)

cosine_criterion = nn.CosineEmbeddingLoss()
mse_criterion = nn.MSELoss()

if args.optimizer == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
elif args.optimizer == 'rmsprop':
    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, momentum=args.momentum)
elif args.optimizer == 'adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


# TODO: update these to be more general
trainloader, testloader = snapshot_localization_encoding_train_test_loaders(
    data,
    encoding_func=encoding_func,
    encoding_dim=args.dim,
    maze_sps=maze_sps,
    n_train_samples=args.n_train_samples,
    n_test_samples=args.n_test_samples,
    batch_size=batch_size,
    n_mazes_to_use=args.n_mazes,
)

validation_set = SnapshotValidationSet(
    dataloader=testloader,
    heatmap_vectors=heatmap_vectors,
    xs=xs,
    ys=ys,
    spatial_encoding='ssp',  # Note: this is actually generic and represents any non-2d encoding
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
        validation_set.run_eval(
            model=model,
            writer=writer,
            epoch=epoch,
        )

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
        ssp_pred = model(combined_inputs)

        cosine_loss = cosine_criterion(ssp_pred, ssp_outputs, torch.ones(ssp_pred.shape[0]))
        mse_loss = mse_criterion(ssp_pred, ssp_outputs)

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
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_thresh)

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
validation_set.run_eval(
    model=model,
    writer=writer,
    epoch=n_epochs,
)


torch.save(model.state_dict(), os.path.join(save_dir, '{}_snapshot_localization_model.pt'.format(args.spatial_encoding)))
