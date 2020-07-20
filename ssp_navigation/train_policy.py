from ssp_navigation.utils.training import PolicyValidationSet, create_policy_dataloader, create_train_test_dataloaders
import torch
import torch.nn as nn
import numpy as np
import argparse
import os
import json
import sys
from tensorboardX import SummaryWriter
from datetime import datetime
from ssp_navigation.utils.models import FeedForward, MLP, LearnedEncoding, LearnedSSPEncoding
# from utils.encodings import get_ssp_encode_func, encode_trig, encode_hex_trig, encode_random_trig, \
#     encode_projection, encode_one_hot, get_one_hot_encode_func, get_pc_gauss_encoding_func, get_tile_encoding_func
# from spatial_semantic_pointers.utils import encode_random
from ssp_navigation.utils.encodings import get_encoding_function
# from functools import partial
import nengo.spa as spa
from ssp_navigation.utils.training import PolicyEvaluation, add_weight_decay
import pandas as pd

parser = argparse.ArgumentParser(
    'Train a function that given a maze and a goal location, computes the direction to move to get to that goal'
)

parser.add_argument('--loss-function', type=str, default='mse', choices=['cosine', 'mse'])
parser.add_argument('--n-mazes', type=int, default=10, help='Number of mazes from the dataset to train with')
parser.add_argument('--epochs', type=int, default=250, help='Number of epochs to train for')
parser.add_argument('--epoch-offset', type=int, default=0,
                    help='Optional offset to start epochs counting from. To be used when continuing training')
parser.add_argument('--viz-period', type=int, default=50, help='number of epochs before a viz set run')
parser.add_argument('--val-period', type=int, default=25, help='number of epochs before a test/validation set run')
parser.add_argument('--spatial-encoding', type=str, default='ssp',
                    choices=[
                        'ssp', 'hex-ssp', 'periodic-hex-ssp', 'grid-ssp', 'ind-ssp', 'orth-proj-ssp',
                        'learned-ssp',
                        'rec-ssp', 'rec-hex-ssp', 'rec-ind-ssp', 'sub-toroid-ssp', 'proj-ssp', 'var-sub-toroid-ssp',
                        'random', '2d', '2d-normalized', 'one-hot', 'hex-trig',
                        'trig', 'random-trig', 'random-rotated-trig', 'random-proj', 'legendre',
                        'learned', 'learned-normalized', 'frozen-learned', 'frozen-learned-normalized',
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
parser.add_argument('--n-train-samples', type=int, default=50000, help='Number of training samples')
parser.add_argument('--n-test-samples', type=int, default=50000, help='Number of testing samples')

parser.add_argument('--tile-mazes', action='store_true', help='put all mazes into the same space')
parser.add_argument('--connected-tiles', action='store_true', help='all mazes in the same space and connected')

parser.add_argument('--hidden-size', type=int, default=512, help='Size of the hidden layer in the model')
parser.add_argument('--n-hidden-layers', type=int, default=1, help='Number of hidden layers in the model')
parser.add_argument('--dropout-fraction', type=float, default=0.0, help='Amount of dropout to apply during training')

parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight-decay', type=float, default=0, help='strength of L2 weight normalization')
parser.add_argument('--weight-histogram', action='store_true', help='Save histogram of the weights')
parser.add_argument('--res', type=int, default=64, help='resolution of the fine maze')
parser.add_argument('--dataset-dir', type=str, default='datasets/mixed_style_100mazes_100goals_64res_13size_13seed')
parser.add_argument('--no-wall-overlay', action='store_true',
                    help='Do not use rainbow colours and wall overlay in validation images')
parser.add_argument('--optimizer', type=str, default='rmsprop', choices=['rmsprop', 'adam', 'sgd'])
parser.add_argument('--variant-subfolder', type=str, default='',
                    help='Optional custom subfolder')
parser.add_argument('--logdir', type=str, default='policy',
                    help='Directory for saved model and tensorboard log, within dataset-dir')
parser.add_argument('--load-saved-model', type=str, default='', help='Saved model to load from')

parser.add_argument('--input-noise', type=float, default=0.0,
                    help='Gaussian noise to add to the inputs when training')  # 0.01
parser.add_argument('--shift-noise', type=float, default=0.0,
                    help='Shifting coordinates before encoding by up to this amount as a form of noise')  # 0.2

parser.add_argument('--decay-phis', action='store_true', help='apply weight decay to phis for learned-ssp')

parser.add_argument('--gpu', type=int, default=-1,
                    help="set to an integer corresponding to the gpu to use. Set to -1 to use the CPU")

parser.add_argument('--out-file', type=str, default="", help='Output file name')

args = parser.parse_args()

if args.connected_tiles:
    dataset_file = os.path.join(args.dataset_dir, 'connected_{}tile_dataset_gen.npz'.format(args.n_mazes))
else:
    dataset_file = os.path.join(args.dataset_dir, 'maze_dataset.npz')

variant_folder = '{}_{}train_{}_id_{}layer_{}units'.format(
    args.spatial_encoding, args.n_train_samples, args.maze_id_type, args.n_hidden_layers, args.hidden_size
)

if args.variant_subfolder != '':
    variant_folder = os.path.join(args.variant_subfolder, variant_folder)

logdir = os.path.join(args.dataset_dir, 'policy', variant_folder)

if os.path.exists(logdir):
    print("Skipping, output already exists for:")
    print(logdir)
    sys.exit(0)

dataset = np.load(dataset_file)

rng = np.random.RandomState(seed=args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# n_mazes by res by res
fine_mazes = dataset['fine_mazes']

# # n_mazes by res by res by 2
# solved_mazes = dataset['solved_mazes']

# # n_mazes by dim
# maze_sps = dataset['maze_sps']

# n_mazes by n_goals by 2
goals = dataset['goals']

n_goals = goals.shape[1]
n_mazes = fine_mazes.shape[0]

if args.gpu == -1:
    device = torch.device('cpu:0')
    pin_memory = False
else:
    device = torch.device('cuda:{}'.format(int(args.gpu)))
    pin_memory = True

if args.maze_id_type == 'ssp':
    id_size = args.maze_id_dim
    maze_sps = dataset['maze_sps']
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
    limit_high = dataset['coarse_mazes'].shape[2]

    # modify the encoding limit to account for all of the environments
    if args.tile_mazes:
        limit_high *= int(np.ceil(np.sqrt(args.n_mazes)))

encoding_func, repr_dim = get_encoding_function(args, limit_low=limit_low, limit_high=limit_high)

# Create a validation/visualization set to run periodically while training and at the end
# validation_set = ValidationSet(data=dataset, maze_indices=np.arange(n_mazes), goal_indices=[0])

# Set up number of mazes/goals to view in the viz set based on how many are available
if n_mazes < 4:
    maze_indices = list(np.arange(n_mazes))
    if n_goals < 4:
        goal_indices = list(np.arange(n_goals))
    else:
        goal_indices = [0, 1, 2, 3]
else:
    maze_indices = [0, 1, 2, 3]
    if n_goals < 2:
        goal_indices = [0]
    else:
        goal_indices = [0, 1]

validation_set = PolicyValidationSet(
    data=dataset, dim=repr_dim, maze_sps=maze_sps, maze_indices=maze_indices, goal_indices=goal_indices, subsample=args.subsample,
    # spatial_encoding=args.spatial_encoding,
    tile_mazes=args.tile_mazes,
    connected_tiles=args.connected_tiles,  # NOTE: viz set currently does not use this
    encoding_func=encoding_func, device=device
)

trainloader, testloader = create_train_test_dataloaders(
    data=dataset, n_train_samples=args.n_train_samples, n_test_samples=args.n_test_samples,
    input_noise=args.input_noise,
    shift_noise=args.shift_noise,
    maze_sps=maze_sps, args=args, n_mazes=args.n_mazes,
    tile_mazes=args.tile_mazes,
    connected_tiles=args.connected_tiles,
    encoding_func=encoding_func, pin_memory=pin_memory
)

# Reset seeds here after generating data
torch.manual_seed(args.seed)
np.random.seed(args.seed)


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

model.to(device)


if args.load_saved_model:
    if 'frozen-learned' in args.spatial_encoding:
        # TODO: make sure this is working correctly
        print("Loading learned first layer parameters from pretrained model")
        state_dict = torch.load(args.load_saved_model)

        for name, param in state_dict.items():
            if name in ['encoding_layer.weight', 'encoding_layer.bias']:
                model.state_dict()[name].copy_(param)

        print("Freezing first layer parameters for training")
        for name, param in model.named_parameters():
            if name in ['encoding_layer.weight', 'encoding_layer.bias']:
                param.requires_grad = False
            if param.requires_grad:
                print(name)
    else:
        model.load_state_dict(torch.load(args.load_saved_model), strict=False)
elif args.spatial_encoding == 'frozen-learned':
    raise NotImplementedError("Must select a model to load from when using frozen-learned spatial encoding")

# Open a tensorboard writer if a logging directory is given
if args.logdir != '':
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    save_dir = os.path.join(logdir, current_time)
    writer = SummaryWriter(log_dir=save_dir)
    if args.weight_histogram:
        # Log the initial parameters
        for name, param in model.named_parameters():
            writer.add_histogram('parameters/' + name, param.clone().cpu().data.numpy(), 0)


validation_set.run_ground_truth(writer=writer)

# criterion = nn.MSELoss()
cosine_criterion = nn.CosineEmbeddingLoss()
mse_criterion = nn.MSELoss()

optim_params = add_weight_decay(model, args.weight_decay, decay_phis=args.decay_phis)

if args.optimizer == 'sgd':
    optimizer = torch.optim.SGD(
        # model.parameters(),
        optim_params,
        lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )
elif args.optimizer == 'rmsprop':
    optimizer = torch.optim.RMSprop(
        # model.parameters(),
        optim_params,
        lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )
elif args.optimizer == 'adam':
    optimizer = torch.optim.Adam(
        # model.parameters(),
        optim_params,
        lr=args.lr, weight_decay=args.weight_decay
    )
else:
    raise NotImplementedError

model.train()
for e in range(args.epoch_offset, args.epochs + args.epoch_offset):
    print('Epoch: {0}'.format(e + 1))

    if e % args.viz_period == 0:
        print("Running Viz Set")
        # do a validation run and save images
        validation_set.run_validation(model, writer, e, use_wall_overlay=not args.no_wall_overlay)

        if e > 0:
            # Save a copy of the model at this stage
            torch.save(model.state_dict(), os.path.join(save_dir, 'model_epoch_{}.pt'.format(e)))

    # Run the test set for validation
    if e % args.val_period == 0:
        print("Running Val Set")
        avg_test_mse_loss = 0
        avg_test_cosine_loss = 0
        n_test_batches = 0
        with torch.no_grad():
            model.eval()
            # Everything is in one batch, so this loop will only happen once
            for i, data in enumerate(testloader):
                maze_loc_goal_ssps, directions, locs, goals = data

                outputs = model(maze_loc_goal_ssps.to(device))

                mse_loss = mse_criterion(outputs, directions.to(device))
                cosine_loss = cosine_criterion(
                    outputs,
                    directions.to(device),
                    torch.ones(maze_loc_goal_ssps.size()[0]).to(device)
                )

                avg_test_mse_loss += mse_loss.data.item()
                avg_test_cosine_loss += cosine_loss.data.item()
                n_test_batches += 1
            model.train()

        if n_test_batches > 0:
            avg_test_mse_loss /= n_test_batches
            avg_test_cosine_loss /= n_test_batches
            print(avg_test_mse_loss, avg_test_cosine_loss)
            writer.add_scalar('test_mse_loss', avg_test_mse_loss, e)
            writer.add_scalar('test_cosine_loss', avg_test_cosine_loss, e)

    avg_mse_loss = 0
    avg_cosine_loss = 0
    n_batches = 0
    for i, data in enumerate(trainloader):
        maze_loc_goal_ssps, directions, locs, goals = data

        if maze_loc_goal_ssps.size()[0] != args.batch_size:
            continue  # Drop data, not enough for a batch
        optimizer.zero_grad()

        outputs = model(maze_loc_goal_ssps.to(device))

        mse_loss = mse_criterion(outputs, directions.to(device))
        cosine_loss = cosine_criterion(
            outputs,
            directions.to(device),
            torch.ones(args.batch_size).to(device)
        )
        # print(loss.data.item())
        avg_mse_loss += mse_loss.data.item()
        avg_cosine_loss += cosine_loss.data.item()
        n_batches += 1

        if args.loss_function == 'mse':
            mse_loss.backward()
        elif args.loss_function == 'cosine':
            cosine_loss.backward()

        optimizer.step()

    if args.logdir != '':
        if n_batches > 0:
            avg_mse_loss /= n_batches
            avg_cosine_loss /= n_batches
            print(avg_mse_loss, avg_cosine_loss)
            writer.add_scalar('avg_mse_loss', avg_mse_loss, e + 1)
            writer.add_scalar('avg_cosine_loss', avg_cosine_loss, e + 1)

        if args.weight_histogram and (e + 1) % 10 == 0:
            for name, param in model.named_parameters():
                writer.add_histogram('parameters/' + name, param.clone().cpu().data.numpy(), e + 1)


print("Testing")
avg_test_mse_loss = 0
avg_test_cosine_loss = 0
n_test_batches = 0
with torch.no_grad():
    model.eval()
    # Everything is in one batch, so this loop will only happen once
    for i, data in enumerate(testloader):
        maze_loc_goal_ssps, directions, locs, goals = data

        outputs = model(maze_loc_goal_ssps.to(device))

        mse_loss = mse_criterion(outputs, directions.to(device))
        cosine_loss = cosine_criterion(
            outputs,
            directions.to(device),
            torch.ones(maze_loc_goal_ssps.size()[0]).to(device)
        )

        avg_test_mse_loss += mse_loss.data.item()
        avg_test_cosine_loss += cosine_loss.data.item()
        n_test_batches += 1

if n_test_batches > 0:
    avg_test_mse_loss /= n_test_batches
    avg_test_cosine_loss /= n_test_batches
    print(avg_test_mse_loss, avg_test_cosine_loss)
    writer.add_scalar('test_mse_loss', avg_test_mse_loss, args.epochs + args.epoch_offset)
    writer.add_scalar('test_cosine_loss', avg_test_cosine_loss, args.epochs + args.epoch_offset)


print("Visualization")
validation_set.run_validation(model, writer, args.epochs + args.epoch_offset, use_wall_overlay=not args.no_wall_overlay)


# Close tensorboard writer
if args.logdir != '':
    writer.close()

    torch.save(model.state_dict(), os.path.join(save_dir, 'model.pt'))

    params = vars(args)

    # if random-sp is used as maze-id-type, then save the sps used as well
    if args.maze_id_type == 'random-sp' and args.maze_id_dim > 0:
        params['maze_sps'] = [list(maze_sps[mi, :]) for mi in range(n_mazes)]

    with open(os.path.join(save_dir, "params.json"), "w") as f:
        json.dump(params, f)

print("Running evaluation script")
if args.out_file == '':
    print("No eval file given, skipping dataframe generation")
else:
    print("Generating data for:")
    print(args.out_file)

    validation_set = PolicyEvaluation(
        data=dataset, dim=repr_dim, maze_sps=maze_sps,
        encoding_func=encoding_func, device=device,
        n_train_samples=args.n_train_samples,
        n_test_samples=args.n_test_samples,
        spatial_encoding=args.spatial_encoding,
        n_mazes=args.n_mazes,
        tile_mazes=args.tile_mazes,
        connected_tiles=args.connected_tiles,
    )

    # Reset seeds here after generating data
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # avg_rmse_train, avg_angle_rmse_train, avg_rmse_test, avg_angle_rmse_test = validation_set.get_rmse(model)
    # train_r2, test_r2 = validation_set.get_r2_score(model)

    model.eval()
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
    # df['Dataset'] = args.dataset  # mixed
    # df['Trained On'] = args.trained_on
    df['Epochs'] = args.epochs
    df['Batch Size'] = args.batch_size
    df['Number of Mazes'] = args.n_mazes
    df['Loss Function'] = args.loss_function
    df['Learning Rate'] = args.lr
    df['Momentum'] = args.momentum
    df['Weight Decay'] = args.weight_decay
    df['Optimizer'] = args.optimizer
    df['Input Noise'] = args.input_noise
    df['Shift Noise'] = args.shift_noise

    df.to_csv(args.out_file)
