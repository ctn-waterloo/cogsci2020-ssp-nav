import torch
import torch.nn as nn
import numpy as np
import argparse

from ssp_navigation.utils.models import MLP, LearnedEncoding
from ssp_navigation.utils.datasets import MazeDataset
import nengo.spa as spa
import matplotlib.pyplot as plt
from ssp_navigation.utils.path import plot_path_predictions_image
from spatial_semantic_pointers.utils import encode_point

# TODO: make it so the user can click on the image, and the location clicked is the goal

parser = argparse.ArgumentParser(
    'View a policy for an interactive goal location on a particular map'
)

parser.add_argument('--map-index', type=int, default=0, help='Index of the map in the dataset to use')
parser.add_argument('--seed', type=int, default=13, help='Seed for training and generating axis SSPs')
parser.add_argument('--limit-low', type=float, default=0, help='lowest coordinate value')
parser.add_argument('--limit-high', type=float, default=13, help='highest coordinate value')
parser.add_argument('--hidden-size', type=int, default=512)
parser.add_argument('--n-hidden-layers', type=int, default=1)
# parser.add_argument('--view-activations', action='store_true', help='view spatial activations of each neuron')
parser.add_argument('--dim', type=int, default=512, help='Dimensionality of the SSPs')
parser.add_argument('--spatial-encoding', type=str, default='ssp',
                    choices=['ssp', 'random', '2d', '2d-normalized', 'one-hot', 'trig', 'random-proj', 'random-trig', 'learned'],
                    help='coordinate encoding for agent location and goal')
parser.add_argument('--maze-id-type', type=str, choices=['ssp', 'one-hot', 'random-sp'], default='one-hot',
                    help='ssp: region corresponding to maze layout.'
                         'one-hot: each maze given a one-hot vector.'
                         'random-sp: each maze given a unique random SP as an ID')
parser.add_argument('--subsample', type=int, default=1, help='amount to subsample for the visualization validation')
parser.add_argument(
    '--dataset', type=str,
    default='maze_datasets/maze_dataset_maze_style_10mazes_25goals_64res_13size_13seed.npz',
)

parser.add_argument('--load-saved-model', type=str, default='', help='Saved model to load from')

args = parser.parse_args()

# res = 64
subsample = 1

data = np.load(args.dataset)

rng = np.random.RandomState(seed=args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# n_mazes by res by res
fine_mazes = data['fine_mazes']

res = fine_mazes.shape[1]

# n_mazes by res by res by 2
solved_mazes = data['solved_mazes']

# n_mazes by dim
maze_sps = data['maze_sps']

# FIXME: temp hardcoding for hierarchical policy
# maze_sps = np.eye(5, 5)

# n_mazes by n_goals by dim
goal_sps = data['goal_sps']

# n_mazes by n_goals by 2
goals = data['goals']

# xs = data['xs']
# ys = data['ys']

x_axis_vec = data['x_axis_sp']
y_axis_vec = data['y_axis_sp']
x_axis_sp = spa.SemanticPointer(data=x_axis_vec)
y_axis_sp = spa.SemanticPointer(data=y_axis_vec)

# limit_low = 0
# limit_high = 13

xs = np.linspace(args.limit_low, args.limit_high, res)
ys = np.linspace(args.limit_low, args.limit_high, res)

n_goals = goals.shape[1]
n_mazes = fine_mazes.shape[0]

wall_overlay = fine_mazes[args.map_index, :].flatten()



if args.maze_id_type == 'ssp':
    id_size = args.dim
elif args.maze_id_type == 'one-hot':
    id_size = n_mazes
    # overwrite data
    maze_sps = np.eye(n_mazes)
else:
    raise NotImplementedError


# Dimension of location representation is dependent on the encoding used
if args.spatial_encoding == 'ssp':
    repr_dim = args.dim
elif args.spatial_encoding == 'random':
    repr_dim = args.dim
elif args.spatial_encoding == '2d':
    repr_dim = 2
elif args.spatial_encoding == 'learned':
    repr_dim = 2
elif args.spatial_encoding == 'frozen-learned':  # use a pretrained learned representation
    repr_dim = 2
elif args.spatial_encoding == '2d-normalized':
    repr_dim = 2
elif args.spatial_encoding == 'one-hot':
    repr_dim = int(np.sqrt(args.dim))**2
elif args.spatial_encoding == 'trig':
    repr_dim = args.dim
elif args.spatial_encoding == 'random-trig':
    repr_dim = args.dim
elif args.spatial_encoding == 'random-proj':
    repr_dim = args.dim

# input is maze, loc, goal ssps, output is 2D direction to move
if 'learned' in args.spatial_encoding:
    model = LearnedEncoding(
        input_size=repr_dim,
        maze_id_size=id_size,
        hidden_size=args.hidden_size,
        output_size=2,
        n_layers=args.n_hidden_layers
    )
else:
    model = MLP(
        input_size=id_size + repr_dim * 2,
        hidden_size=args.hidden_size,
        output_size=2,
        n_layers=args.n_hidden_layers
    )

if args.load_saved_model:
    model.load_state_dict(torch.load(args.load_saved_model), strict=False)

model.eval()

n_samples = res * res

# Visualization
viz_locs = np.zeros((n_samples, 2))
viz_goals = np.zeros((n_samples, 2))
viz_loc_sps = np.zeros((n_samples, repr_dim))
viz_goal_sps = np.zeros((n_samples, repr_dim))
viz_output_dirs = np.ones((n_samples, 2))
viz_maze_sps = np.zeros((n_samples, maze_sps.shape[1]))

# Generate data so each batch contains a single maze and goal
si = 0  # sample index, increments each time



for xi in range(0, res, subsample):
    for yi in range(0, res, subsample):
        loc_x = xs[xi]
        loc_y = ys[yi]

        viz_locs[si, 0] = loc_x
        viz_locs[si, 1] = loc_y
        # viz_goals[si, :] = goals[mi, gi, :]
        if args.spatial_encoding == 'ssp':
            viz_loc_sps[si, :] = encode_point(loc_x, loc_y, x_axis_sp, y_axis_sp).v
        # elif args.spatial_encoding == 'random':
        #     viz_loc_sps[si, :] = encode_random(loc_x, loc_y, dim)
        elif args.spatial_encoding == '2d' or args.spatial_encoding == 'learned' or args.spatial_encoding == 'frozen-learned':
            viz_loc_sps[si, :] = np.array([loc_x, loc_y])
        # elif args.spatial_encoding == '2d-normalized':
        #     viz_loc_sps[si, :] = ((np.array([loc_x, loc_y]) - limit_low) * 2 / (limit_high - limit_low)) - 1
        # elif args.spatial_encoding == 'one-hot':
        #     viz_loc_sps[si, :] = encode_one_hot(x=loc_x, y=loc_y, xs=xso, ys=yso)
        # elif args.spatial_encoding == 'trig':
        #     viz_loc_sps[si, :] = encode_trig(x=loc_x, y=loc_y, dim=dim)
        # elif args.spatial_encoding == 'random-trig':
        #     viz_loc_sps[si, :] = encode_random_trig(x=loc_x, y=loc_y, dim=dim)
        # elif args.spatial_encoding == 'random-proj':
        #     viz_loc_sps[si, :] = encode_projection(x=loc_x, y=loc_y, dim=dim)

        # viz_goal_sps[si, :] = goal_sps[mi, gi, :]
        #
        # viz_output_dirs[si, :] = solved_mazes[mi, gi, xi, yi, :]

        viz_maze_sps[si, :] = maze_sps[args.map_index]

        si += 1






fig = plt.figure()
ax = fig.add_subplot(111)

# initialize image with correct dimensions
ax.imshow(np.zeros((64, 64)), cmap='hsv', interpolation=None)


def image_coord_to_maze_coord(x, y, res=64):
    x_img = (x / res) * (xs[-1] - xs[0]) + xs[0]
    y_img = (y / res) * (ys[-1] - ys[0]) + ys[0]

    return (x_img, y_img)


def on_click(event):

    if (event.xdata is not None) and (event.ydata is not None):

        # TODO: need to make sure these coordinates are in the right reference frame, and translate them if they are not
        # goal = (event.xdata, event.ydata)
        # goal = image_coord_to_maze_coord(event.xdata, event.ydata)
        goal = image_coord_to_maze_coord(event.ydata, event.xdata)
        print(goal)

        if args.spatial_encoding == 'ssp':
            goal_ssp = encode_point(goal[0], goal[1], x_axis_sp, y_axis_sp).v
        else:
            goal_ssp = np.array([goal[0], goal[1]])

        for i in range(res*res):
            viz_goal_sps[i, :] = goal_ssp


        dataset_viz = MazeDataset(
            maze_ssp=viz_maze_sps,
            loc_ssps=viz_loc_sps,
            goal_ssps=viz_goal_sps,
            locs=viz_locs,
            goals=viz_goals,
            direction_outputs=viz_output_dirs,  # unused, but should be for wall overlay
        )

        # Each batch will contain the samples for one maze. Must not be shuffled
        vizloader = torch.utils.data.DataLoader(
            dataset_viz, batch_size=res*res, shuffle=False, num_workers=0,
        )

        for i, data in enumerate(vizloader):

            maze_loc_goal_ssps, directions, locs, goals = data

            outputs = model(maze_loc_goal_ssps)

            # NOTE: no ground truth is currently given, so no loss or RMSE can be calculated
            # loss = criterion(outputs, directions)

            # wall_overlay = (directions.detach().numpy()[:, 0] == 0) & (directions.detach().numpy()[:, 1] == 0)

            fig_pred, rmse = plot_path_predictions_image(
                ax=ax,
                directions_pred=outputs.detach().numpy(),
                directions_true=locs.detach().numpy(),
                wall_overlay=wall_overlay
            )
            ax.set_title("No ground truth given for RMSE")
            fig.canvas.draw()


fig.canvas.mpl_connect('button_press_event', on_click)
plt.show()
