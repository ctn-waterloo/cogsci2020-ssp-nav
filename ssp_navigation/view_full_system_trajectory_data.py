import argparse
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

parser = argparse.ArgumentParser('Run the full system on a maze task and record the trajectories taken')

parser.add_argument('--fname', type=str, default='fs_traj_data.npz', help='file to save the trajectory data to')
parser.add_argument('--maze-index', type=int, default=0, help='index within the dataset for the maze to use')
parser.add_argument('--dataset', type=str,
                    default='datasets/mixed_style_100mazes_100goals_64res_13size_13seed/maze_dataset.npz',
                    help='dataset to get the maze layout from')

parser.add_argument('--start-x', type=float, default=0, help='x-coord of the agent start location')
parser.add_argument('--start-y', type=float, default=0, help='y-coord of the agent start location')
parser.add_argument('--goal-x', type=float, default=0, help='x-coord of the goal location')
parser.add_argument('--goal-y', type=float, default=0, help='y-coord of the goal location')

parser.add_argument('--plot-icons', action='store_true', help='plot icons for the agent start and for each goal')

args = parser.parse_args()

dataset = np.load(args.dataset)

limit_low = 0
limit_high = dataset['coarse_mazes'].shape[2]

map_array = dataset['coarse_mazes'][args.maze_index, :, :]


data = np.load(args.fname)

# shape is n_trajectories, n_steps, 2
# or shape is n_goals, n_trajectories, n_steps, 2
locations = data['locations']

if len(locations.shape) == 3:
    multigoal = False
    n_goals = 1
    n_trajectories = locations.shape[0]
    n_steps = locations.shape[1]
else:
    multigoal = True
    n_goals = locations.shape[0]
    n_trajectories = locations.shape[1]
    n_steps = locations.shape[2]

# print(np.max(locations))
# print(np.min(locations))

fig, ax = plt.subplots(tight_layout=True, figsize=(4, 4))

# plt.imshow(1 - map_array.T, cmap='gray')
ax.imshow(1 - map_array.T, cmap='gray')

if multigoal:
    colours = sns.color_palette(n_colors=n_goals)
    for g in range(n_goals):
        for i in range(n_trajectories):
            indices = np.where((locations[g, i, :, 0] != 0) | (locations[g, i, :, 1] != 0))
            print(len(indices[0]))
            ax.plot(locations[g, i, indices, 0][0], locations[g, i, indices, 1][0], color=colours[g], alpha=min(1, 2./n_trajectories))
else:

    for i in range(n_trajectories):
        indices = np.where((locations[i, :, 0] != 0) | (locations[i, :, 1] != 0))
        print(len(indices[0]))
        # plt.plot(locations[i, indices, 0][0], locations[i, indices, 1][0])
        ax.plot(locations[i, indices, 0][0], locations[i, indices, 1][0])
        # plt.plot(locations[i, :, 0], locations[i, :, 1])


if args.plot_icons:
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox

    def getImage(path, zoom=0.25):
        if '64' in path:
            final_zoom = zoom*2
        elif '96' in path:
            final_zoom = zoom*1.5
        else:
            final_zoom = zoom
        return OffsetImage(plt.imread(path), zoom=final_zoom)


    # TODO: use different images for this paper. Have them be household objects or food
    # root = '/home/bjkomer/cogsci2019-ssp/images'
    root = 'images'
    paths = [
        root + '/icons8-mouse-96.png',
        root + '/icons8-wine-bottle-64.png',
        root + '/icons8-cheese-128.png',
        root + '/icons8-watermelon-64.png',
        root + '/icons8-paprika-64.png',
        root + '/icons8-grape-64.png',
    ]

    goals = np.array(
        [
            [6, 6], # mouse
            [11, 2],  # [10.4, 2.4], #[10, 3.5],  #blue
            [6, 10], #[6, 10.4],  # [6, 10],  # orange
            [1, 1.3],#[1, 1.1],  # [1.5, 2.5], #[2, 3],  # green
            [10.5, 8.9], #[10.7, 9.0],  # [10, 9],  # red
            [1, 11],  # [4, 11.3], #[2, 11]  # purple
        ]
    )

    y = goals[:, 0]
    x = goals[:, 1]

    # x = np.array([3.5, 9, 3, 11, 1])
    # y = np.array([10.5, 2, 3, 10, 7])

    ax.scatter(y, x)
    # ax.set_xlim([-5, 5])
    # ax.set_ylim([-5, 5])
    ax.set_aspect('equal', 'box')

    artists = []
    for x0, y0, path in zip(x, y, paths):
        ab = AnnotationBbox(getImage(path), (y0, x0), frameon=False)
        artists.append(ax.add_artist(ab))

ax.set_axis_off()

plt.show()
