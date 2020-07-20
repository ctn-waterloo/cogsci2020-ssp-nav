import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(
    'View a full tiled maze'
)

parser.add_argument('--n-mazes', type=int, default=100, help='Number of mazes from the dataset to train with')
parser.add_argument('--dataset-dir', type=str, default='datasets/mixed_style_100mazes_100goals_64res_13size_13seed')

args = parser.parse_args()

dataset_file = os.path.join(args.dataset_dir, 'maze_dataset.npz')

data = np.load(dataset_file)

# n_mazes by res by res
fine_mazes = data['fine_mazes'][:args.n_mazes, :, :]

res = fine_mazes.shape[1]
side_len = int(np.sqrt(args.n_mazes))

# the +1 is an offset to make the image look nicer
full_maze = np.ones((1+side_len*res, 1+side_len*res))

for x in range(side_len):
    for y in range(side_len):
        full_maze[1+x*res:1+(x+1)*res, 1+y*res:1+(y+1)*res] = fine_mazes[x*side_len + y, :, :]


# colour in a very visible overlay of the heirarchical map
overlay = False#True

# constants for getting the offsets for connecting mazes correct
to_edge = res - 3
to_center = int(res/2)
# show the passageways
# create connections between mazes
# left-right connections
for i in range(side_len - 1):
    for j in range(side_len):
        x = i*res + to_edge
        y = j*res + to_center - 1
        # check if a connection will be valid
        # if np.all(full_maze[x:x+6, y] == np.array([0, 1, 1, 1, 1, 0])):
        if full_maze[1+x-6, y] == 0 and full_maze[1+x+6, y] == 0:
            # set pathway to open
            full_maze[1+x-6:1+x+7, 1+y-3:1+y] = 0
            # full_maze[1+x - 6-res//2:1+x + 7+res//2, 1+y-3:1+y] = .5  # for viewing/debugging
            if overlay:
                full_maze[1 + x - 6 - res // 2:1 + x + 7 + res // 2, 1 + y -res//8:1 + y + res//8] = .5


# up-down connections
for j in range(side_len - 1):
    for i in range(side_len):
        x = i*res + to_center
        y = j*res + to_edge
        # check if a connection will be valid
        # if np.all(full_maze[x, y:y+6] == np.array([0, 1, 1, 1, 1, 0])):
        if full_maze[1+x, 1+y - 6] == 0 and full_maze[1+x, 1+y + 6] == 0:
            # set pathway to open
            full_maze[1+x:1+x+3, 1+y-6:1+y+7] = 0
            # full_maze[1+x:1+x + 3, 1+y - 6 - res//2:1+y + 7 + res//2] = .5  # for viewing/debugging
            if overlay:
                full_maze[1 + x-res//8:1 + x+res//8, 1 + y - 6 - res // 2:1 + y + 7 + res // 2] = .5

fig, ax = plt.subplots(1, 1, tight_layout=True, figsize=(4, 4))

if overlay:
    # with colour for the overlay
    maze_image = np.zeros((full_maze.shape[0] - 4, full_maze.shape[1] - 4, 3))

    maze_image[:, :, 0] = 1 - full_maze[:-4, :-4]
    maze_image[:, :, 1] = 1 - full_maze[:-4, :-4]
    maze_image[:, :, 2] = 1 - full_maze[:-4, :-4]

    overlay_indices = np.where(maze_image[:, :, :] == .5)
    print(overlay_indices)

    # print(maze_image[overlay_indices, 0])
    maze_image[overlay_indices[0], overlay_indices[1], 0] = 1
    maze_image[overlay_indices[0], overlay_indices[1], 1] = 0.25
    maze_image[overlay_indices[0], overlay_indices[1], 2] = 0.25
    # maze_image[overlay_indices, 0] = 0
    # maze_image[overlay_indices, 1] = 0
    # maze_image[overlay_indices, 2] = 0.2

    ax.imshow(maze_image)
else:
    # the :-4 is an offset to make the image look nicer
    ax.imshow(1 - full_maze[:-4, :-4], cmap='gray')
    # plt.figure()
    # plt.imshow(1 - full_maze, cmap='gray')

ax.axis('off')
plt.show()
