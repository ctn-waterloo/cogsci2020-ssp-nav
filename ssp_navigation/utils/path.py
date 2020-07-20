# allow code to work on machines without a display
import matplotlib
import os
display = os.environ.get('DISPLAY')
if display is None or 'localhost' in display:
    matplotlib.use('agg')

import numpy as np
import matplotlib.pyplot as plt
from spatial_semantic_pointers.utils import generate_region_vector
from gridworlds.maze_generation import generate_maze

from skimage.draw import line, line_aa

# Up, Down, Left, Right
U = 1
L = 2
D = 3
R = 4

example_path = np.array([
    [R, R, R, R, R, D],
    [U, U, 0, R, R, 0],
    [U, 0, 0, 0, 0, 0],
    [U, L, L, L, L, L],
    [U, 0, 0, 0, 0, 0],
    [U, L, L, L, L, L],
])

example_xs = np.linspace(-1, 1, example_path.shape[0])
example_ys = np.linspace(-1, 1, example_path.shape[1])


def dir_to_vec(direction):
    if direction == U:
        return np.array([0, 1])
    elif direction == L:
        return np.array([-1, 0])
    elif direction == R:
        return np.array([1, 0])
    elif direction == D:
        return np.array([0, -1])
    elif direction == 0:
        return np.array([0, 0])
    else:
        raise NotImplementedError


def path_function(coord, path, xs, ys):
    ind_x = (np.abs(xs - coord[0])).argmin()
    ind_y = (np.abs(ys - coord[1])).argmin()

    return dir_to_vec(path[ind_x, ind_y])


def plot_path_predictions(directions, coords, name='', min_val=-1, max_val=1,
                          type='quiver', dcell=1, ax=None, wall_overlay=None):
    """
    plot direction predictions by colouring based on direction, and putting the dot at the coord
    both directions and coords are (n_samples, 2) vectors
    dcell: distance between two neighboring cells, used for scaling arrors on a quiver plot
    ax: if given plot on this axis, otherwise create a new figure and axis
    wall_overlay: if set, plot the specified locations as a wall rather than whatever the prediction is
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None

    if type == 'quiver':
        if wall_overlay is not None:
            for n in range(directions.shape[0]):
                if wall_overlay[n]:
                    directions[n, 0] = 0
                    directions[n, 1] = 0
        ax.quiver(coords[:, 0], coords[:, 1], directions[:, 0], directions[:, 1], scale=1./dcell, units='xy')
    elif type == 'colour':
        for n in range(directions.shape[0]):
            x = coords[n, 0]
            y = coords[n, 1]

            if wall_overlay is not None and wall_overlay[n]:
                # Set to wall value if an overlay is requested here
                xa = 0
                ya = 0
            else:
                # Note: this clipping shouldn't be necessary
                xa = np.clip(directions[n, 0], min_val, max_val)
                ya = np.clip(directions[n, 1], min_val, max_val)

            r = float(((xa - min_val) / (max_val - min_val)))
            # g = float(((ya - min_val) / (max_val - min_val)))
            b = float(((ya - min_val) / (max_val - min_val)))

            # ax.scatter(x, y, color=(r, g, 0))
            ax.scatter(x, y, color=(r, 0, b))
    else:
        raise NotImplementedError

    if name:
        fig.suptitle(name)

    return fig


def plot_path_predictions_image(directions_pred, directions_true, name='', ax=None, wall_overlay=None, trim=False):

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None

    angles_flat_pred = np.arctan2(directions_pred[:, 1], directions_pred[:, 0])
    angles_flat_true = np.arctan2(directions_true[:, 1], directions_true[:, 0])

    # Create 3 possible offsets to cover all cases
    angles_offset_true = np.zeros((len(angles_flat_true), 3))
    angles_offset_true[:, 0] = angles_flat_true - 2 * np.pi
    angles_offset_true[:, 1] = angles_flat_true
    angles_offset_true[:, 2] = angles_flat_true + 2 * np.pi

    angles_offset_true -= angles_flat_pred.reshape(len(angles_flat_pred), 1)
    angles_offset_true = np.abs(angles_offset_true)

    angle_error = np.min(angles_offset_true, axis=1)

    angle_squared_error = angle_error**2
    if wall_overlay is not None:
        angle_rmse = np.sqrt(angle_squared_error[np.where(wall_overlay == 0)].mean())
    else:
        angle_rmse = np.sqrt(angle_squared_error.mean())

    # NOTE: this assumes the data can be reshaped into a perfect square
    size = int(np.sqrt(angles_flat_pred.shape[0]))

    angles_pred = angles_flat_pred.reshape((size, size))
    if trim:
        angles_pred = angles_pred[:-5, :-5]

    ax.imshow(angles_pred, cmap='hsv', interpolation=None)

    if wall_overlay is not None:
        # Overlay black as the walls, use transparent everywhere else
        wall_locations = wall_overlay.reshape((size, size))
        overlay = np.zeros((size, size, 4)).astype(np.uint8)

        for i in range(size):
            for j in range(size):
                if wall_locations[i, j]:
                    overlay[i, j, 3] = 255
                else:
                    overlay[i, j, :] = 0

        if trim:
            overlay = overlay[:-5, :-5, :]
        ax.imshow(overlay, interpolation=None)

    sin = np.sin(angles_flat_pred)
    cos = np.cos(angles_flat_pred)

    pred_dir_normalized = np.vstack([cos, sin]).T

    squared_error = (pred_dir_normalized - directions_true)**2

    # only calculate mean across the non-wall elements
    # mse = np.mean(squared_error[np.where(wall_overlay == 0)])
    mse = squared_error[np.where(wall_overlay == 0)].mean()

    rmse = np.sqrt(mse)

    print("rmse", rmse)
    print("angle rmse", angle_rmse)

    if fig is not None:
        fig.suptitle("RMSE: {}".format(angle_rmse))

        if name:
            fig.suptitle(name)
    else:
        ax.set_title("RMSE: {}".format(angle_rmse))

    return fig, angle_rmse


def get_path_predictions_image(directions_pred, directions_true, wall_overlay=None):

    angles_flat_pred = np.arctan2(directions_pred[:, 1], directions_pred[:, 0])
    angles_flat_true = np.arctan2(directions_true[:, 1], directions_true[:, 0])

    # Create 3 possible offsets to cover all cases
    angles_offset_true = np.zeros((len(angles_flat_true), 3))
    angles_offset_true[:, 0] = angles_flat_true - 2 * np.pi
    angles_offset_true[:, 1] = angles_flat_true
    angles_offset_true[:, 2] = angles_flat_true + 2 * np.pi

    angles_offset_true -= angles_flat_pred.reshape(len(angles_flat_pred), 1)
    angles_offset_true = np.abs(angles_offset_true)

    angle_error = np.min(angles_offset_true, axis=1)

    angle_squared_error = angle_error**2
    if wall_overlay is not None:
        angle_rmse = np.sqrt(angle_squared_error[np.where(wall_overlay == 0)].mean())
    else:
        angle_rmse = np.sqrt(angle_squared_error.mean())

    # NOTE: this assumes the data can be reshaped into a perfect square
    size = int(np.sqrt(angles_flat_pred.shape[0]))

    angles_pred = angles_flat_pred.reshape((size, size))
    angles_true = angles_flat_true.reshape((size, size))

    if wall_overlay is not None:
        # Overlay black as the walls, use transparent everywhere else
        wall_locations = wall_overlay.reshape((size, size))
        overlay = np.zeros((size, size, 4)).astype(np.uint8)

        for i in range(size):
            for j in range(size):
                if wall_locations[i, j]:
                    overlay[i, j, 3] = 255
                else:
                    overlay[i, j, :] = 0
    else:
        overlay = None

    sin = np.sin(angles_flat_pred)
    cos = np.cos(angles_flat_pred)

    pred_dir_normalized = np.vstack([cos, sin]).T

    squared_error = (pred_dir_normalized - directions_true)**2

    # only calculate mean across the non-wall elements
    # mse = np.mean(squared_error[np.where(wall_overlay == 0)])
    mse = squared_error[np.where(wall_overlay == 0)].mean()

    rmse = np.sqrt(mse)

    return angles_pred, angles_true, overlay, rmse, angle_rmse


def generate_maze_sp(size, xs, ys, x_axis_sp, y_axis_sp, normalize=True, obstacle_ratio=.2, map_style='blocks'):
    """
    Returns a maze ssp as well as an occupancy grid for the maze (both fine and coarse)
    """

    # Create a random maze with no inaccessible regions
    if map_style == 'maze':
        # temp hack to get the sizes right
        maze = generate_maze(map_style=map_style, side_len=size + 2, obstacle_ratio=obstacle_ratio)
    else:
        maze = generate_maze(map_style=map_style, side_len=size, obstacle_ratio=obstacle_ratio)

    # Map the maze structure to the resolution of xs and ys
    x_range = xs[-1] - xs[0]
    y_range = ys[-1] - ys[0]
    fine_maze = np.zeros((len(xs), len(ys)))

    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            xi = np.clip(int(np.round(((x - xs[0]) / x_range) * size)), 0, size - 1)
            yi = np.clip(int(np.round(((y - ys[0]) / y_range) * size)), 0, size - 1)
            # If the corresponding location on the coarse maze is a wall, make it a wall on the fine maze as well
            if maze[xi, yi] == 1:
                fine_maze[i, j] = 1

    # Create a region vector for the maze
    sp = generate_region_vector(
        desired=fine_maze, xs=xs, ys=ys,
        x_axis_sp=x_axis_sp, y_axis_sp=y_axis_sp, normalize=normalize
    )

    return sp, maze, fine_maze


def expand_node(distances, solved_maze, maze, node, wall_value=1000, strict_cornering=False):
    current_value = distances[node[0], node[1]]

    # Generate list of indices for all nodes around the current node
    # NOTE: shouldn't need a bounds check since the edge of all mazes is walls
    dist_sorted_indices = np.dstack(np.unravel_index(np.argsort(distances.ravel()), distances.shape))

    checks = [
        (node[0] + 1, node[1], current_value + 1),
        (node[0] - 1, node[1], current_value + 1),
        (node[0], node[1] + 1, current_value + 1),
        (node[0], node[1] - 1, current_value + 1),
    ]

    new_nodes = []

    for c in checks:
        x, y, value = c
        if (0 <= x < distances.shape[0]) and (0 <= y < distances.shape[1]):
            if distances[x, y] != 2*wall_value and (distances[x, y] == -1 or distances[x, y] > value):
                new_nodes.append((x, y))
                # space is free, now find the shortest distance to get here
                # attempt to draw a line to the closest nodes to the goal
                for i in range(dist_sorted_indices.shape[1]):  # note that the first dimension has shape 1, using 2nd
                    # attempt to draw line
                    if strict_cornering:
                        rr, cc, _ = line_aa(int(x), int(y), int(dist_sorted_indices[0, i, 0]), int(dist_sorted_indices[0, i, 1]))
                    else:
                        rr, cc = line(int(x), int(y), int(dist_sorted_indices[0, i, 0]), int(dist_sorted_indices[0, i, 1]))

                    if ((len(rr) > 2) and np.all(distances[rr[1:-1], cc[1:-1]] < 2*wall_value)) or np.all(distances[rr, cc] < 2*wall_value):

                        x_best = dist_sorted_indices[0, i, 0]
                        y_best = dist_sorted_indices[0, i, 1]
                        # TODO: should there be a version that uses the displacement vector rather than normalizing direction?
                        direction = np.array([x_best - x, y_best - y]).astype(np.float32)
                        direction /= np.linalg.norm(direction)
                        distance = np.sqrt((x_best - x)**2 + (y_best - y)**2)

                        # distance to the best node + the distance from the best node to the goal
                        distances[x, y] = distance + distances[x_best, y_best]
                        # direction to the best node
                        solved_maze[x, y, :] = direction
                        # print(direction)
                        break
                assert(distances[dist_sorted_indices[0, i, 0], dist_sorted_indices[0, i, 1]] < wall_value)

    return new_nodes


# TODO: make some tests and visualizations to make sure this function is doing the correct thing
def solve_maze(maze, start_indices, goal_indices, full_solve=False, wall_value=1000, strict_cornering=False):

    # Direction to move for each cell in the maze
    solved_maze = np.zeros((maze.shape[0], maze.shape[1], 2))

    # distances from every point to the goal, along the shortest path
    # value of 'wall_value' corresponds to unknown. Large value (double wall_value) corresponds to impassable wall
    # both unknown and wall are set to high values so sorting by distance can work correctly
    distances = wall_value * np.ones((maze.shape[0], maze.shape[1]))

    distances[goal_indices[0], goal_indices[1]] = 0

    distances += wall_value * maze

    # Algorithm:
    # Each expanded node will be assigned a distance to the goal, as well as a direction to move to get to the goal
    # To assign distance, find the closest already expanded node to the goal
    # that a straight line can be drawn to from the current node
    # Add the distance of that line to the distance recorded in that node to be the distance of this new node
    # The direction will be the direction of the line
    # Maintain an ordered list of closest nodes to make checking easier
    # Start off with the goal node as expanded with distance of 0.
    # Expand only direct neighbors of expanded nodes (direct adjacent, not including diagonals)
    # Can optionally stop once the start node is expanded and assigned a direction

    to_expand = [goal_indices]

    while len(to_expand) > 0:
        next_node = to_expand.pop()
        new_nodes = expand_node(
            distances=distances,
            solved_maze=solved_maze,
            maze=maze,
            node=next_node,
            wall_value=wall_value,
            strict_cornering=strict_cornering,
        )
        to_expand += new_nodes

        # break out early if the start has been found
        if not full_solve and (next_node[0] == start_indices[0]) and (next_node[1] == start_indices[1]):
            break

    return solved_maze


def test_solve_maze():
    maze = np.array([
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 1, 0, 1, 0, 1],
        [1, 0, 0, 1, 0, 0, 0, 1],
        [1, 1, 1, 1, 0, 0, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
    ])
    maze = np.array([
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
    ])

    solved_maze = solve_maze(maze=maze, start_indices=np.array([1, 1]), goal_indices=np.array([6, 1]), full_solve=True)

    directions = np.zeros((maze.shape[0]*maze.shape[1], 2))
    locs = np.zeros((maze.shape[0]*maze.shape[1], 2))

    for i in range(maze.shape[0]):
        for j in range(maze.shape[1]):
            directions[i*maze.shape[1] + j, :] = solved_maze[i, j, :]
            locs[i * maze.shape[1] + j, :] = np.array([i, j])

    fig, ax = plt.subplots()
    q = ax.quiver(locs[:, 0], locs[:, 1], directions[:, 0], directions[:, 1], scale=2, units='xy')
    plt.show()


def test_generate_maze_sp(size=10, limit_low=-5, limit_high=5, res=64, dim=512, seed=13):
    from spatial_semantic_pointers.utils import make_good_unitary, get_heatmap_vectors
    from spatial_semantic_pointers.plots import plot_heatmap

    rng = np.random.RandomState(seed=seed)

    x_axis_sp = make_good_unitary(dim=dim, rng=rng)
    y_axis_sp = make_good_unitary(dim=dim, rng=rng)

    xs = np.linspace(limit_low, limit_high, res)
    ys = np.linspace(limit_low, limit_high, res)

    sp, maze, fine_maze = generate_maze_sp(
        size, xs, ys, x_axis_sp, y_axis_sp, normalize=True, obstacle_ratio=.2, map_style='blocks'
    )

    fig, ax = plt.subplots(1, 4)

    ax[0].imshow(maze)
    ax[1].imshow(fine_maze)
    heatmap_vectors = get_heatmap_vectors(xs, ys, x_axis_sp, y_axis_sp)
    plot_heatmap(sp.v, heatmap_vectors, ax[2], xs, ys, name='', vmin=-1, vmax=1, cmap='plasma', invert=True)
    plot_heatmap(sp.v, heatmap_vectors, ax[3], xs, ys, name='', vmin=None, vmax=None, cmap='plasma', invert=True)

    plt.show()


if __name__ == '__main__':
    # Run some tests
    # test_solve_maze()
    test_generate_maze_sp(dim=512, seed=13, res=256)
    # test_generate_maze_sp(dim=2048, seed=13)

