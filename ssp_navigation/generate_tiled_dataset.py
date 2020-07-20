import argparse
import numpy as np
import os
from ssp_navigation.utils.path import solve_maze
from skimage.draw import line, line_aa
from ssp_navigation.utils.misc import gaussian_2d


class Graph(object):

    def __init__(self, name='graph', matrix=None, data=None):

        self.name = name

        self.nodes = []

        self.n_nodes = 0

        if matrix is not None:
            self.construct_from_matrix(matrix, data)

    def construct_from_matrix(self, matrix, data=None):
        """
        Construct a graph based on connectivity defined in a matrix
        0 means the two nodes are not connected
        any number represents the connections strength
        data is an option list of information to go with each node
        """

        assert matrix.shape[0] == matrix.shape[1]

        self.n_nodes = matrix.shape[0]

        # Add all the nodes to the graphs
        for i in range(self.n_nodes):
            if data is not None:
                self.nodes.append(Node(index=i, data=data[i]))
            else:
                self.nodes.append(Node(index=i))

            # Set the appropriate neighbors for the node
            for j in range(self.n_nodes):
                if matrix[i, j] > 0:
                    self.nodes[i].add_neighbor(j, matrix[i, j])

    def search_graph(self, start_node, end_node):
        """
        Returns the path from start_node to end_node as a list
        :param start_node: index of the start node
        :param end_node: index of the end node
        :return: list of nodes along the path
        """

        # Keep track of shortest distances to each node (from the end_node)
        distances = 1000 * np.ones((self.n_nodes,))

        distances[end_node] = 0

        to_expand = [end_node]

        finished = False

        while len(to_expand) > 0:# and not finished:
            next_node = to_expand.pop()

            for neighbor, dist in self.nodes[next_node].neighbors.items():
                # If a faster way to get to the neighbor is detected, update the distances with the
                # new shorter distance, and set this neighbor to be expanded
                if distances[next_node] + dist < distances[neighbor]:
                    distances[neighbor] = distances[next_node] + dist
                    to_expand.append(neighbor)

        # Now follow the shortest path from start to goal
        path = []

        # If the start and goal cannot be connected, return an empty list
        if distances[start_node] >= 1000:
            return []

        current_node = start_node

        while current_node != end_node:
            path.append(current_node)
            # choose the node connected to the current node with the shortest distance to the goal
            neighbors = self.nodes[current_node].get_neighbors()

            closest_n = -1
            closest_dist = 1000
            for n in neighbors:
                # print("neighbor: {0}".format(n))
                if distances[n] < closest_dist:
                    closest_dist = distances[n]
                    closest_n = n

            current_node = closest_n
            assert current_node != -1

        path.append(current_node)

        return path

    def search_graph_by_location(self, start_location, end_location):

        start_node = self.get_node_for_location(start_location)
        end_node = self.get_node_for_location(end_location)

        # self.search_graph() gives a list of indices, need to convert those to x-y locations before returning
        return [self.nodes[n].data['location'] for n in self.search_graph(start_node.index, end_node.index)]

    def get_node_for_location(self, location):

        closest_dist = 10000
        closest_node = -1
        for n in self.nodes:
            if np.all(n.data['location'] == location):
                # If there is an exact match, return it now
                return n
            else:
                # If it's not a match, see if it is the closest match
                dist = np.linalg.norm(n.data['location'] - location)
                if dist < closest_dist:
                    closest_dist = dist
                    closest_node = n

        # No match found, return the closest node
        # Due to rounding there will almost never be an exact match, so the warning message is not helpful
        # print("Warning: no matching node location found, using closest location of {0} for {1}".format(
        #     closest_node.data['location'], location
        # ))
        return closest_node

    def is_valid_path(self, path):
        """
        Given a sequence of node indices, return true if they represent a valid path in the graph and false otherwise
        :param path: list of node indices
        :return: boolean
        """

        for i in range(len(path) - 1):
            if path[i + 1] not in self.nodes[path[i]].get_neighbors():
                return False

        return True


    def plot_graph(self, ax, xlim=(0, 10), ylim=(0, 10), invert=False):
        """
        plots the graph structure of connected nodes on an a pyplot axis
        assumes that each node has a x-y location in its data
        """

        ax.cla()

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        for n in self.nodes:
            x = n.data['location'][0]
            y = n.data['location'][1]

            # Draw a dot at the node location
            if invert:
                # Invert y for plotting
                # Flip x and y axes for plotting
                ax.scatter(y, ylim[1] - x)
            else:
                ax.scatter(x, y)

            # Go through the neighbor indices of each neighbor
            for ni in n.get_neighbors():
                # Draw a line between the current node and the neighbor node
                nx = self.nodes[ni].data['location'][0]
                ny = self.nodes[ni].data['location'][1]
                if invert:
                    # Invert y for plotting
                    # Flip x and y axes for plotting
                    ax.plot([y, ny], [ylim[1] - x, ylim[1] - nx])
                else:
                    ax.plot([x, nx], [y, ny])

    def graph_image(self, xs, ys):
        """
        creates an image that represents the graph structure of connected nodes
        used for plotting as an image, so that the scale is the same as the occupancy images
        """

        img = np.zeros((len(xs), len(ys)))

        for n in self.nodes:
            x = n.data['location'][0]
            y = n.data['location'][1]

            # Index of the coordinate in the array
            # NOTE: this might be slightly incorrect....?
            ix = (np.abs(xs - x)).argmin()
            iy = (np.abs(ys - y)).argmin()

            img[ix, iy] = 1  #TEMP

            # Go through the neighbor indices of each neighbor
            for ni in n.get_neighbors():
                # Draw a line between the current node and the neighbor node
                nx = self.nodes[ni].data['location'][0]
                ny = self.nodes[ni].data['location'][1]

                inx = (np.abs(xs - nx)).argmin()
                iny = (np.abs(ys - ny)).argmin()

                # rr, cc = line(ix, iy, inx, iny)

                # anti-aliased line
                rr, cc, val = line_aa(ix, iy, inx, iny)

                img[rr, cc] = val


        return img


class Node(object):

    def __init__(self, index, data=None):

        self.index = index

        # key value pairs of neighbors
        # keys are the unique index of the neighbor, value is the distance
        self.neighbors = {}

        self.data = data

    def add_neighbor(self, neighbor_index, distance):

        self.neighbors[neighbor_index] = distance

    def get_neighbors(self):

        return list(self.neighbors.keys())

parser = argparse.ArgumentParser(
    'Generate a dataset with a giant maze consisting of a tiling of smaller mazes'
)

parser.add_argument('--dataset-dir', type=str, default='datasets/mixed_style_100mazes_100goals_64res_13size_13seed')
parser.add_argument('--side-len', type=int, default=10, choices=[2, 3, 4, 5, 6, 7, 8, 9, 10])
parser.add_argument('--output-name', type=str, required=True)
# Parameters for modifying optimal path to not cut corners
parser.add_argument('--energy-sigma', type=float, default=0.25, help='std for the wall energy gaussian')
parser.add_argument('--energy-scale', type=float, default=0.75, help='scale for the wall energy gaussian')

args = parser.parse_args()

dataset_file = os.path.join(args.dataset_dir, 'maze_dataset.npz')
data = np.load(dataset_file)

n_mazes = args.side_len ** 2

# n_mazes by res by res
fine_mazes = data['fine_mazes'][:n_mazes, :, :]

# n_mazes by size by size
coarse_mazes = data['coarse_mazes'][:n_mazes, :, :]

# n_mazes by n_goals by res by res by 2
solved_mazes = data['solved_mazes'][:n_mazes, :, :, :, :]

# n_mazes by n_goals by 2
goals = data['goals'][:n_mazes, :, :]

n_goals = goals.shape[1]
res = fine_mazes.shape[1]

full_maze = np.zeros((args.side_len*res, args.side_len*res))

# connectivity of the tiles
global_connectivity = np.zeros((n_mazes, n_mazes))


# convert coord to index in connectivity graph
def ci(x, y):
    return x*args.side_len + y


for x in range(args.side_len):
    for y in range(args.side_len):
        full_maze[x*res:(x+1)*res, y*res:(y+1)*res] = fine_mazes[x*args.side_len + y, :, :]

# new goals corresponding to the square in front of a newly added doorway
# ground truth needs to be calculated for each of them
new_goals = []

# constants for getting the offsets for connecting mazes correct
to_edge = res - 3
to_center = int(res/2)

# create connections between mazes
# left-right connections
for i in range(args.side_len - 1):
    for j in range(args.side_len):
        x = i*res + to_edge
        y = j*res + to_center - 1
        # check if a connection will be valid
        # if np.all(full_maze[x:x+6, y] == np.array([0, 1, 1, 1, 1, 0])):
        if full_maze[x-6, y] == 0 and full_maze[x+6, y] == 0:
            # set pathway to open
            full_maze[x-6:x+7, y-3:y] = 0
            # full_maze[x - 6-res//2:x + 7+res//2, y-3:y] = .5  # for viewing/debugging
            # mark in connectivity graph
            global_connectivity[ci(i, j), ci(i + 1, j)] = 1
            global_connectivity[ci(i + 1, j), ci(i, j)] = 1

            # indices of maze with the door, location of door, indices for maze where the door leads
            new_goals.append([i, j, x-3, y-1, i + 1, j])
            new_goals.append([i + 1, j, x+3, y-1, i, j])

# up-down connections
for j in range(args.side_len - 1):
    for i in range(args.side_len):
        x = i*res + to_center
        y = j*res + to_edge
        # check if a connection will be valid
        # if np.all(full_maze[x, y:y+6] == np.array([0, 1, 1, 1, 1, 0])):
        if full_maze[x, y - 6] == 0 and full_maze[x, y + 6] == 0:
            # set pathway to open
            full_maze[x:x+3, y-6:y+7] = 0
            # full_maze[x:x + 3, y - 6 - res//2:y + 7 + res//2] = .5  # for viewing/debugging
            # mark in connectivity graph
            global_connectivity[ci(i, j + 1), ci(i, j)] = 1
            global_connectivity[ci(i, j), ci(i, j + 1)] = 1

            # indices of maze with the door, location of door, indices for maze where the door leads
            new_goals.append([i, j, x+1, y-3, i, j + 1])
            new_goals.append([i, j + 1, x+1, y+3, i, j])


n_new_goals = len(new_goals)

n_total_goals = n_goals*n_mazes + n_new_goals

# too much memory, changing approach
# full_solved_mazes = np.zeros((n_total_goals, res*args.side_len, res*args.side_len, 2))

new_solved_mazes = np.zeros((n_new_goals, res, res, 2))

# slice out the appropriate maze for each new goal
# compute an optimal policy from anywhere in the maze to that goal
# TODO: need to add wall avoidance energy to these

# for a local traversal from index1 to index2, which policy should be used
local_door_dict = {}

# used for the gaussian in wall avoidance
xs = np.linspace(0, coarse_mazes.shape[1], res)
ys = np.linspace(0, coarse_mazes.shape[1], res)

for gi in range(n_new_goals):
    print("Generating solution for new goal {} of {}".format(gi + 1, n_new_goals))
    i = new_goals[gi][0]
    j = new_goals[gi][1]
    global_x = new_goals[gi][2]
    global_y = new_goals[gi][3]
    local_x = global_x - i * res
    local_y = global_y - j * res
    goal_index = [local_x, local_y]
    fine_maze_slice = full_maze[i*res:(i+1)*res, j*res:(j+1)*res]
    # Compute the optimal path given this goal
    # Full solve is set to true, so start_indices is ignored
    new_solved_mazes[gi, :, :, :] = solve_maze(
        fine_maze_slice, start_indices=goal_index, goal_indices=goal_index, full_solve=True
    )

    ######################
    # Add wall avoidance #
    ######################

    # will contain a sum of gaussians placed at all wall locations
    wall_energy = np.zeros_like(fine_maze_slice)

    # will contain directions corresponding to the gradient of the wall energy
    # to be added to the solved maze to augment the directions chosen
    wall_gradient = np.zeros_like(new_solved_mazes[gi, :, :, :])

    meshgrid = np.meshgrid(xs, ys)

    # Compute wall energy
    for xi, x in enumerate(xs):
        for yi, y in enumerate(ys):
            if fine_maze_slice[xi, yi]:
                wall_energy[:, :] += args.energy_scale * gaussian_2d(x=x, y=y, meshgrid=meshgrid,
                                                                         sigma=args.energy_sigma)

    # Compute wall gradient using wall energy

    gradients = np.gradient(wall_energy[:, :])

    x_grad = gradients[0]
    y_grad = gradients[1]

    # Note all the flipping and transposing to get things right
    wall_gradient[:, :, 1] = -x_grad.T
    wall_gradient[:, :, 0] = -y_grad.T

    # modify the maze solution with the gradient
    new_solved_mazes[gi, :, :, :] = new_solved_mazes[gi, :, :, :] + wall_gradient

    # re-normalize the desired directions
    for xi in range(res):
        for yi in range(res):
            norm = np.linalg.norm(new_solved_mazes[gi, xi, yi, :])
            # If there is a wall, make the output be (0, 0)
            if fine_maze_slice[xi, yi]:
                new_solved_mazes[gi, xi, yi, 0] = 0
                new_solved_mazes[gi, xi, yi, 1] = 0
            elif norm != 0:
                new_solved_mazes[gi, xi, yi, :] /= np.linalg.norm(new_solved_mazes[gi, xi, yi, :])

    ###########################
    # Set the door dictionary #
    ###########################

    maze_id = i*args.side_len + j
    next_maze_id = new_goals[gi][4]*args.side_len + new_goals[gi][5]

    local_door_dict['{}_{}'.format(maze_id, next_maze_id)] = new_solved_mazes[gi, :, :, :]


# for every pair of start goal mazes in the higher level structure, compute which door to take

# for a global traversal from index1 to index2, which policy should be used
door_dict = {}
global_graph = Graph(matrix=global_connectivity)

for maze_start in range(n_mazes):
    for maze_end in range(n_mazes):
        if maze_start != maze_end:
            path = global_graph.search_graph(start_node=maze_start, end_node=maze_end)
            # index of the best maze to move to from the start
            best_maze = path[1]
            # # get the new_goal index that corresponds to the policy that will move to that doorway
            # '??'

            # set the global door dict to the correct local door dict
            door_dict['{}_{}'.format(maze_start, maze_end)] = local_door_dict['{}_{}'.format(maze_start, best_maze)]

# # now construct the gigantic optimal policy array
#
# # for each sub-maze, for each goal, if that goal is within the submaze, slice over the appropriate policy
# # if the goal is outside the submaze, slice in the door policy that matches with the door that should be taken
#
# for submaze_index in range(n_mazes):
#     for goal_index in range(n_total_goals):
#         full_solved_mazes[goal_index]
#         solved_mazes[submaze_index]



# import matplotlib.pyplot as plt
# plt.figure()
# plt.imshow(1-full_maze, cmap='gray')
# plt.figure()
# plt.imshow(global_connectivity)
# plt.show()


np.savez(
    args.output_name,
    full_maze=full_maze,
    goals=goals,
    fine_mazes=fine_mazes,
    coarse_mazes=data['coarse_mazes'],
    solved_mazes=solved_mazes,
    xs=data['xs'],
    ys=data['ys'],
    **door_dict
)
