import argparse
import numpy as np
import os
from ssp_navigation.utils.misc import gaussian_2d
# from gridworlds.map_utils import generate_sensor_readings


parser = argparse.ArgumentParser(
    'Generate random mazes, and their solutions for particular goal locations. Optionally include sensor data'
)

# General maze and SSP parameters
parser.add_argument('--maze-size', type=int, default=13, help='Size of the coarse maze structure')
parser.add_argument('--map-style', type=str, default='mixed', choices=['blocks', 'maze', 'mixed', 'corridors'], help='Style of maze')
parser.add_argument('--res', type=int, default=64, help='resolution of the fine maze')
# parser.add_argument('--limit-low', type=float, default=0, help='lowest coordinate value')
# parser.add_argument('--limit-high', type=float, default=13, help='highest coordinate value')
parser.add_argument('--n-mazes', type=int, default=100, help='number of different maze configurations')
parser.add_argument('--n-goals', type=int, default=100, help='number of different goal locations for each maze')
parser.add_argument('--save-dir', type=str, default='datasets')
parser.add_argument('--seed', type=int, default=13)

# Parameters for sensor measurements
parser.add_argument('--n-sensors', type=int, default=36, help='number of distance sensors around the agent')
parser.add_argument('--fov', type=float, default=360, help='field of view of distance sensors, in degrees')
parser.add_argument('--max-dist', type=float, default=10, help='maximum distance for distance sensor')

parser.add_argument('--n-sensor-mazes', type=int, default=10)
parser.add_argument('--samples-per-maze', type=int, default=100000, help='samples per maze, if zero, use a fine grid')

parser.add_argument('--colour-func', type=str, default='many', choices=['basic', 'many'])

args = parser.parse_args()


def generate_colour_sensor_readings(map_arr,
                                    colour_func,
                                    n_sensors=30,
                                    fov_rad=np.pi,
                                    x=0,
                                    y=0,
                                    th=0,
                                    max_sensor_dist=10,
                                    ):
    """
    Given a map, agent location in the map, number of sensors, field of view
    calculate the distance readings of each sensor to the nearest obstacle
    uses supersampling to find the approximate collision points
    """
    dists = np.zeros((n_sensors, 4))

    angs = np.linspace(-fov_rad / 2. + th, fov_rad / 2. + th, n_sensors)

    for i, ang in enumerate(angs):
        dists[i, :] = get_collision_coord(map_arr, x, y, ang, colour_func, max_sensor_dist)

    return dists



def get_collision_coord(map_array, x, y, th, colour_func,
                        max_sensor_dist=10,
                        dr=.05,
                        ):
    """
    Find the first occupied space given a start point and direction
    """
    # Define the step sizes
    dx = np.cos(th)*dr
    dy = np.sin(th)*dr

    # Initialize to starting point
    cx = x
    cy = y

    # R, G, B, dist
    ret = np.zeros((4, ))

    for i in range(int(max_sensor_dist/dr)):
        # Move one unit in the direction of the sensor
        cx += dx
        cy += dy

        if map_array[int(round(cx)), int(round(cy))] == 1:
            ret[:3] = colour_func(cx, cy)
            ret[3] = (i - 1)*dr
            return ret

    return max_sensor_dist


def free_space(pos, map_array, width, height):
    """
    Returns True if the position corresponds to a free space in the map
    :param pos: 2D floating point x-y coordinates
    """
    # TODO: doublecheck that rounding is the correct thing to do here
    x = np.clip(int(np.round(pos[0])), 0, width - 1)
    y = np.clip(int(np.round(pos[1])), 0, height - 1)
    return map_array[x, y] == 0


if args.colour_func == 'basic':
    colour_centers = np.array([
        # [3, 3],
        # [10, 4],
        # [2, 8],
        [3, 3],
        [10, 4],
        [7, 7],
    ])

    def colour_func(x, y, sigma=7):
        ret = np.zeros((3, ))

        for c in range(3):
            ret[c] = np.exp(-((colour_centers[c, 0] - x) ** 2 + (colour_centers[c, 1] - y) ** 2) / (sigma**2))

        return ret
elif args.colour_func == 'many':
    rng_colour = np.random.RandomState(seed=13)
    # colour_centers = rng_colour.uniform(low=0, high=13, size=(12, 2))
    colour_centers = rng_colour.uniform(low=0, high=13, size=(18, 2))

    def colour_func(x, y, sigma=3.5, height=.33):
        ret = np.zeros((3,))

        for c, colour_center in enumerate(colour_centers):
            ret[c % 3] += height * np.exp(-((colour_center[0] - x) ** 2 + (colour_center[1] - y) ** 2) / (sigma ** 2))

        ret = np.clip(ret, 0, 1)

        return ret

# # testing the colour locations
# import matplotlib.pyplot as plt
#
# xs = np.linspace(0, 13, 128)
#
# img = np.zeros((128, 128, 3))
#
# for i, x in enumerate(xs):
#     for j, y in enumerate(xs):
#         img[i, j, :] = colour_func(x, y)
#
# plt.imshow(img)
#
#
# plt.show()
# assert False

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

base_folder_name = '{}_style_{}mazes_{}goals_{}res_{}size_{}seed'.format(
    args.map_style, args.n_mazes, args.n_goals, args.res, args.maze_size, args.seed
)
# base_folder_name = '{}_style_{}mazes_{}goals_{}res_{}size_{}_to_{}_{}seed'.format(
#     args.map_style, args.n_mazes, args.n_goals, args.res, args.maze_size, args.limit_low, args.limit_high, args.seed
# )

base_path = os.path.join(args.save_dir, base_folder_name)

if not os.path.exists(base_path):
    os.makedirs(base_path)

base_name = os.path.join(base_path, 'maze_dataset.npz')

sensor_folder_name = 'coloured_{}mazes_{}sensors_{}fov_{}samples_colour_{}'.format(
    args.n_sensor_mazes,
    args.n_sensors,
    args.fov,
    args.samples_per_maze,
    args.colour_func,
)

sensor_path = os.path.join(args.save_dir, base_folder_name, sensor_folder_name)

if not os.path.exists(sensor_path):
    os.makedirs(sensor_path)

if args.samples_per_maze == 0:
    sensor_name = os.path.join(sensor_path, 'coloured_sensor_dataset_test.npz')
else:
    sensor_name = os.path.join(sensor_path, 'coloured_sensor_dataset.npz')

base_data = np.load(base_name)

xs = base_data['xs']
ys = base_data['ys']
x_axis_vec = base_data['x_axis_sp']
y_axis_vec = base_data['y_axis_sp']
coarse_mazes = base_data['coarse_mazes']
fine_mazes = base_data['fine_mazes']
solved_mazes = base_data['solved_mazes']
maze_sps = base_data['maze_sps']
goal_sps = base_data['goal_sps']
goals = base_data['goals']

n_mazes = solved_mazes.shape[0]
n_goals = solved_mazes.shape[1]
res = solved_mazes.shape[2]
coarse_maze_size = coarse_mazes.shape[1]

limit_low = xs[0]
limit_high = xs[-1]

sensor_scaling = (limit_high - limit_low) / coarse_maze_size

# Scale to the coordinates of the coarse maze, for getting distance measurements
xs_scaled = ((xs - limit_low) / (limit_high - limit_low)) * coarse_maze_size
ys_scaled = ((ys - limit_low) / (limit_high - limit_low)) * coarse_maze_size

n_sensors = args.n_sensors
fov_rad = args.fov * np.pi / 180

dist_sensors = np.zeros((n_mazes, res, res, n_sensors))

if not os.path.exists(sensor_name):
    print("Generating Sensor Data")

    n_sensors = args.n_sensors
    fov_rad = args.fov * np.pi / 180

    if args.samples_per_maze == 0:

        # R, G, B, distance
        dist_sensors = np.zeros((args.n_sensor_mazes, res, res, n_sensors, 4))

        # Generate sensor readings for every location in xs and ys in each maze
        for mi in range(args.n_sensor_mazes):
            # Print that replaces the line each time
            print('\x1b[2K\r Map {0} of {1}'.format(mi + 1, args.n_sensor_mazes), end="\r")

            for xi, x in enumerate(xs_scaled):
                for yi, y in enumerate(ys_scaled):

                    # Only compute measurements if not in a wall
                    if fine_mazes[mi, xi, yi] == 0:
                        # Compute sensor measurements and scale them based on xs and ys
                        dist_sensors[mi, xi, yi, :, :] = generate_colour_sensor_readings(
                            map_arr=coarse_mazes[mi, :, :],
                            colour_func=colour_func,
                            n_sensors=n_sensors,
                            fov_rad=fov_rad,
                            x=x,
                            y=y,
                            th=0,
                            max_sensor_dist=args.max_dist,
                        ) * sensor_scaling
        np.savez(
            sensor_name,
            dist_sensors=dist_sensors,
            coarse_mazes=coarse_mazes,
            fine_mazes=fine_mazes,
            xs=xs,
            ys=ys,
            max_dist=args.max_dist,
            fov=args.fov,
        )
    else:

        dist_sensors = np.zeros((args.n_sensor_mazes, args.samples_per_maze, args.n_sensors, 4))
        locations = np.zeros((args.n_sensor_mazes, args.samples_per_maze, 2))

        rng = np.random.RandomState(seed=args.seed)

        for mi in range(args.n_sensor_mazes):
            # Print that replaces the line each time
            print('\x1b[2K\r Map {0} of {1}'.format(mi + 1, args.n_sensor_mazes), end="\r")

            for i in range(args.samples_per_maze):

                while True:
                    xy = rng.uniform(xs_scaled[0], xs_scaled[-1], size=(2,))
                    if free_space(xy, coarse_mazes[mi, :, :], coarse_maze_size, coarse_maze_size):
                        break

                locations[mi, i, :] = xy

                # Compute sensor measurements and scale them based on xs and ys
                dist_sensors[mi, i, :, :] = generate_colour_sensor_readings(
                    map_arr=coarse_mazes[mi, :, :],
                    colour_func=colour_func,
                    n_sensors=n_sensors,
                    fov_rad=fov_rad,
                    x=xy[0],
                    y=xy[1],
                    th=0,
                    max_sensor_dist=args.max_dist,
                ) * sensor_scaling

        np.savez(
            sensor_name,
            dist_sensors=dist_sensors,
            locations=locations,
            coarse_mazes=coarse_mazes,
            fine_mazes=fine_mazes,
            xs=xs,
            ys=ys,
            max_dist=args.max_dist,
            fov=args.fov,
        )

    print("Sensor data generated for {0} mazes".format(args.n_sensor_mazes))



else:
    print("Sensor data already exists for these parameters")
