import argparse
import numpy as np
import os
import nengo.spa as spa
from ssp_navigation.utils.path import generate_maze_sp, solve_maze
from ssp_navigation.utils.misc import gaussian_2d
from ssp_navigation.utils.agents import RandomTrajectoryAgent
from gridworlds.map_utils import generate_sensor_readings
from gridworlds.envs import GridWorldEnv, generate_obs_dict
from spatial_semantic_pointers.utils import make_good_unitary, encode_point


parser = argparse.ArgumentParser(
    'Generate random mazes, and their solutions for particular goal locations. Optionally include sensor data'
)

# General maze and SSP parameters
parser.add_argument('--maze-size', type=int, default=13, help='Size of the coarse maze structure')
parser.add_argument('--map-style', type=str, default='maze', choices=['blocks', 'maze', 'mixed', 'corridors'], help='Style of maze')
parser.add_argument('--res', type=int, default=64, help='resolution of the fine maze')
# parser.add_argument('--limit-low', type=float, default=0, help='lowest coordinate value')
# parser.add_argument('--limit-high', type=float, default=13, help='highest coordinate value')
parser.add_argument('--n-mazes', type=int, default=10, help='number of different maze configurations')
parser.add_argument('--n-goals', type=int, default=25, help='number of different goal locations for each maze')
parser.add_argument('--save-dir', type=str, default='datasets')
parser.add_argument('--seed', type=int, default=13)
parser.add_argument('--dim', type=int, default=512)

# Parameters for modifying optimal path to not cut corners
parser.add_argument('--energy-sigma', type=float, default=0.25, help='std for the wall energy gaussian')
parser.add_argument('--energy-scale', type=float, default=0.75, help='scale for the wall energy gaussian')

# Parameters for sensor measurements
parser.add_argument('--n-sensors', type=int, default=36, help='number of distance sensors around the agent')
parser.add_argument('--fov', type=float, default=360, help='field of view of distance sensors, in degrees')
parser.add_argument('--max-dist', type=float, default=10, help='maximum distance for distance sensor')

# Parameters for trajectory dataset
parser.add_argument('--n-trajectories', type=int, default=400, help='number of distinct full trajectories per map in the training set')
parser.add_argument('--trajectory-steps', type=int, default=250, help='number of steps in each trajectory')

# Options for disabling some dataset generation
parser.add_argument('--no-sensors', action='store_true', help='do not generate sensor dataset')
parser.add_argument('--no-trajectory', action='store_true', help='do not generate trajectory data')

args = parser.parse_args()

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

sensor_folder_name = '{}sensors_{}fov'.format(
    args.n_sensors,
    args.fov,
)

sensor_path = os.path.join(args.save_dir, base_folder_name, sensor_folder_name)

if not os.path.exists(sensor_path):
    os.makedirs(sensor_path)

sensor_name = os.path.join(sensor_path, 'maze_dataset.npz')

traj_folder_name = '{}t_{}s'.format(
    args.n_trajectories,
    args.trajectory_steps,
)

traj_path = os.path.join(args.save_dir, base_folder_name, sensor_folder_name, traj_folder_name)

if not os.path.exists(traj_path):
    os.makedirs(traj_path)

traj_name = os.path.join(traj_path, 'trajectory_dataset.npz')

limit_low = 0
limit_high = args.maze_size

if not os.path.exists(base_name):
    print("Generating base maze data")

    rng = np.random.RandomState(seed=args.seed)

    x_axis_sp = make_good_unitary(dim=args.dim, rng=rng)
    y_axis_sp = make_good_unitary(dim=args.dim, rng=rng)

    np.random.seed(args.seed)

    xs_coarse = np.linspace(limit_low, limit_high, args.maze_size)
    ys_coarse = np.linspace(limit_low, limit_high, args.maze_size)

    xs = np.linspace(limit_low, limit_high, args.res)
    ys = np.linspace(limit_low, limit_high, args.res)

    coarse_mazes = np.zeros((args.n_mazes, args.maze_size, args.maze_size))
    fine_mazes = np.zeros((args.n_mazes, args.res, args.res))
    solved_mazes = np.zeros((args.n_mazes, args.n_goals, args.res, args.res, 2))
    maze_sps = np.zeros((args.n_mazes, args.dim))

    # locs = np.zeros((args.n_mazes, args.n_goals, 2))
    goals = np.zeros((args.n_mazes, args.n_goals, 2))
    # loc_sps = np.zeros((args.n_mazes, args.n_goals, args.dim))
    goal_sps = np.zeros((args.n_mazes, args.n_goals, args.dim))

    for mi in range(args.n_mazes):
        # Generate a random maze
        if args.map_style == 'mixed':
            map_style = np.random.choice(['blocks', 'maze'])
        else:
            map_style = args.map_style

        maze_ssp, coarse_maze, fine_maze = generate_maze_sp(
            size=args.maze_size,
            xs=xs,
            ys=ys,
            x_axis_sp=x_axis_sp,
            y_axis_sp=y_axis_sp,
            normalize=True,
            obstacle_ratio=.2,
            map_style=map_style,
        )
        coarse_mazes[mi, :, :] = coarse_maze
        fine_mazes[mi, :, :] = fine_maze

        maze_sps[mi, :] = maze_ssp.v

        # Get a list of possible goal locations to choose (will correspond to all free spaces in the coarse maze)
        # free_spaces = np.argwhere(coarse_maze == 0)
        # Get a list of possible goal locations to choose (will correspond to all free spaces in the fine maze)
        free_spaces = np.argwhere(fine_maze == 0)
        print(free_spaces.shape)
        # downsample free spaces
        free_spaces = free_spaces[(free_spaces[:, 0] % 4 == 0) & (free_spaces[:, 1] % 4 == 0)]
        print(free_spaces.shape)

        n_free_spaces = free_spaces.shape[0]

        # no longer choosing indices based on any fine location, so that the likelihood of overlap between mazes increases
        # which should decrease the chances of overfitting based on goal location

        if args.n_goals > n_free_spaces:
            # workaround for there being more goals than spaces. This is non-ideal and shouldn't happen
            print("Warning, more goals desired than coarse free spaces in maze. n_free_spaces = {}".format(n_free_spaces))
            goal_indices = np.random.choice(n_free_spaces, args.n_goals, replace=True)
        else:
            goal_indices = np.random.choice(n_free_spaces, args.n_goals, replace=False)

        for n in range(args.n_goals):
            print("Generating Sample {} of {}".format(n+1, args.n_goals))
            # 2D coordinate of the goal
            goal_index = free_spaces[goal_indices[n], :]

            goal_x = xs[goal_index[0]]
            goal_y = ys[goal_index[1]]

            # make sure the goal is placed in an open space
            assert(fine_maze[goal_index[0], goal_index[1]] == 0)

            # Compute the optimal path given this goal
            # Full solve is set to true, so start_indices is ignored
            solved_maze = solve_maze(fine_maze, start_indices=goal_index, goal_indices=goal_index, full_solve=True)

            goals[mi, n, 0] = goal_x
            goals[mi, n, 1] = goal_y
            goal_sps[mi, n, :] = encode_point(goal_x, goal_y, x_axis_sp, y_axis_sp).v

            solved_mazes[mi, n, :, :, :] = solved_maze

    # Compute energy function to avoid walls

    # will contain a sum of gaussians placed at all wall locations
    wall_energy = np.zeros_like(fine_mazes)

    # will contain directions corresponding to the gradient of the wall energy
    # to be added to the solved maze to augment the directions chosen
    wall_gradient = np.zeros_like(solved_mazes)

    meshgrid = np.meshgrid(xs, ys)

    # Compute wall energy
    for ni in range(args.n_mazes):
        for xi, x in enumerate(xs):
            for yi, y in enumerate(ys):
                if fine_mazes[ni, xi, yi]:
                    wall_energy[ni, :, :] += args.energy_scale * gaussian_2d(x=x, y=y, meshgrid=meshgrid,
                                                                             sigma=args.energy_sigma)

    # Compute wall gradient using wall energy
    for ni in range(args.n_mazes):

        gradients = np.gradient(wall_energy[ni, :, :])

        x_grad = gradients[0]
        y_grad = gradients[1]

        for gi in range(args.n_goals):
            # Note all the flipping and transposing to get things right
            wall_gradient[ni, gi, :, :, 1] = -x_grad.T
            wall_gradient[ni, gi, :, :, 0] = -y_grad.T

    # modify the maze solution with the gradient
    solved_mazes = solved_mazes + wall_gradient

    # re-normalize the desired directions
    for ni in range(args.n_mazes):
        for gi in range(args.n_goals):
            for xi in range(args.res):
                for yi in range(args.res):
                    norm = np.linalg.norm(solved_mazes[ni, gi, xi, yi, :])
                    # If there is a wall, make the output be (0, 0)
                    if fine_mazes[ni, xi, yi]:
                        solved_mazes[ni, gi, xi, yi, 0] = 0
                        solved_mazes[ni, gi, xi, yi, 1] = 0
                    elif norm != 0:
                        solved_mazes[ni, gi, xi, yi, :] /= np.linalg.norm(solved_mazes[ni, gi, xi, yi, :])

    x_axis_vec = x_axis_sp.v
    y_axis_vec = y_axis_sp.v

    np.savez(
        base_name,
        coarse_mazes=coarse_mazes,
        fine_mazes=fine_mazes,
        solved_mazes=solved_mazes,
        maze_sps=maze_sps,
        x_axis_sp=x_axis_vec,
        y_axis_sp=y_axis_vec,
        goal_sps=goal_sps,
        goals=goals,
        xs=xs,
        ys=ys,
    )

else:
    print("Loading existing base maze data")
    # base data already exists, load it instead of generating
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

    x_axis_sp = spa.SemanticPointer(data=x_axis_vec)
    y_axis_sp = spa.SemanticPointer(data=y_axis_vec)

if (not args.no_sensors):
    if not os.path.exists(sensor_name):
        print("Generating Sensor Data")

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

        # Generate sensor readings for every location in xs and ys in each maze
        for mi in range(n_mazes):
            # Print that replaces the line each time
            print('\x1b[2K\r Map {0} of {1}'.format(mi+1, n_mazes), end="\r")
            # NOTE: only need to create the env if other 'sensors' such as boundary cells are used
            # # Generate a GridWorld environment corresponding to the current maze in order to calculate sensor measurements
            # env = GridWorldEnv(
            #     map_array=coarse_mazes[mi, :, :],
            #     movement_type='holonomic',
            #     continuous=True,
            # )
            #
            # sensors = env.get_dist_sensor_readings(
            #     state=state,
            #     n_sensors=n_sensors,
            #     fov_rad=fov_rad,
            #     max_dist=args.max_dist,
            #     normalize=False
            # )

            for xi, x in enumerate(xs_scaled):
                for yi, y in enumerate(ys_scaled):

                    # Only compute measurements if not in a wall
                    if fine_mazes[mi, xi, yi] == 0:
                        # Compute sensor measurements and scale them based on xs and ys
                        dist_sensors[mi, xi, yi, :] = generate_sensor_readings(
                            map_arr=coarse_mazes[mi, :, :],
                            n_sensors=n_sensors,
                            fov_rad=fov_rad,
                            x=x,
                            y=y,
                            th=0,
                            max_sensor_dist=args.max_dist,
                            debug_value=0,
                        ) * sensor_scaling

        print("Sensor data generated for {0} mazes".format(n_mazes))

        np.savez(
            sensor_name,
            dist_sensors=dist_sensors,
            coarse_mazes=coarse_mazes,
            fine_mazes=fine_mazes,
            solved_mazes=solved_mazes,
            maze_sps=maze_sps,
            x_axis_sp=x_axis_vec,
            y_axis_sp=y_axis_vec,
            goal_sps=goal_sps,
            goals=goals,
            xs=xs,
            ys=ys,
            max_dist=args.max_dist,
            fov=args.fov,
        )

    else:
        print("Sensor data already exists for these parameters")

        # sensor data already exists, load it instead of generating
        sensor_data = np.load(base_name)

        xs = sensor_data['xs']
        ys = sensor_data['ys']
        x_axis_vec = sensor_data['x_axis_sp']
        y_axis_vec = sensor_data['y_axis_sp']
        coarse_mazes = sensor_data['coarse_mazes']
        fine_mazes = sensor_data['fine_mazes']
        solved_mazes = sensor_data['solved_mazes']
        maze_sps = sensor_data['maze_sps']
        goal_sps = sensor_data['goal_sps']
        goals = sensor_data['goals']


if (not args.no_sensors) or (not args.no_trajectory):
    if not os.path.exists(traj_name):
        print("Generating Trajectory Data")

        # Gridworld environment parametesr
        params = {
            'continuous': True,
            'fov': args.fov,
            'n_sensors': args.n_sensors,
            'max_sensor_dist': args.max_dist,
            'normalize_dist_sensors': False,
            'movement_type': 'holonomic',
            'seed': args.seed,
            'map_style': args.map_style,
            'map_size': args.maze_size,
            'fixed_episode_length': False,  # Setting to false so location resets are not automatic
            'episode_length': args.trajectory_steps,
            'max_lin_vel': 5,
            'max_ang_vel': 5,
            'dt': 0.1,
            'full_map_obs': False,
            'pob': 0,
            'n_grid_cells': 0,
            'heading': 'none',
            'location': 'none',
            'goal_loc': 'none',
            'goal_vec': 'none',
            'bc_n_ring': 0,
            'hd_n_cells': 0,
            'goal_csp': False,
            'goal_distance': 0,  # 0 means completely random, goal isn't used anyway
            'agent_csp': True,
            'csp_dim': args.dim,
            'x_axis_vec': x_axis_sp,
            'y_axis_vec': y_axis_sp,
        }

        obs_dict = generate_obs_dict(params)

        positions = np.zeros((args.n_mazes, args.n_trajectories, args.trajectory_steps, 2))
        dist_sensors = np.zeros((args.n_mazes, args.n_trajectories, args.trajectory_steps, args.n_sensors))

        # spatial semantic pointers for position
        ssps = np.zeros((args.n_mazes, args.n_trajectories, args.trajectory_steps, args.dim))
        # velocity in x and y
        cartesian_vels = np.zeros((args.n_mazes, args.n_trajectories, args.trajectory_steps, 2))

        def get_ssp_activation(pos):

            return encode_point(pos[0] * args.ssp_scaling, pos[1] * args.ssp_scaling, x_axis_sp, y_axis_sp).v

        for mi in range(args.n_mazes):
            # # Print that replaces the line each time
            # print('\x1b[2K\r Map {} of {}'.format(mi + 1, args.n_mazes), end="\r")
            print("Map {} of {}".format(mi + 1, args.n_mazes))

            env = GridWorldEnv(
                map_array=coarse_mazes[mi, :, :],
                observations=obs_dict,
                movement_type=params['movement_type'],
                max_lin_vel=params['max_lin_vel'],
                max_ang_vel=params['max_ang_vel'],
                continuous=params['continuous'],
                max_steps=params['episode_length'],
                fixed_episode_length=params['fixed_episode_length'],
                dt=params['dt'],
                screen_width=300,
                screen_height=300,
                # TODO: use these parameters appropriately, and save them with the dataset
                # csp_scaling=ssp_scaling,  # multiply state by this value before creating a csp
                # csp_offset=ssp_offset,  # subtract this value from state before creating a csp
            )

            agent = RandomTrajectoryAgent(obs_index_dict=env.obs_index_dict, n_sensors=args.n_sensors)

            obs_index_dict = env.obs_index_dict

            for n in range(args.n_trajectories):
                # Print that replaces the line each time
                print('\x1b[2K\r Generating Trajectory {} of {}'.format(n + 1, args.n_trajectories), end="\r")
                # print("Generating Trajectory {} of {}".format(n + 1, args.n_trajectories))

                obs = env.reset(goal_distance=params['goal_distance'])

                dist_sensors[mi, n, 0, :] = obs[env.obs_index_dict['dist_sensors']]
                ssps[mi, n, 0, :] = obs[env.obs_index_dict['agent_csp']]

                # TODO: should this be converted from env coordinates to SSP coordinates?
                positions[mi, n, 0, 0] = env.state[0]
                positions[mi, n, 0, 1] = env.state[1]

                for s in range(1, args.trajectory_steps):
                    action = agent.act(obs)

                    cartesian_vels[mi, n, s - 1, 0] = action[0]
                    cartesian_vels[mi, n, s - 1, 1] = action[1]

                    obs, reward, done, info = env.step(action)

                    dist_sensors[mi, n, s, :] = obs[env.obs_index_dict['dist_sensors']]
                    ssps[mi, n, s, :] = obs[env.obs_index_dict['agent_csp']]  # TODO: make sure this observation is correct

                    positions[mi, n, s, 0] = env.state[0]
                    positions[mi, n, s, 1] = env.state[1]

                # Purely for consistency, this action is not used
                action = agent.act(obs)

                cartesian_vels[mi, n, -1, 0] = action[0]
                cartesian_vels[mi, n, -1, 1] = action[1]

        np.savez(
            traj_name,
            positions=positions,
            dist_sensors=dist_sensors,
            ssps=ssps,
            cartesian_vels=cartesian_vels,
            x_axis_vec=x_axis_vec,
            y_axis_vec=y_axis_vec,
            coarse_mazes=coarse_mazes,
            # ssp_scaling=args.ssp_scaling,
            # ssp_offset=args.ssp_offset,
            # ssp_scaling=ssp_scaling,
            # ssp_offset=ssp_offset,
            # NOTE: this code assumes a 1 to 1 mapping of env coordinates to SSP coordinates
            ssp_scaling=1,
            ssp_offset=0,
        )

    else:
        print("Trajectory data already exists for these parameters")
