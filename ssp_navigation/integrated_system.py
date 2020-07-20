import numpy as np
import argparse
import torch
import os
from gridworlds.envs import GridWorldEnv, generate_obs_dict
from gridworlds.constants import possible_objects

from ssp_navigation.utils.agents import GoalFindingAgent, IntegSystemAgent
import nengo.spa as spa
import nengo_spa as nengo_spa
from collections import OrderedDict
from spatial_semantic_pointers.utils import encode_point, ssp_to_loc, get_heatmap_vectors

from ssp_navigation.utils.models import FeedForward, MLP, SpikingPolicy, SpikingLocalization, \
    SpikingPolicyNengo, SpikingLocalizationNengo
from ssp_navigation.utils.training import LocalizationModel
from ssp_navigation.utils.path import solve_maze
from ssp_navigation.utils.encodings import get_encoding_function, add_encoding_params

"""
- semantic goal to goal ssp through deconvolution of known memory and cleanup
- sensory info and velocity commands to agent ssp localization network and past estimates
- ssp goal and agent ssp to velocity command through policy function
"""

parser = argparse.ArgumentParser('Run a full system on a maze task')

parser.add_argument('--spiking', action='store_true', help='Use spiking components')

parser.add_argument('--cleanup-network', type=str,
                    default='networks/cleanup_network.pt',
                    help='SSP cleanup network')
parser.add_argument('--localization-network', type=str,
                    default='networks/localization_network_colour_10maze_512dim_2048hs_st.pt',
                    help='localization from distance sensors and context')
parser.add_argument('--policy-network', type=str,
                    default='networks/policy_network_10maze_512dim_2048hs_st.pt',
                    help='goal navigation network')

parser.add_argument('--dataset', type=str,
                    default='datasets/mixed_style_100mazes_100goals_64res_13size_13seed/maze_dataset.npz',
                    help='dataset to get the maze layout from')
parser.add_argument('--maze-index', type=int, default=0, help='index within the dataset for the maze to use')
parser.add_argument('--noise', type=float, default=0.25, help='magnitude of gaussian noise to add to the actions')
parser.add_argument('--use-dataset-goals', action='store_true', help='use only the goals the policy was trained on')
# parser.add_argument('--ghost', type=str, default='trajectory_loc', choices=['none', 'snapshot_loc', 'trajectory_loc'],
#                     help='what information should be displayed by the ghost')
parser.add_argument('--ghost', action='store_true', help='show the localization estimate')
parser.add_argument('--use-cleanup-gt', action='store_true')
# parser.add_argument('--use-policy-gt', action='store_true')
parser.add_argument('--use-localization-gt', action='store_true')

parser.add_argument('--maze-id-dim', default=256, help='dimensionality of the maze ID')
parser.add_argument('--n-goals', default=10)
parser.add_argument('--normalize-action', action='store_true', help='normalize action to be on the unit circle')
# parser.add_argument('--new-sensor-ratio', default=0.1, help='weighting on new sensor information')
parser.add_argument('--new-sensor-ratio', default=0.5, help='weighting on new sensor information')

# network parameters
parser.add_argument('--cleanup-hs', type=int, default=512, help='hidden size for cleanup')
parser.add_argument('--localization-hs', type=int, default=2048, help='hidden size for localization')
parser.add_argument('--n-hidden-layers-localization', type=int, default=1, help='number of hidden layers in the localization network')
parser.add_argument('--localization-dropout-fraction', type=float, default=0.1)
parser.add_argument('--policy-hs', type=int, default=2048, help='hidden size for policy')
parser.add_argument('--n-hidden-layers-policy', type=int, default=1, help='number of hidden layers in the policy network')
parser.add_argument('--policy-dropout-fraction', type=float, default=0.15)

parser.add_argument('--evaluate', action='store_true', help='if set, evaluate system on a variety of mazes')
parser.add_argument('--record-trajectories', action='store_true', help='if set, record agent position at every step')
parser.add_argument('--env-seed', type=int, default=13)

# encoding parameters with correct defaults for this task
parser.add_argument('--spatial-encoding', type=str, default='sub-toroid-ssp',
                    choices=[
                        'ssp', 'hex-ssp', 'periodic-hex-ssp', 'grid-ssp', 'ind-ssp', 'orth-proj-ssp',
                        'sub-toroid-ssp', 'proj-ssp', 'var-sub-toroid-ssp',
                        'random', '2d', '2d-normalized', 'one-hot', 'hex-trig',
                        'trig', 'random-trig', 'random-rotated-trig', 'random-proj', 'legendre',
                        'learned', 'learned-normalized', 'frozen-learned', 'frozen-learned-normalized',
                        'pc-gauss', 'pc-dog', 'tile-coding'
                    ],
                    help='coordinate encoding for agent location and goal')
parser.add_argument('--freq-limit', type=float, default=10,
                    help='highest frequency of sine wave for random-trig encodings')
parser.add_argument('--hex-freq-coef', type=float, default=2.5,
                    help='constant to scale frequencies by for hex-trig')
parser.add_argument('--pc-gauss-sigma', type=float, default=0.75, help='sigma for the gaussians')
parser.add_argument('--pc-diff-sigma', type=float, default=0.5, help='sigma for subtracted gaussian in DoG')
parser.add_argument('--hilbert-points', type=int, default=1, choices=[0, 1, 2, 3],
                    help='pc centers. 0: random uniform. 1: hilbert curve. 2: evenly spaced grid. 3: hex grid')
parser.add_argument('--n-tiles', type=int, default=8, help='number of layers for tile coding')
parser.add_argument('--n-bins', type=int, default=0, help='number of bins for tile coding')
parser.add_argument('--ssp-scaling', type=float, default=0.5)
parser.add_argument('--grid-ssp-min', type=float, default=0.25, help='minimum plane wave scale')
parser.add_argument('--grid-ssp-max', type=float, default=2.0, help='maximum plane wave scale')
parser.add_argument('--phi', type=float, default=0.5, help='phi as a fraction of pi for orth-proj-ssp')
parser.add_argument('--n-proj', type=int, default=3, help='projection dimension for sub toroids')
parser.add_argument('--scale-ratio', type=float, default=0, help='ratio between sub toroid scales')

parser.add_argument('--dim', type=int, default=512, help='Dimensionality of the semantic pointers')
parser.add_argument('--limit', type=float, default=5, help='The limits of the space')
parser.add_argument('--seed', type=int, default=13)

# parser = add_encoding_params(parser)

args = parser.parse_args()

data = np.load(args.dataset)

rng = np.random.RandomState(seed=args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# n_mazes by res by res
fine_mazes = data['fine_mazes']

# # n_mazes by res by res by 2
# solved_mazes = data['solved_mazes']

# # n_mazes by dim
# maze_sps = data['maze_sps']

# n_mazes by n_goals by 2
goals = data['goals']

# n_goals = goals.shape[1]
n_mazes = fine_mazes.shape[0]

id_size = args.maze_id_dim

maze_sps = np.zeros((n_mazes, args.maze_id_dim))
# overwrite data
for mi in range(n_mazes):
    maze_sps[mi, :] = spa.SemanticPointer(args.maze_id_dim).v

# the unsqueeze is to add the batch dimension
map_id = torch.Tensor(maze_sps[args.maze_index, :]).unsqueeze(0)

n_sensors = 36

colour_centers = np.array([
    [3, 3],
    [10, 4],
    [7, 7],
])


def colour_func(x, y, sigma=7):
    ret = np.zeros((3, ))

    for c in range(3):
        ret[c] = np.exp(-((colour_centers[c, 0] - x) ** 2 + (colour_centers[c, 1] - y) ** 2) / (sigma**2))

    return ret

params = {
    'continuous': True,
    'fov': 360,
    'n_sensors': n_sensors,
    'colour_func': colour_func,
    'max_sensor_dist': 10,
    'normalize_dist_sensors': False,
    'movement_type': 'holonomic',
    'seed': args.seed,
    # 'map_style': args.map_style,
    'map_size': 10,
    'fixed_episode_length': False,  # Setting to false so location resets are not automatic
    'episode_length': 1000,#500, 1000,  #200,
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
    'csp_dim': 0,
    'goal_csp': False,
    'agent_csp': False,

    # set up rewards so minimum score is -1 and maximum score is +1 (based on 1000 steps max)
    'wall_penalty': -.001,
    'movement_cost': -.001,
    'goal_reward': 1.,

    'goal_distance': 0,#args.goal_distance  # 0 means completely random
}

obs_dict = generate_obs_dict(params)

np.random.seed(params['seed'])

# data = np.load(args.dataset)

# n_mazes by size by size
coarse_mazes = data['coarse_mazes']
coarse_size = coarse_mazes.shape[1]
# n_maps = coarse_mazes.shape[0]

# n_mazes by res by res
fine_mazes = data['fine_mazes']
xs = data['xs']
ys = data['ys']
res = fine_mazes.shape[1]

coarse_xs = np.linspace(xs[0], xs[-1], coarse_size)
coarse_ys = np.linspace(ys[0], ys[-1], coarse_size)

map_array = coarse_mazes[args.maze_index, :, :]
limit_low = 0
limit_high = 13
encoding_func, repr_dim = get_encoding_function(args, limit_low=limit_low, limit_high=limit_high)

# x_axis_sp = spa.SemanticPointer(data=data['x_axis_sp'])
# y_axis_sp = spa.SemanticPointer(data=data['y_axis_sp'])
x_axis_vec = encoding_func(1, 0)
y_axis_vec = encoding_func(0, 1)
x_axis_sp = nengo_spa.SemanticPointer(data=x_axis_vec)
y_axis_sp = nengo_spa.SemanticPointer(data=y_axis_vec)
heatmap_vectors = get_heatmap_vectors(xs, ys, x_axis_sp, y_axis_sp)
coarse_heatmap_vectors = get_heatmap_vectors(coarse_xs, coarse_ys, x_axis_sp, y_axis_sp)

# fixed random set of locations for the goals
limit_range = xs[-1] - xs[0]

goal_sps = data['goal_sps']
goals = data['goals']
# print(np.min(goals))
# print(np.max(goals))
goals_scaled = ((goals - xs[0]) / limit_range) * coarse_size
# print(np.min(goals_scaled))
# print(np.max(goals_scaled))

n_goals = args.n_goals
object_locations = OrderedDict()
vocab = {}
for i in range(n_goals):
    sp_name = possible_objects[i]
    if args.use_dataset_goals:
        object_locations[sp_name] = goals_scaled[args.maze_index, i]  # using goal locations from the dataset
    else:
        # If set to None, the environment will choose a random free space on init
        object_locations[sp_name] = None
    # vocab[sp_name] = spa.SemanticPointer(ssp_dim)
    vocab[sp_name] = nengo_spa.SemanticPointer(data=np.random.uniform(-1, 1, size=args.dim)).normalized()

env = GridWorldEnv(
    map_array=map_array,
    object_locations=object_locations,  # object locations explicitly chosen so a fixed SSP memory can be given
    observations=obs_dict,
    movement_type=params['movement_type'],
    max_lin_vel=params['max_lin_vel'],
    max_ang_vel=params['max_ang_vel'],
    continuous=params['continuous'],
    max_steps=params['episode_length'],
    fixed_episode_length=params['fixed_episode_length'],
    wall_penalty=params['wall_penalty'],
    movement_cost=params['movement_cost'],
    goal_reward=params['goal_reward'],
    dt=params['dt'],
    screen_width=300,
    screen_height=300,
    debug_ghost=True,
    seed=args.env_seed,
)

# Fill the item memory with the correct SSP for remembering the goal locations
item_memory = nengo_spa.SemanticPointer(data=np.zeros((args.dim,)))

for i in range(n_goals):
    sp_name = possible_objects[i]
    x_env, y_env = env.object_locations[sp_name][[0, 1]]

    # Need to scale to SSP coordinates
    # Env is 0 to 13, SSP is -5 to 5
    x = ((x_env - 0) / coarse_size) * limit_range + xs[0]
    y = ((y_env - 0) / coarse_size) * limit_range + ys[0]

    item_memory += vocab[sp_name] * encode_point(x, y, x_axis_sp, y_axis_sp)
# item_memory.normalize()
item_memory = item_memory.normalized()

# Component functions of the full system

cleanup_network = FeedForward(input_size=args.dim, hidden_size=args.cleanup_hs, output_size=args.dim)
if not args.use_cleanup_gt:
    cleanup_network.load_state_dict(
        torch.load(args.cleanup_network, map_location=lambda storage, loc: storage),
        strict=True
    )
    cleanup_network.eval()

if args.spiking:
    # policy_network = SpikingPolicy(
    #     param_file='networks/policy_params_1000000samples_250epochs',
    #     dim=256,
    #     maze_id_dim=256,
    #     hidden_size=1024,
    #     net_seed=13,
    #     n_steps=30,
    # )
    policy_network = SpikingPolicyNengo(
        param_file='networks/nengo_policy_obj_2layer_mse_hs2048_5000000samples_100epochs_1e-05reg.pkl',
        dim=256,
        maze_id_dim=256,
        hidden_size=2048,
        n_layers=2,
        net_seed=13,
        n_steps=30,
    )
else:
    policy_network = MLP(
        input_size=id_size + repr_dim * 2,
        hidden_size=args.policy_hs,
        output_size=2,
        n_layers=args.n_hidden_layers_policy,
        dropout_fraction=args.policy_dropout_fraction,
    )

    policy_network.load_state_dict(
        torch.load(args.policy_network, map_location=lambda storage, loc: storage),
        strict=True
    )
    policy_network.eval()

if args.spiking:
    # localization_network = SpikingLocalization(
    #     param_file='networks/localization_params_mse_hs1024_100000samples_25epochs',
    #     dim=256,
    #     maze_id_dim=256,
    #     n_sensors=36,
    #     hidden_size=1024,
    #     net_seed=13,
    #     n_steps=30,
    # )
    localization_network = SpikingLocalizationNengo(
        param_file='networks/nengo_localization_obj_1layer_mse_hs4096_3000000samples_25epochs_1e-05reg.pkl',
        dim=256,
        maze_id_dim=256,
        n_sensors=36,
        hidden_size=4096,
        n_layers=1,
        net_seed=13,
        n_steps=30,
    )
else:
    # localization_network = FeedForward(
    #     input_size=n_sensors + args.maze_id_dim,
    #     hidden_size=args.localization_hs,
    #     output_size=args.dim,
    # )
    localization_network = MLP(
        input_size=n_sensors * 4 + args.maze_id_dim,
        hidden_size=args.localization_hs,
        output_size=args.dim,
        n_layers=args.n_hidden_layers_localization,
        dropout_fraction=args.localization_dropout_fraction,
    )
    if not args.use_localization_gt:
        localization_network.load_state_dict(
            torch.load(args.localization_network, map_location=lambda storage, loc: storage),
            strict=True
        )
        localization_network.eval()


# # Ground truth versions of the above functions
# planner = OptimalPlanner(continuous=True, directional=False)


def cleanup_gt(env):
    x = ((env.goal_state[0] - 0) / coarse_size) * limit_range + xs[0]
    y = ((env.goal_state[1] - 0) / coarse_size) * limit_range + ys[0]
    goal_ssp = encode_point(x, y, x_axis_sp, y_axis_sp)
    return torch.Tensor(goal_ssp.v).unsqueeze(0)


def localization_gt(env):
    x = ((env.state[0] - 0) / coarse_size) * limit_range + xs[0]
    y = ((env.state[1] - 0) / coarse_size) * limit_range + ys[0]
    agent_ssp = encode_point(x, y, x_axis_sp, y_axis_sp)
    return torch.Tensor(agent_ssp.v).unsqueeze(0)


agent = IntegSystemAgent(
    cleanup_network=cleanup_network,
    localization_network=localization_network,
    policy_network=policy_network,
    new_sensor_ratio=args.new_sensor_ratio,
    cleanup_gt=cleanup_gt,
    localization_gt=localization_gt,
    x_axis_vec=x_axis_vec,
    y_axis_vec=y_axis_vec,
    dt=params['dt'],
    spiking=args.spiking,
)


if not args.evaluate:
    # view the system in action

    num_episodes = 10
    time_steps = 10000#100
    returns = np.zeros((num_episodes,))
    for e in range(num_episodes):
        obs = env.reset(goal_distance=params['goal_distance'])

        print(env.goal_state)

        # env.goal_object is the string name for the goal
        # env.goal_state[[0, 1]] is the 2D location for that goal
        semantic_goal = vocab[env.goal_object]

        distances = torch.Tensor(obs).unsqueeze(0)
        agent.localize(
            distances,
            map_id,
            env,
            use_localization_gt=args.use_localization_gt
        )
        action = np.zeros((2,))
        for s in range(params['episode_length']):
            env.render()
            env._render_extras()

            distances = torch.Tensor(obs).unsqueeze(0)
            velocity = torch.Tensor(action).unsqueeze(0)

            # TODO: need to give the semantic goal and the maze_id to the agent
            # action = agent.act(obs)
            action = agent.act(
                distances=distances,
                velocity=velocity,  # TODO: use real velocity rather than action, because of hitting walls
                semantic_goal=semantic_goal,
                map_id=map_id,
                item_memory=item_memory,

                env=env,  # only used for some ground truth options
                use_cleanup_gt=args.use_cleanup_gt,
                use_localization_gt=args.use_localization_gt,
            )

            # if args.ghost != 'none' and False:
            #     if args.ghost == 'trajectory_loc':
            #         # localization ghost
            #         agent_ssp = agent.localization_network(
            #             inputs=(torch.cat([velocity, distances, map_id], dim=1),),
            #             initial_ssp=agent.agent_ssp
            #         ).squeeze(0)
            #     elif args.ghost == 'snapshot_loc':
            #         # snapshot localization ghost
            #         agent_ssp = agent.snapshot_localization_network(torch.cat([distances, map_id], dim=1))
            #     agent_loc = ssp_to_loc(agent_ssp.squeeze(0).detach().numpy(), heatmap_vectors, xs, ys)
            #     # Scale to env coordinates, from (-5,5) to (0,13)
            #     agent_loc = ((agent_loc - xs[0]) / limit_range) * coarse_size
            #     env.render_ghost(x=agent_loc[0], y=agent_loc[1])

            if args.ghost:
                agent_loc = ssp_to_loc(agent.clean_agent_ssp.squeeze(0).detach().numpy(), heatmap_vectors, xs, ys)
                # Scale to env coordinates, from (-5,5) to (0,13)
                agent_loc = ((agent_loc - xs[0]) / limit_range) * coarse_size
                env.render_ghost(x=agent_loc[0], y=agent_loc[1])

            if args.normalize_action:
                mag = np.linalg.norm(action)
                if mag > 0.001:
                    action = action / np.linalg.norm(action)

            # Add small amount of noise to the action
            action += np.random.normal(size=2) * args.noise

            obs, reward, done, info = env.step(action)
            # print(obs)
            returns[e] += reward
            # if reward != 0:
            #    print(reward)
            # time.sleep(dt)
            if done:
                break

    print(returns)
else:
    n_mazes = 50  # number of maze/memory combos to evaluate on
    # evaluate system by recording performance.
    if not os.path.exists('output'):
        os.makedirs('output')
    fname = 'output/results_integ_noise{}_s{}_{}envs'.format(args.noise, args.env_seed, n_mazes)
    if args.normalize_action:
        fname += '_norm_action'
    if args.use_cleanup_gt:
        fname += '_cleanup_gt'
    if args.use_localization_gt:
        fname += '_loc_gt'
    fname += '.npz'

    n_mazes_max = 10  # number of mazes to evaluate on
    num_episodes = 100  # number of episodes per maze
    returns = np.zeros((n_mazes, num_episodes,))
    start_locs = np.zeros((n_mazes, num_episodes, 2))
    goal_locs = np.zeros((n_mazes, num_episodes, 2))
    vocab_rng = np.random.RandomState(seed=args.env_seed)
    if args.record_trajectories:
        # initialize to -1, so negative entries mean the goal has already been reached
        trajectories = np.ones((n_mazes, num_episodes, params['episode_length'], 2))*-1
    for total_maze_index in range(n_mazes):
        maze_index = total_maze_index % n_mazes_max
        print("Maze {} of {}".format(total_maze_index + 1, n_mazes))
        # Generate a new environment object for each maze
        map_array = coarse_mazes[maze_index, :, :]
        object_locations = OrderedDict()
        vocab = {}
        for i in range(n_goals):
            sp_name = possible_objects[i]
            if args.use_dataset_goals:
                object_locations[sp_name] = goals_scaled[args.maze_index, i]  # using goal locations from the dataset
            else:
                # If set to None, the environment will choose a random free space on init
                object_locations[sp_name] = None
            # vocab[sp_name] = spa.SemanticPointer(ssp_dim)
            vocab[sp_name] = nengo_spa.SemanticPointer(data=vocab_rng.uniform(-1, 1, size=args.dim)).normalized()
        env = GridWorldEnv(
            map_array=map_array,
            object_locations=object_locations,  # object locations explicitly chosen so a fixed SSP memory can be given
            observations=obs_dict,
            movement_type=params['movement_type'],
            max_lin_vel=params['max_lin_vel'],
            max_ang_vel=params['max_ang_vel'],
            continuous=params['continuous'],
            max_steps=params['episode_length'],
            fixed_episode_length=params['fixed_episode_length'],
            wall_penalty=params['wall_penalty'],
            movement_cost=params['movement_cost'],
            goal_reward=params['goal_reward'],
            dt=params['dt'],
            screen_width=300,
            screen_height=300,
            debug_ghost=True,
            seed=args.env_seed + total_maze_index,
        )

        # Fill the item memory with the correct SSP for remembering the goal locations
        item_memory = nengo_spa.SemanticPointer(data=np.zeros((args.dim,)))

        for i in range(n_goals):
            sp_name = possible_objects[i]
            x_env, y_env = env.object_locations[sp_name][[0, 1]]

            # Need to scale to SSP coordinates
            # Env is 0 to 13, SSP is -5 to 5
            x = ((x_env - 0) / coarse_size) * limit_range + xs[0]
            y = ((y_env - 0) / coarse_size) * limit_range + ys[0]

            item_memory += vocab[sp_name] * encode_point(x, y, x_axis_sp, y_axis_sp)
        # item_memory.normalize()
        item_memory = item_memory.normalized()

        # the unsqueeze is to add the batch dimension
        map_id = torch.Tensor(maze_sps[maze_index, :]).unsqueeze(0)

        for e in range(num_episodes):
            obs = env.reset(goal_distance=params['goal_distance'])

            # record starting position and goal for this episode
            start_locs[total_maze_index, e, :] = env.state[[0, 1]]
            goal_locs[total_maze_index, e, :] = env.goal_state[[0, 1]]

            # env.goal_object is the string name for the goal
            # env.goal_state[[0, 1]] is the 2D location for that goal
            semantic_goal = vocab[env.goal_object]

            distances = torch.Tensor(obs).unsqueeze(0)
            agent.localize(
                distances,
                map_id,
                env,
                use_localization_gt=args.use_localization_gt
            )
            action = np.zeros((2,))
            for s in range(params['episode_length']):
                # env.render()
                # env._render_extras()

                distances = torch.Tensor(obs).unsqueeze(0)
                velocity = torch.Tensor(action).unsqueeze(0)

                # TODO: need to give the semantic goal and the maze_id to the agent
                # action = agent.act(obs)
                action = agent.act(
                    distances=distances,
                    velocity=velocity,  # TODO: use real velocity rather than action, because of hitting walls
                    semantic_goal=semantic_goal,
                    map_id=map_id,
                    item_memory=item_memory,

                    env=env,  # only used for some ground truth options
                    use_cleanup_gt=args.use_cleanup_gt,
                    use_localization_gt=args.use_localization_gt,
                )

                if args.normalize_action:
                    mag = np.linalg.norm(action)
                    if mag > 0.001:
                        action = action / np.linalg.norm(action)

                # Add small amount of noise to the action
                action += np.random.normal(size=2) * args.noise

                obs, reward, done, info = env.step(action)
                returns[total_maze_index, e] += reward

                if args.record_trajectories:
                    trajectories[total_maze_index, e, s, :] = env.state[[0, 1]]

                if done:
                    break

            print("Maze {}, EP {}, Ret: {}".format(total_maze_index + 1, e + 1, returns[maze_index, e]))

    if args.record_trajectories:
        np.savez(
            fname,
            returns=returns,
            start_locs=start_locs,
            goal_locs=goal_locs,
            trajectories=trajectories,
        )
    else:
        np.savez(
            fname,
            returns=returns,
            start_locs=start_locs,
            goal_locs=goal_locs,
        )

