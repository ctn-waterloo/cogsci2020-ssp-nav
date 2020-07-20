import numpy as np
import argparse
import torch
from gridworlds.envs import GridWorldEnv, generate_obs_dict
from gridworlds.constants import possible_objects

from ssp_navigation.utils.agents import GoalFindingAgent
# TODO: really should clean up this nonsense at some point
import nengo.spa as nspa
import nengo_spa as spa
from collections import OrderedDict
from spatial_semantic_pointers.utils import encode_point, ssp_to_loc, get_heatmap_vectors

from ssp_navigation.utils.models import FeedForward, MLP
from ssp_navigation.utils.training import LocalizationModel
from ssp_navigation.utils.path import solve_maze

from ssp_navigation.utils.encodings import get_encoding_function, get_encoding_heatmap_vectors

parser = argparse.ArgumentParser('Run the full system on a maze task and record the trajectories taken')

parser.add_argument('--fname', type=str, default='fs_traj_data.npz', help='file to save the trajectory data to')

parser.add_argument('--n-trajectories', type=int, default=10, help='number of trajectories per goal')
# parser.add_argument('--n-goals', type=int, help='number of different goals per maze layout')
# parser.add_argument('--n-mazes', type=int, help='number of different maze layouts')
parser.add_argument('--start-x', type=float, default=0, help='x-coord of the agent start location')
parser.add_argument('--start-y', type=float, default=0, help='y-coord of the agent start location')
parser.add_argument('--goal-x', type=float, default=1, help='x-coord of the goal location')
parser.add_argument('--goal-y', type=float, default=1, help='y-coord of the goal location')

parser.add_argument('--n-goals', type=int, default=5, help='number of different goals to get to')


parser.add_argument('--cleanup-network', type=str,
                    default='networks/cleanup_network.pt',
                    help='SSP cleanup network')
parser.add_argument('--localization-network', type=str,
                    default='networks/localization_network.pt',
                    help='localization from sensors and velocity')
parser.add_argument('--policy-network', type=str,
                    # default='networks/policy_network.pt',
                    default='datasets/mixed_style_100mazes_100goals_64res_13size_13seed/policy/med_dim_exps/hex-ssp/dim256/scaling0.5/mazes10/hidsize256/seed13/hex-ssp_100000train_random-sp_id_1layer_256units/Jan14_07-27-21/model.pt',
                    help='goal navigation network')
parser.add_argument('--snapshot-localization-network', type=str,
                    default='networks/snapshot_localization_network.pt',
                    help='localization from sensors without velocity, for initialization')
parser.add_argument('--n-hidden-layers-policy', type=int, default=1, help='number of hidden layers in the policy network')
parser.add_argument('--dataset', type=str,
                    default='datasets/mixed_style_100mazes_100goals_64res_13size_13seed/maze_dataset.npz',
                    help='dataset to get the maze layout from')
parser.add_argument('--maze-index', type=int, default=0, help='index within the dataset for the maze to use')
parser.add_argument('--noise', type=float, default=0.25, help='magnitude of gaussian noise to add to the actions')
# parser.add_argument('--use-snapshot-localization', action='store_true', help='localize with distance sensors only')
parser.add_argument('--use-dataset-goals', action='store_true', help='use only the goals the policy was trained on')
parser.add_argument('--ghost', type=str, default='trajectory_loc', choices=['none', 'snapshot_loc', 'trajectory_loc'],
                    help='what information should be displayed by the ghost')
parser.add_argument('--use-cleanup-gt', action='store_true')
parser.add_argument('--use-policy-gt', action='store_true')
parser.add_argument('--use-localization-gt', action='store_true')
parser.add_argument('--use-snapshot-localization-gt', action='store_true')

# network parameters
parser.add_argument('--policy-hidden-size', type=int, default=256, help='Size of the hidden layer in the model')
parser.add_argument('--cleanup-hidden-size', type=int, default=256, help='Size of the hidden layer in the model')
parser.add_argument('--localization-hidden-size', type=int, default=256, help='Size of the hidden layer in the model')
parser.add_argument('--n-hidden-layers', type=int, default=1, help='Number of hidden layers in the model')

# spatial encoding parameters
parser.add_argument('--spatial-encoding', type=str, default='hex-ssp',
                    choices=[
                        'ssp', 'hex-ssp', 'random', '2d', '2d-normalized', 'one-hot', 'hex-trig',
                        'trig', 'random-trig', 'random-proj', 'learned', 'frozen-learned',
                        'pc-gauss', 'pc-dog', 'tile-coding'
                    ],
                    help='coordinate encoding for agent location and goal')
parser.add_argument('--hex-freq-coef', type=float, default=2.5, help='constant to scale frequencies by for hex-trig')
parser.add_argument('--pc-gauss-sigma', type=float, default=0.25, help='sigma for the gaussians')
parser.add_argument('--pc-diff-sigma', type=float, default=0.5, help='sigma for subtracted gaussian in DoG')
parser.add_argument('--n-tiles', type=int, default=8, help='number of layers for tile coding')
parser.add_argument('--n-bins', type=int, default=0, help='number of bins for tile coding')
parser.add_argument('--ssp-scaling', type=float, default=0.5)

parser.add_argument('--maze-id-dim', type=int, default=256, help='Dimensionality for the Maze ID')
parser.add_argument('--seed', type=int, default=13, help='Seed for training and generating axis SSPs')
parser.add_argument('--dim', type=int, default=256, help='Dimensionality of the SSPs')

args = parser.parse_args()

rng = np.random.RandomState(seed=args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Assuming maze-id-type is random-sp
n_mazes = 100
id_size = args.maze_id_dim
maze_sps = np.zeros((n_mazes, args.maze_id_dim))

if args.use_policy_gt:
    for mi in range(n_mazes):
        maze_sps[mi, mi] = 1
else:
    # overwrite data
    for mi in range(n_mazes):
        maze_sps[mi, :] = nspa.SemanticPointer(args.maze_id_dim).v

# ssp_dim = 512
n_sensors = 36

params = {
    'continuous': True,
    'fov': 360,
    'n_sensors': n_sensors,
    'max_sensor_dist': 10,
    'normalize_dist_sensors': False,
    'movement_type': 'holonomic',
    'seed': args.seed,
    # 'map_style': args.map_style,
    'map_size': 10,
    'fixed_episode_length': False,  # Setting to false so location resets are not automatic
    'episode_length': 1000,  #200,
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

    'goal_distance': 0,#args.goal_distance  # 0 means completely random
}

obs_dict = generate_obs_dict(params)

np.random.seed(params['seed'])

data = np.load(args.dataset)

limit_low = 0
limit_high = data['coarse_mazes'].shape[2]

encoding_func, repr_dim = get_encoding_function(args, limit_low=limit_low, limit_high=limit_high)

# n_mazes by size by size
coarse_mazes = data['coarse_mazes']
coarse_size = coarse_mazes.shape[1]
n_maps = coarse_mazes.shape[0]

# map_id = np.zeros((n_maps,))
# map_id[args.maze_index] = 1
# map_id = torch.Tensor(map_id).unsqueeze(0)
map_id = torch.Tensor(maze_sps[args.maze_index, :]).unsqueeze(0)

# n_mazes by res by res
fine_mazes = data['fine_mazes']
xs = data['xs']
ys = data['ys']
res = fine_mazes.shape[1]
# print(data['xs'])
# xs = np.linspace(0, 13, res)
# ys = np.linspace(0, 13, res)

coarse_xs = np.linspace(xs[0], xs[-1], coarse_size)
coarse_ys = np.linspace(ys[0], ys[-1], coarse_size)

map_array = coarse_mazes[args.maze_index, :, :]

# x_axis_sp = spa.SemanticPointer(data=data['x_axis_sp'])
# y_axis_sp = spa.SemanticPointer(data=data['y_axis_sp'])
# heatmap_vectors = get_heatmap_vectors(xs, ys, x_axis_sp, y_axis_sp)
# coarse_heatmap_vectors = get_heatmap_vectors(coarse_xs, coarse_ys, x_axis_sp, y_axis_sp)


if args.use_policy_gt:
    heatmap_vectors = get_encoding_heatmap_vectors(xs, ys, args.dim, encoding_func)

# fixed random set of locations for the goals
limit_range = xs[-1] - xs[0]

goal_sps = data['goal_sps']
goals = data['goals']
# print(np.min(goals))
# print(np.max(goals))
goals_scaled = ((goals - xs[0]) / limit_range) * coarse_size
# print(np.min(goals_scaled))
# print(np.max(goals_scaled))

# Manually override the first goal with the one that will be used
# other goals are still used so that the memory has multiple items, making the task more challenging
goals_scaled[args.maze_index, 0, 0] = args.goal_x
goals_scaled[args.maze_index, 0, 1] = args.goal_y

n_goals = 5#10  # TODO: make this a parameter
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
    vocab[sp_name] = spa.SemanticPointer(data=np.random.uniform(-1, 1, size=args.dim)).normalized()

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
    dt=params['dt'],
    fixed_start=True,
    fixed_goal=True,
    screen_width=300,
    screen_height=300,
    debug_ghost=True,
)

# # hacking in the desired start and goal states
env.start_state[0] = args.start_x
env.start_state[1] = args.start_y
# env.goal_state[0] = args.goal_x
# env.goal_state[1] = args.goal_y
# # the first goal in the list is the one that will be used
# env.goal_object_index = 0
# env.goal_object = env.goal_object_list[env.goal_object_index]

# Fill the item memory with the correct SSP for remembering the goal locations
item_memory = spa.SemanticPointer(data=np.zeros((args.dim,)))


# hacking in specific goal locations for plot
goals = np.array(
    [
        [11, 2], #[10.4, 2.4], #[10, 3.5],  #blue
        [6, 10.4], #[6, 10],  # orange
        [1, 1.1], #[1.5, 2.5], #[2, 3],  # green
        [10.7, 9.0], #[10, 9],  # red
        [1, 11], #[4, 11.3], #[2, 11]  # purple
    ]
)

for i in range(n_goals):
    sp_name = possible_objects[i]
    if i < 5:
        env.object_locations[sp_name][[0, 1]] = goals[i, :]  # overriding goal locations
    x_env, y_env = env.object_locations[sp_name][[0, 1]]

    # # Need to scale to SSP coordinates
    # # Env is 0 to 13, SSP is -5 to 5
    # x = ((x_env - 0) / coarse_size) * limit_range + xs[0]
    # y = ((y_env - 0) / coarse_size) * limit_range + ys[0]
    # newer code does not do this scaling, any scaling would be in the 'encoding_func'
    x = x_env
    y = y_env

    # item_memory += vocab[sp_name] * encode_point(x, y, x_axis_sp, y_axis_sp)
    item_memory += vocab[sp_name] * spa.SemanticPointer(
        data=encoding_func(x=x, y=y)
    )
# item_memory.normalize()
item_memory = item_memory.normalized()

# Component functions of the full system

cleanup_network = FeedForward(input_size=args.dim, hidden_size=args.cleanup_hidden_size, output_size=args.dim)
if not args.use_cleanup_gt:
    cleanup_network.load_state_dict(torch.load(args.cleanup_network, map_location=lambda storage, loc: storage), strict=True)
    cleanup_network.eval()

# NOTE: localization network unused for these experiments, snapshot only
# # Input is x and y velocity plus the distance sensor measurements, plus map ID
# localization_network = LocalizationModel(
#     input_size=2 + n_sensors + n_maps,
#     unroll_length=1, #rollout_length,
#     sp_dim=args.dim
# )
# if not args.use_localization_gt:
#     localization_network.load_state_dict(torch.load(args.localization_network, map_location=lambda storage, loc: storage), strict=True)
#     localization_network.eval()

# if args.n_hidden_layers_policy == 1:
#     policy_network = FeedForward(input_size=id_size + args.dim * 2, output_size=2)
# else:
#     policy_network = MLP(input_size=id_size + args.dim * 2, output_size=2, n_layers=args.n_hidden_layers_policy)
policy_network = MLP(
    input_size=id_size + repr_dim * 2,
    hidden_size=args.policy_hidden_size,
    output_size=2,
    n_layers=args.n_hidden_layers
)
if not args.use_policy_gt:
    policy_network.load_state_dict(torch.load(args.policy_network, map_location=lambda storage, loc: storage), strict=True)
    policy_network.eval()

snapshot_localization_network = FeedForward(
    input_size=n_sensors + id_size,
    hidden_size=args.localization_hidden_size,
    output_size=args.dim,
)
if not args.use_snapshot_localization_gt:
    snapshot_localization_network.load_state_dict(torch.load(args.snapshot_localization_network, map_location=lambda storage, loc: storage), strict=True)
    snapshot_localization_network.eval()


# # Ground truth versions of the above functions
# planner = OptimalPlanner(continuous=True, directional=False)


def cleanup_gt(env):
    x = ((env.goal_state[0] - 0) / coarse_size) * limit_range + xs[0]
    y = ((env.goal_state[1] - 0) / coarse_size) * limit_range + ys[0]
    # goal_ssp = encode_point(x, y, x_axis_sp, y_axis_sp)
    goal_ssp = spa.SemanticPointer(
        data=encoding_func(x=x, y=y)
    )
    return torch.Tensor(goal_ssp.v).unsqueeze(0)


def localization_gt(env):
    x = ((env.state[0] - 0) / coarse_size) * limit_range + xs[0]
    y = ((env.state[1] - 0) / coarse_size) * limit_range + ys[0]
    # agent_ssp = encode_point(x, y, x_axis_sp, y_axis_sp)
    agent_ssp = spa.SemanticPointer(
        data=encoding_func(x=x, y=y)
    )
    return torch.Tensor(agent_ssp.v).unsqueeze(0)


def policy_gt(map_id, agent_ssp, goal_ssp, env, coarse_planning=True):
    if coarse_planning:
        maze = coarse_mazes[np.argmax(map_id)]

        if True:
            # Convert agent and goal SSP into 2D locations
            agent_loc = ssp_to_loc(agent_ssp.squeeze(0).detach().numpy(), heatmap_vectors, xs, ys)
            goal_loc = ssp_to_loc(goal_ssp.squeeze(0).detach().numpy(), heatmap_vectors, xs, ys)

            agent_loc_scaled = ((agent_loc - xs[0]) / limit_range) * coarse_size
            goal_loc_scaled = ((goal_loc - xs[0]) / limit_range) * coarse_size

            start_indices = np.round(agent_loc_scaled).astype(np.int32)
            goal_indices = np.round(goal_loc_scaled).astype(np.int32)
            # start_indices = np.ceil(agent_loc_scaled).astype(np.int32)
            # goal_indices = np.ceil(goal_loc_scaled).astype(np.int32)
        else:
            vs = np.tensordot(agent_ssp.squeeze(0).detach().numpy(), coarse_heatmap_vectors, axes=([0], [2]))
            start_indices = np.unravel_index(vs.argmax(), vs.shape)

            vs = np.tensordot(goal_ssp.squeeze(0).detach().numpy(), coarse_heatmap_vectors, axes=([0], [2]))
            goal_indices = np.unravel_index(vs.argmax(), vs.shape)

        # env.render_ghost(x=start_indices[0], y=start_indices[1])

        solved_maze = solve_maze(maze, start_indices=start_indices, goal_indices=goal_indices, full_solve=True, strict_cornering=True)

        # Return the action for the current location
        return solved_maze[start_indices[0], start_indices[1], :]
    else:
        maze = fine_mazes[np.argmax(map_id)]

        vs = np.tensordot(agent_ssp.squeeze(0).detach().numpy(), heatmap_vectors, axes=([0], [2]))
        start_indices = np.unravel_index(vs.argmax(), vs.shape)

        vs = np.tensordot(goal_ssp.squeeze(0).detach().numpy(), heatmap_vectors, axes=([0], [2]))
        goal_indices = np.unravel_index(vs.argmax(), vs.shape)

        solved_maze = solve_maze(maze, start_indices=start_indices, goal_indices=goal_indices, full_solve=False, strict_cornering=True)

        # Return the action for the current location
        return solved_maze[start_indices[0], start_indices[1], :]


agent = GoalFindingAgent(
    cleanup_network=cleanup_network,
    localization_network=None,#localization_network,
    policy_network=policy_network,
    snapshot_localization_network=snapshot_localization_network,

    use_snapshot_localization=True, #args.use_snapshot_localization,

    cleanup_gt=cleanup_gt,
    localization_gt=localization_gt,
    policy_gt=policy_gt,
    snapshot_localization_gt=localization_gt,  # Exact same function as regular localization ground truth
)

num_episodes = args.n_trajectories
time_steps = 1000 #10000#100
returns = np.zeros((num_episodes,))
# The x-y location of the agent on each time step of each episode, for drawing the trajectories
locations = np.zeros((args.n_goals, num_episodes, time_steps, 2))
for g in range(args.n_goals):

    # hacking in specific goal
    env.goal_object_index = g
    env.goal_object = env.goal_object_list[env.goal_object_index]
    x_env, y_env = env.object_locations[env.goal_object][[0, 1]]
    env.goal_state[0] = x_env
    env.goal_state[1] = y_env

    for e in range(num_episodes):
        # This will reset to the same start and goal states every time
        obs = env.reset(goal_distance=params['goal_distance'])

        # env.goal_object is the string name for the goal
        # env.goal_state[[0, 1]] is the 2D location for that goal
        semantic_goal = vocab[env.goal_object]

        distances = torch.Tensor(obs).unsqueeze(0)
        agent.snapshot_localize(
            distances,
            map_id,
            env,
            use_localization_gt=args.use_snapshot_localization_gt
        )

        # used for running a little longer than reaching the bounding box of the goal
        end_step = params['episode_length']

        action = np.zeros((2,))
        for s in range(params['episode_length']):
            # no rendering for these experiments, want them running fast
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
                use_policy_gt=args.use_policy_gt,
            )

            # if args.ghost != 'none':
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

            # Add small amount of noise to the action
            action += np.random.normal(size=2) * args.noise

            # Record the current location of the agent
            locations[g, e, s, 0] = env.state[0]
            locations[g, e, s, 1] = env.state[1]

            obs, reward, done, info = env.step(action)
            # print(obs)
            returns[e] += reward
            # if reward != 0:
            #    print(reward)
            # time.sleep(dt)
            if done:
                # break
                # run a few more steps to converge on the center of the object
                if end_step == params['episode_length']:
                    end_step = s + 10

            # break X steps after the goal bounding box has been touched
            if s >= end_step:
                break


# print(returns)

# save the trajectory data
np.savez(
    args.fname,
    locations=locations,
)
