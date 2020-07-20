import numpy as np
import torch


class RandomTrajectoryAgent(object):

    def __init__(self,
                 obs_index_dict,
                 lin_vel_rayleigh_scale=.13,
                 rot_vel_std=330,
                 dt=0.1, #0.02
                 avoid_walls=True,
                 avoidance_weight=.005,
                 n_sensors=36,
                 ):

        self.obs_index_dict = obs_index_dict

        # Initialize simulated head direction
        self.th = 0

        self.lin_vel_rayleigh_scale = lin_vel_rayleigh_scale
        self.rot_vel_std = rot_vel_std
        self.dt = dt

        # If true, modify the random motion to be biased towards avoiding walls
        self.avoid_walls = avoid_walls

        # Note that the two axes are 90 degrees apart to work
        self.avoidance_matrix = np.zeros((2, n_sensors))
        self.avoidance_matrix[0, :] = np.roll(np.linspace(-1., 1., n_sensors), 3*n_sensors//4) * avoidance_weight
        self.avoidance_matrix[1, :] = np.linspace(-1., 1., n_sensors) * avoidance_weight

    def act(self, obs):

        distances = obs[self.obs_index_dict['dist_sensors']]

        lin_vel = np.random.rayleigh(scale=self.lin_vel_rayleigh_scale)
        ang_vel = np.random.normal(loc=0, scale=self.rot_vel_std) * np.pi / 180

        cartesian_vel_x = np.cos(self.th) * lin_vel
        cartesian_vel_y = np.sin(self.th) * lin_vel

        # for testing obstacle avoidance only
        # cartesian_vel_x = 0
        # cartesian_vel_y = 0

        self.th = self.th + ang_vel * self.dt
        if self.th > np.pi:
            self.th -= 2*np.pi
        elif self.th < -np.pi:
            self.th += 2*np.pi

        # optional wall avoidance bias
        if self.avoid_walls:
            vel_change = np.dot(self.avoidance_matrix, distances)
            cartesian_vel_x += vel_change[0]
            cartesian_vel_y += vel_change[1]

        # TODO: add a goal achieving bias? To get a trajectory that covers more area

        return np.array([cartesian_vel_x, cartesian_vel_y]) * 3


class GoalFindingAgent(object):

    def __init__(self,
                 cleanup_network, localization_network, policy_network, snapshot_localization_network,
                 cleanup_gt, localization_gt, policy_gt, snapshot_localization_gt,
                 use_snapshot_localization=False,
                 ):

        self.cleanup_network = cleanup_network
        self.localization_network = localization_network
        self.policy_network = policy_network
        self.snapshot_localization_network = snapshot_localization_network

        # Ground truth versions of the above functions
        self.cleanup_gt = cleanup_gt
        self.localization_gt = localization_gt
        self.policy_gt = policy_gt
        self.snapshot_localization_gt = snapshot_localization_gt

        # If true, the snapshot localization network will be used every step
        self.use_snapshot_localization = use_snapshot_localization

        # need a reasonable default for the agent ssp
        # maybe use a snapshot localization network that only uses distance sensors to initialize it?
        self.agent_ssp = None

    def snapshot_localize(self, distances, map_id, env, use_localization_gt=False):

        if use_localization_gt:
            self.agent_ssp = self.localization_gt(env)
        else:
            # inputs = torch.cat([torch.Tensor(distances), torch.Tensor(map_id)])
            inputs = torch.cat([distances, map_id], dim=1)  # dim is set to 1 because there is a batch dimension
            self.agent_ssp = self.snapshot_localization_network(inputs)

            # normalize the agent SSP
            self.agent_ssp = self.agent_ssp / float(np.linalg.norm(self.agent_ssp.detach().numpy()))

    def act(self, distances, velocity, semantic_goal, map_id, item_memory, env,
            use_cleanup_gt=False, use_localization_gt=False, use_policy_gt=False):
        """
        :param distances: Series of distance sensor measurements
        :param velocity: 2D velocity of the agent from last timestep
        :param semantic_goal: A semantic pointer for the item of interest
        :param map_id: Vector acting as context for the current map. Typically a one-hot vector
        :param item_memory: Spatial semantic pointer containing a summation of LOC*ITEM
        :param env: Instance of the environment, for use with ground truth methods
        :param use_cleanup_gt: If set, use a ground truth cleanup memory
        :param use_localization_gt: If set, use the ground truth localization
        :param use_policy_gt: If set, use a ground truth policy to compute the action
        :return: 2D holonomic velocity action
        """

        with torch.no_grad():
            if use_cleanup_gt:
                goal_ssp = self.cleanup_gt(env)
            else:
                noisy_goal_ssp = item_memory *~ semantic_goal
                goal_ssp = self.cleanup_network(torch.Tensor(noisy_goal_ssp.v).unsqueeze(0))
                # Normalize the result
                goal_ssp = goal_ssp / float(np.linalg.norm(goal_ssp.detach().numpy()))

            if use_localization_gt:
                self.agent_ssp = self.localization_gt(env)
            else:
                if self.use_snapshot_localization:
                    self.snapshot_localize(distances=distances, map_id=map_id, env=env, use_localization_gt=False)
                else:
                    # Just one step of the recurrent network
                    # dim is set to 1 because there is a batch dimension
                    # inputs are wrapped as a tuple because the network is expecting a tuple of tensors
                    self.agent_ssp = self.localization_network(
                        inputs=(torch.cat([velocity, distances, map_id], dim=1),),
                        initial_ssp=self.agent_ssp
                    ).squeeze(0)

            if use_policy_gt:
                vel_action = self.policy_gt(map_id=map_id, agent_ssp=self.agent_ssp, goal_ssp=goal_ssp, env=env)
            else:
                vel_action = self.policy_network(
                    torch.cat([map_id, self.agent_ssp, goal_ssp], dim=1)
                ).squeeze(0).detach().numpy()

            # TODO: possibly do a transform on the action output if the environment needs it

            return vel_action


class IntegSystemAgent(object):

    def __init__(self,
                 cleanup_network,
                 localization_network,
                 policy_network,
                 cleanup_gt,
                 localization_gt,
                 new_sensor_ratio,
                 x_axis_vec,
                 y_axis_vec,
                 dt,
                 spiking=False,
                 ):

        self.cleanup_network = cleanup_network
        self.localization_network = localization_network
        self.policy_network = policy_network

        # Ground truth versions of the above functions
        self.cleanup_gt = cleanup_gt
        self.localization_gt = localization_gt

        # need a reasonable default for the agent ssp
        # maybe use a snapshot localization network that only uses distance sensors to initialize it?
        self.agent_ssp = None
        self.clean_agent_ssp = None

        # fraction of weighting on new sensor measurement
        # old measurement will be (1 - ratio)
        self.new_sensor_ratio = new_sensor_ratio

        # used for updating position estimate based on velocity
        self.x_axis_vec = x_axis_vec
        self.y_axis_vec = y_axis_vec
        self.xf = np.fft.fft(self.x_axis_vec)
        self.yf = np.fft.fft(self.y_axis_vec)
        self.dt = dt

        self.spiking = spiking

    def localize(self, distances, map_id, env, use_localization_gt=False):

        if use_localization_gt:
            self.agent_ssp = self.localization_gt(env)
        else:
            if self.spiking:
                inputs = np.concatenate([distances.detach().numpy(), map_id.detach().numpy()], axis=1)
                self.agent_ssp = self.localization_network.predict(inputs)
                self.agent_ssp = torch.Tensor(self.agent_ssp / float(np.linalg.norm(self.agent_ssp)))
            else:
                # inputs = torch.cat([torch.Tensor(distances), torch.Tensor(map_id)])
                inputs = torch.cat([distances, map_id], dim=1)  # dim is set to 1 because there is a batch dimension
                self.agent_ssp = self.localization_network(inputs)

                # normalize the agent SSP
                self.agent_ssp = self.agent_ssp / float(np.linalg.norm(self.agent_ssp.detach().numpy()))

    def apply_velocity(self, ssp, vel):
        # extra zero index is because of the batch dimension
        vel_ssp = (self.xf**vel[0, 0]*self.dt)*(self.yf**vel[0, 1]*self.dt)
        return np.fft.ifft(np.fft.fft(ssp)*vel_ssp).real

    def act(self, distances, velocity, semantic_goal, map_id, item_memory, env,
            use_cleanup_gt=False, use_localization_gt=False):
        """
        :param distances: Series of distance sensor measurements
        :param velocity: 2D velocity of the agent from last timestep
        :param semantic_goal: A semantic pointer for the item of interest
        :param map_id: Vector acting as context for the current map. Typically a one-hot vector
        :param item_memory: Spatial semantic pointer containing a summation of LOC*ITEM
        :param env: Instance of the environment, for use with ground truth methods
        :param use_cleanup_gt: If set, use a ground truth cleanup memory
        :param use_localization_gt: If set, use the ground truth localization
        :return: 2D holonomic velocity action
        """

        with torch.no_grad():
            if use_localization_gt:
                self.agent_ssp = self.localization_gt(env).squeeze(0)
            else:
                if self.spiking:
                    inputs = np.concatenate([distances.detach().numpy(), map_id.detach().numpy()], axis=1)
                    agent_ssp_estimate = torch.Tensor(self.localization_network.predict(inputs)).squeeze(0)
                    agent_ssp_estimate = agent_ssp_estimate / float(np.linalg.norm(agent_ssp_estimate))
                else:
                    agent_ssp_estimate = self.localization_network(
                            inputs=torch.cat([distances, map_id], dim=1)
                        ).squeeze(0)

                    agent_ssp_estimate = agent_ssp_estimate / float(np.linalg.norm(agent_ssp_estimate.detach().numpy()))

                # TODO: do some appropriate mixing here
                self.agent_ssp = torch.Tensor(
                    self.apply_velocity(
                        self.agent_ssp.detach().numpy(),
                        velocity.detach().numpy()
                    )
                ).squeeze(0) * (1 - self.new_sensor_ratio) + agent_ssp_estimate * self.new_sensor_ratio
                # ).squeeze(0) * self.new_sensor_ratio + agent_ssp_estimate * (1 - self.new_sensor_ratio)

            if use_cleanup_gt:
                goal_ssp = self.cleanup_gt(env)
                # NOTE: should the agent SSP be cleaned as well?
                self.clean_agent_ssp = self.agent_ssp.unsqueeze(0)
            else:
                noisy_goal_ssp = item_memory *~ semantic_goal
                goal_ssp = self.cleanup_network(torch.Tensor(noisy_goal_ssp.v).unsqueeze(0))
                # Normalize the result
                goal_ssp = goal_ssp / float(np.linalg.norm(goal_ssp.detach().numpy()))

                if not use_localization_gt:
                    # clean up the agent estimate for use in the network
                    self.clean_agent_ssp = self.cleanup_network(torch.Tensor(self.agent_ssp).unsqueeze(0))
                    self.clean_agent_ssp = self.clean_agent_ssp / float(np.linalg.norm(self.clean_agent_ssp.detach().numpy()))
                else:
                    self.clean_agent_ssp = self.agent_ssp.unsqueeze(0)

            if self.spiking:
                inputs = np.concatenate(
                    [
                        map_id.detach().numpy(),
                        self.clean_agent_ssp.detach().numpy(),
                        goal_ssp.detach().numpy()
                    ],
                    axis=1
                )
                vel_action = self.policy_network.predict(inputs)[0, :]
            else:
                vel_action = self.policy_network(
                    # torch.cat([map_id, self.agent_ssp, goal_ssp], dim=1)
                    torch.cat([map_id, self.clean_agent_ssp, goal_ssp], dim=1)
                ).squeeze(0).detach().numpy()

            # TODO: possibly do a transform on the action output if the environment needs it

            return vel_action
