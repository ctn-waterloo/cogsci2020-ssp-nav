import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np
import nengo
import tensorflow as tf
import pickle

try:
    import nengo_dl
except:
    pass


class FeedForward(nn.Module):
    """
    Single hidden layer feed-forward model
    """

    def __init__(self, input_size=512, hidden_size=512, output_size=512):
        super(FeedForward, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # For compatibility with DeepRL code
        self.feature_dim = hidden_size #output_size #hidden_size #FIXME, this should be hidden size, the downstream code needs modification

        self.input_layer = nn.Linear(self.input_size, self.hidden_size)
        self.output_layer = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, inputs):

        features = F.relu(self.input_layer(inputs))
        prediction = self.output_layer(features)

        return prediction

    def forward_activations(self, inputs):
        """Returns the hidden layer activations as well as the prediction"""

        features = F.relu(self.input_layer(inputs))
        prediction = self.output_layer(features)

        return prediction, features


class MLP(nn.Module):
    """
    Multi-layer feed-forward model
    """

    def __init__(self, input_size=512, hidden_size=512, output_size=512, n_layers=2, dropout_fraction=0.0):
        super(MLP, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        # Need to use ModuleList rather than a regular Python list so that the module correctly keeps track
        # of the parameters, and allows network.to(device) to work correctly
        self.inner_layers = nn.ModuleList()

        self.dropout = nn.Dropout(p=dropout_fraction)

        self.input_layer = nn.Linear(self.input_size, self.hidden_size)
        for i in range(self.n_layers - 1):
            self.inner_layers.append(nn.Linear(self.hidden_size, self.hidden_size))
        self.output_layer = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, inputs):

        features = self.dropout(F.relu(self.input_layer(inputs)))
        for i in range(self.n_layers - 1):
            features = self.dropout(F.relu(self.inner_layers[i](features)))
        prediction = self.output_layer(features)

        return prediction

    def forward_activations(self, inputs):
        """Returns the last hidden layer activations as well as the prediction"""

        features = self.dropout(F.relu(self.input_layer(inputs)))
        for i in range(self.n_layers - 1):
            features = self.dropout(F.relu(self.inner_layers[i](features)))
        prediction = self.output_layer(features)

        return prediction, features


class LearnedEncoding(nn.Module):

    def __init__(self, input_size=2, encoding_size=512, maze_id_size=512,
                 hidden_size=512, output_size=2, n_layers=1, dropout_fraction=0.0):
        super(LearnedEncoding, self).__init__()

        self.input_size = input_size
        self.encoding_size = encoding_size
        self.maze_id_size = maze_id_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        # Need to use ModuleList rather than a regular Python list so that the module correctly keeps track
        # of the parameters, and allows network.to(device) to work correctly
        self.inner_layers = nn.ModuleList()

        self.dropout = nn.Dropout(p=dropout_fraction)

        self.encoding_layer = nn.Linear(self.input_size, self.encoding_size)

        self.input_layer = nn.Linear(self.encoding_size*2 + self.maze_id_size, self.hidden_size)
        for i in range(self.n_layers - 1):
            self.inner_layers.append(nn.Linear(self.hidden_size, self.hidden_size))
        self.output_layer = nn.Linear(self.hidden_size, self.output_size)

    def forward_activations_encoding(self, inputs):
        """Returns the hidden layer activations as well as the prediction"""

        if self.maze_id_size == 0:
            loc_pos = inputs[:, :self.input_size]
            goal_pos = inputs[:, self.input_size:self.input_size * 2]

            loc_encoding = F.relu(self.encoding_layer(loc_pos))
            goal_encoding = F.relu(self.encoding_layer(goal_pos))
            features = self.dropout(F.relu(self.input_layer(torch.cat([loc_encoding, goal_encoding], dim=1))))
        else:
            maze_id = inputs[:, :self.maze_id_size]
            loc_pos = inputs[:, self.maze_id_size:self.maze_id_size + self.input_size]
            goal_pos = inputs[:, self.maze_id_size + self.input_size:self.maze_id_size + self.input_size*2]

            loc_encoding = F.relu(self.encoding_layer(loc_pos))
            goal_encoding = F.relu(self.encoding_layer(goal_pos))
            features = self.dropout(F.relu(self.input_layer(torch.cat([maze_id, loc_encoding, goal_encoding], dim=1))))

        for i in range(self.n_layers - 1):
            features = self.dropout(F.relu(self.inner_layers[i](features)))
        prediction = self.output_layer(features)

        return prediction, features, loc_encoding, goal_encoding

    def forward(self, inputs):

        return self.forward_activations_encoding(inputs)[0]


class EncodingLayer(nn.Module):

    def __init__(self, input_size=2, encoding_size=512):
        super(EncodingLayer, self).__init__()

        self.input_size = input_size
        self.encoding_size = encoding_size

        self.encoding_layer = nn.Linear(self.input_size, self.encoding_size)

    def forward(self, inputs):

        encoding = F.relu(self.encoding_layer(inputs))

        return encoding


class LearnedSSPEncoding(nn.Module):

    def __init__(self, input_size=2, encoding_size=512, maze_id_size=512,
                 hidden_size=512, output_size=2, n_layers=1, dropout_fraction=0.0):
        """This model learns the phi values of the SSP encoding along with the rest of the network"""
        super(LearnedSSPEncoding, self).__init__()

        self.input_size = input_size
        self.encoding_size = encoding_size
        self.maze_id_size = maze_id_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        # Learnable SSP encoding
        self.encoding_layer = SSPTransform(coord_dim=self.input_size, ssp_dim=self.encoding_size)

        # Need to use ModuleList rather than a regular Python list so that the module correctly keeps track
        # of the parameters, and allows network.to(device) to work correctly
        self.inner_layers = nn.ModuleList()

        self.dropout = nn.Dropout(p=dropout_fraction)

        self.input_layer = nn.Linear(self.encoding_size*2 + self.maze_id_size, self.hidden_size)
        for i in range(self.n_layers - 1):
            self.inner_layers.append(nn.Linear(self.hidden_size, self.hidden_size))
        self.output_layer = nn.Linear(self.hidden_size, self.output_size)

    def forward_activations_encoding(self, inputs):
        """Returns the hidden layer activations as well as the prediction"""

        if self.maze_id_size == 0:
            loc_pos = inputs[:, :self.input_size]
            goal_pos = inputs[:, self.input_size:self.input_size * 2]

            loc_encoding = self.encoding_layer(loc_pos)
            goal_encoding = self.encoding_layer(goal_pos)
            features = self.dropout(F.relu(self.input_layer(torch.cat([loc_encoding, goal_encoding], dim=1))))
        else:
            maze_id = inputs[:, :self.maze_id_size]
            loc_pos = inputs[:, self.maze_id_size:self.maze_id_size + self.input_size]
            goal_pos = inputs[:, self.maze_id_size + self.input_size:self.maze_id_size + self.input_size*2]

            loc_encoding = self.encoding_layer(loc_pos)
            goal_encoding = self.encoding_layer(goal_pos)
            features = self.dropout(F.relu(self.input_layer(torch.cat([maze_id, loc_encoding, goal_encoding], dim=1))))

        for i in range(self.n_layers - 1):
            features = self.dropout(F.relu(self.inner_layers[i](features)))
        prediction = self.output_layer(features)

        return prediction, features, loc_encoding, goal_encoding

    def forward(self, inputs):

        return self.forward_activations_encoding(inputs)[0]


class SSPTransform(nn.Module):

    def __init__(self, coord_dim, ssp_dim):
        super(SSPTransform, self).__init__()

        # dimensionality of the input coordinates
        self.coord_dim = coord_dim

        # dimensionality of the SSP
        self.ssp_dim = ssp_dim

        # number of phi parameters to learn
        self.n_param = (ssp_dim-1) // 2

        self.phis = nn.Parameter(torch.Tensor(self.coord_dim, self.n_param))

        # initialize parameters
        torch.nn.init.uniform_(self.phis, a=-np.pi + 0.001, b=np.pi - 0.001)

        # number of phis, plus constant, plus potential nyquist if even
        self.tau_len = (self.ssp_dim // 2) + 1
        # constants used in the transformation
        # first dimension is batch dimension, set to 1 to be broadcastable
        self.const_phase = nn.Parameter(
            torch.zeros(1, self.ssp_dim, self.tau_len),
            requires_grad=False
        )
        for a in range(self.ssp_dim):
            for k in range(self.tau_len):
                self.const_phase[:, a, k] = 2*np.pi*k*a/self.ssp_dim

        # The 2/N or 1/N scaling applied outside of the cos
        # 2/N on all terms with phi, 1/N on constant and nyquist if it exists
        self.const_scaling = nn.Parameter(
            torch.ones(1, 1, self.tau_len)*2./self.ssp_dim,
            requires_grad=False
        )
        self.const_scaling[:, :, 0] = 1./self.ssp_dim
        if self.ssp_dim % 2 == 0:
            self.const_scaling[:, :, -1] = 1. / self.ssp_dim

    def forward(self, inputs):

        batch_size = inputs.shape[0]

        full_phis = torch.zeros(self.coord_dim, self.tau_len, dtype=inputs.dtype)
        full_phis[:, 1:self.n_param + 1] = self.phis
        full_phis = full_phis.to(inputs.device)
        shift = torch.zeros(batch_size, 1, self.tau_len, dtype=inputs.dtype).to(inputs.device)
        shift[:, 0, :] = torch.mm(inputs, full_phis)

        return (torch.cos(shift.to(inputs.device) + self.const_phase) * self.const_scaling).sum(axis=2)


# class LearnedEncoding(nn.Module):
#
#     def __init__(self, input_size=2, encoding_size=512, maze_id_size=512, hidden_size=512, output_size=2):
#         super(LearnedEncoding, self).__init__()
#
#         self.input_size = input_size
#         self.encoding_size = encoding_size
#         self.maze_id_size = maze_id_size
#         self.hidden_size = hidden_size
#         self.output_size = output_size
#
#         self.encoding_layer = nn.Linear(self.input_size, self.encoding_size)
#         # self.input_layer = nn.Linear(self.encoding_size, self.hidden_size)
#         self.input_layer = nn.Linear(self.encoding_size*2 + self.maze_id_size, self.hidden_size)
#         self.output_layer = nn.Linear(self.hidden_size, self.output_size)
#
#     def forward_activations_encoding(self, inputs):
#         """Returns the hidden layer activations as well as the prediction"""
#
#         maze_id = inputs[:, :self.maze_id_size]
#         loc_pos = inputs[:, self.maze_id_size:self.maze_id_size + self.input_size]
#         goal_pos = inputs[:, self.maze_id_size + self.input_size:self.maze_id_size + self.input_size*2]
#
#         loc_encoding = F.relu(self.encoding_layer(loc_pos))
#         goal_encoding = F.relu(self.encoding_layer(goal_pos))
#         features = F.relu(self.input_layer(torch.cat([maze_id, loc_encoding, goal_encoding], dim=1)))
#         prediction = self.output_layer(features)
#
#         return prediction, features, loc_encoding, goal_encoding
#
#     def forward(self, inputs):
#
#         return self.forward_activations_encoding(inputs)[0]


class TwoLayer(nn.Module):

    def __init__(self, input_size=2, encoding_size=512, hidden_size=512, output_size=2):
        super(TwoLayer, self).__init__()

        self.input_size = input_size
        self.encoding_size = encoding_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.encoding_layer = nn.Linear(self.input_size, self.encoding_size)
        self.input_layer = nn.Linear(self.encoding_size, self.hidden_size)
        self.output_layer = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, inputs):
        encoding = F.relu(self.encoding_layer(inputs))
        features = F.relu(self.input_layer(encoding))
        prediction = self.output_layer(features)

        return prediction

    def forward_activations(self, inputs):
        """Returns the hidden layer activations as well as the prediction"""

        encoding = F.relu(self.encoding_layer(inputs))
        features = F.relu(self.input_layer(encoding))
        prediction = self.output_layer(features)

        return prediction, features

    def forward_activations_encoding(self, inputs):
        """Returns the hidden layer activations and encoding layer activations, as well as the prediction"""

        encoding = F.relu(self.encoding_layer(inputs))
        features = F.relu(self.input_layer(encoding))
        prediction = self.output_layer(features)

        return prediction, features, encoding


def load_model(model_path, params_path, n_mazes):

    with open(params_path, "r") as f:
        params = json.load(f)

    if params['maze_id_type'] == 'ssp':
        id_size = params['dim']
    elif params['maze_id_type'] == 'one-hot':
        id_size = n_mazes
    else:
        raise NotImplementedError

    # Dimension of location representation is dependent on the encoding used
    if params['spatial_encoding'] == 'ssp':
        repr_dim = params['dim']
    elif params['spatial_encoding'] == 'random':
        repr_dim = params['dim']
    elif params['spatial_encoding'] == '2d':
        repr_dim = 2
    elif params['spatial_encoding'] == 'learned':
        repr_dim = 2
    elif params['spatial_encoding'] == '2d-normalized':
        repr_dim = 2
    elif params['spatial_encoding'] == 'one-hot':
        repr_dim = int(np.sqrt(params['dim'])) ** 2
    elif params['spatial_encoding'] == 'trig':
        repr_dim = params['dim']
    elif params['spatial_encoding'] == 'random-trig':
        repr_dim = params['dim']
    elif params['spatial_encoding'] == 'random-proj':
        repr_dim = params['dim']
    else:
        raise NotImplementedError

    if params['spatial_encoding'] == 'learned':
        # input is maze, loc, goal ssps, output is 2D direction to move
        model = LearnedEncoding(input_size=repr_dim, maze_id_size=id_size, hidden_size=512, output_size=2)
    else:
        # input is maze, loc, goal ssps, output is 2D direction to move
        model = FeedForward(input_size=id_size + repr_dim * 2, output_size=2)

    model.load_state_dict(torch.load(model_path), strict=False)

    return model, params['maze_id_type']


class SpikingPolicy(object):

    def __init__(self, param_file, dim=256, maze_id_dim=256, hidden_size=1024, net_seed=13, n_steps=30):
        self.param_file = param_file
        self.net = nengo.Network(seed=net_seed)
        with self.net:
            # set some default parameters for the neurons that will make
            # the training progress more smoothly
            # net.config[nengo.Ensemble].max_rates = nengo.dists.Choice([100])
            # net.config[nengo.Ensemble].intercepts = nengo.dists.Choice([0])
            self.net.config[nengo.Connection].synapse = None
            neuron_type = nengo.LIF(amplitude=0.01)

            # this is an optimization to improve the training speed,
            # since we won't require stateful behaviour in this example
            nengo_dl.configure_settings(stateful=False)

            # the input node that will be used to feed in (context, location, goal)
            inp = nengo.Node(np.zeros((dim * 2 + maze_id_dim,)))

            x = nengo_dl.Layer(tf.keras.layers.Dense(units=hidden_size))(inp)
            x = nengo_dl.Layer(neuron_type)(x)

            out = nengo_dl.Layer(tf.keras.layers.Dense(units=2))(x)

            self.out_p = nengo.Probe(out, label="out_p")
            self.out_p_filt = nengo.Probe(out, synapse=0.1, label="out_p_filt")

        # self.sim = nengo_dl.Simulator(self.net, minibatch_size=1)
        # self.sim.load_params(param_file)
        # self.sim.compile(loss={self.out_p_filt: mse_loss})
        self.n_steps = n_steps

    def predict(self, inputs):


        # param_file = 'networks/policy_params_1000000samples_250epochs'
        # dim = 256
        # maze_id_dim = 256
        # hidden_size = 1024
        # net_seed = 13
        # n_steps = 30
        # self.net = nengo.Network(seed=net_seed)
        # with self.net:
        #     # set some default parameters for the neurons that will make
        #     # the training progress more smoothly
        #     # net.config[nengo.Ensemble].max_rates = nengo.dists.Choice([100])
        #     # net.config[nengo.Ensemble].intercepts = nengo.dists.Choice([0])
        #     self.net.config[nengo.Connection].synapse = None
        #     neuron_type = nengo.LIF(amplitude=0.01)
        #
        #     # this is an optimization to improve the training speed,
        #     # since we won't require stateful behaviour in this example
        #     nengo_dl.configure_settings(stateful=False)
        #
        #     # the input node that will be used to feed in (context, location, goal)
        #     inp = nengo.Node(np.zeros((dim * 2 + maze_id_dim,)))
        #
        #     x = nengo_dl.Layer(tf.keras.layers.Dense(units=hidden_size))(inp)
        #     x = nengo_dl.Layer(neuron_type)(x)
        #
        #     out = nengo_dl.Layer(tf.keras.layers.Dense(units=2))(x)
        #
        #     self.out_p = nengo.Probe(out, label="out_p")
        #     self.out_p_filt = nengo.Probe(out, synapse=0.1, label="out_p_filt")
        #
        self.sim = nengo_dl.Simulator(self.net, minibatch_size=1)
        self.sim.load_params(self.param_file)
        self.sim.compile(loss={self.out_p_filt: mse_loss})
        # self.n_steps = n_steps





        tiled_input = np.tile(inputs[:, None, :], (1, self.n_steps, 1))

        pred_eval = self.sim.predict(tiled_input)
        return pred_eval[self.out_p_filt][:, 10:, :].mean(axis=1)


class SpikingLocalization(object):

    def __init__(self, param_file, dim=256, maze_id_dim=256, n_sensors=36, hidden_size=1024, net_seed=13, n_steps=30):
        self.net = nengo.Network(seed=net_seed)
        with self.net:
            # set some default parameters for the neurons that will make
            # the training progress more smoothly
            # net.config[nengo.Ensemble].max_rates = nengo.dists.Choice([100])
            # net.config[nengo.Ensemble].intercepts = nengo.dists.Choice([0])
            self.net.config[nengo.Connection].synapse = None
            neuron_type = nengo.LIF(amplitude=0.01)

            # this is an optimization to improve the training speed,
            # since we won't require stateful behaviour in this example
            nengo_dl.configure_settings(stateful=False)

            # the input node that will be used to feed in (context, location, goal)
            inp = nengo.Node(np.zeros((n_sensors * 4 + maze_id_dim,)))

            x = nengo_dl.Layer(tf.keras.layers.Dense(units=hidden_size))(inp)
            x = nengo_dl.Layer(neuron_type)(x)

            out = nengo_dl.Layer(tf.keras.layers.Dense(units=dim))(x)

            self.out_p = nengo.Probe(out, label="out_p")
            self.out_p_filt = nengo.Probe(out, synapse=0.1, label="out_p_filt")

        self.sim = nengo_dl.Simulator(self.net, minibatch_size=1)
        self.sim.load_params(param_file)
        self.sim.compile(loss={self.out_p_filt: mse_loss})
        self.n_steps = n_steps

    def predict(self, inputs):
        tiled_input = np.tile(inputs[:, None, :], (1, self.n_steps, 1))

        pred_eval = self.sim.predict(tiled_input)
        return pred_eval[self.out_p_filt][:, 10:, :].mean(axis=1)


def mse_loss(y_true, y_pred):
    return tf.metrics.MSE(
        y_true[:, -1], y_pred[:, -1]
    )


class SpikingPolicyNengo(object):

    def __init__(self, param_file, dim=256, maze_id_dim=256, hidden_size=2048, n_layers=2, net_seed=13, n_steps=30):
        self.param_file = param_file
        self.net = nengo.Network(seed=net_seed)
        with self.net:
            # set some default parameters for the neurons that will make
            # the training progress more smoothly
            # net.config[nengo.Ensemble].max_rates = nengo.dists.Choice([100])
            # net.config[nengo.Ensemble].intercepts = nengo.dists.Choice([0])
            self.net.config[nengo.Connection].synapse = None
            self.net.config[nengo.Connection].transform = nengo_dl.dists.Glorot()
            neuron_type = nengo.LIF(amplitude=0.01)

            # this is an optimization to improve the training speed,
            # since we won't require stateful behaviour in this example
            nengo_dl.configure_settings(stateful=False)

            # the input node that will be used to feed in (context, location, goal)
            inp = nengo.Node(np.zeros((dim * 2 + maze_id_dim,)))

            if '.pkl' in param_file:
                print("Loading values from pickle file")

                policy_params = pickle.load(open(param_file, 'rb'))
                policy_inp_params = policy_params[0]
                policy_ens_params = policy_params[1]
                policy_out_params = policy_params[2]

                hidden_ens = nengo.Ensemble(
                    n_neurons=hidden_size,
                    dimensions=1,
                    # dimensions=args.dim * 2 + args.maze_id_dim,
                    neuron_type=neuron_type,
                    **policy_ens_params
                )

                out = nengo.Node(size_in=2)

                if n_layers == 1:
                    conn_in = nengo.Connection(
                        inp, hidden_ens.neurons, synapse=None,
                        **policy_inp_params
                    )
                    conn_out = nengo.Connection(
                        hidden_ens.neurons, out, synapse=None,
                        **policy_out_params
                    )
                elif n_layers == 2:
                    policy_mid_params = policy_params[3]
                    policy_ens_two_params = policy_params[4]

                    hidden_ens_two = nengo.Ensemble(
                        n_neurons=hidden_size,
                        dimensions=1,
                        neuron_type=neuron_type,
                        **policy_ens_two_params
                    )

                    conn_in = nengo.Connection(
                        inp, hidden_ens.neurons, synapse=0.001,
                        **policy_inp_params
                    )
                    conn_mid = nengo.Connection(
                        hidden_ens.neurons, hidden_ens_two.neurons, synapse=0.001,
                        **policy_mid_params
                    )
                    conn_out = nengo.Connection(
                        hidden_ens_two.neurons, out, synapse=0.001,
                        **policy_out_params
                    )
                else:
                    raise NotImplementedError
            else:

                hidden_ens = nengo.Ensemble(
                    n_neurons=hidden_size,
                    dimensions=1,
                    neuron_type=neuron_type
                )

                out = nengo.Node(size_in=2)

                if n_layers == 1:

                    conn_in = nengo.Connection(inp, hidden_ens.neurons, synapse=None)
                    conn_out = nengo.Connection(hidden_ens.neurons, out, synapse=None)
                elif n_layers == 2:

                    hidden_ens_two = nengo.Ensemble(
                        n_neurons=hidden_size,
                        dimensions=1,
                        neuron_type=neuron_type
                    )

                    conn_in = nengo.Connection(inp, hidden_ens.neurons, synapse=None)
                    conn_mid = nengo.Connection(hidden_ens.neurons, hidden_ens_two.neurons, synapse=None)
                    conn_out = nengo.Connection(hidden_ens_two.neurons, out, synapse=None)
                else:
                    raise NotImplementedError

            out_p = nengo.Probe(out, label="out_p")
            self.out_p_filt = nengo.Probe(out, synapse=0.1, label="out_p_filt")

        # self.sim = nengo_dl.Simulator(self.net, minibatch_size=1)
        ## self.sim.load_params(param_file)
        # self.sim.compile(loss={self.out_p_filt: mse_loss})
        self.n_steps = n_steps

    def predict(self, inputs):

        self.sim = nengo_dl.Simulator(self.net, minibatch_size=1)
        # self.sim.load_params(self.param_file)
        self.sim.compile(loss={self.out_p_filt: mse_loss})
        # self.n_steps = n_steps

        tiled_input = np.tile(inputs[:, None, :], (1, self.n_steps, 1))

        pred_eval = self.sim.predict(tiled_input)
        return pred_eval[self.out_p_filt][:, 10:, :].mean(axis=1)


class SpikingLocalizationNengo(object):

    def __init__(self, param_file, dim=256, maze_id_dim=256, n_sensors=36, hidden_size=4096, n_layers=1, net_seed=13, n_steps=30):
        self.param_file = param_file
        self.net = nengo.Network(seed=net_seed)
        with self.net:
            # set some default parameters for the neurons that will make
            # the training progress more smoothly

            self.net.config[nengo.Connection].synapse = None
            neuron_type = nengo.LIF(amplitude=0.01)

            # this is an optimization to improve the training speed,
            # since we won't require stateful behaviour in this example
            self.net.config[nengo.Connection].transform = nengo_dl.dists.Glorot()
            nengo_dl.configure_settings(stateful=False)

            # the input node that will be used to feed in (context, location, goal)
            inp = nengo.Node(np.zeros((36 * 4 + maze_id_dim,)))

            if '.pkl' in param_file:
                print("Loading values from pickle file")

                localization_params = pickle.load(open(param_file, 'rb'))
                localization_inp_params = localization_params[0]
                localization_ens_params = localization_params[1]
                localization_out_params = localization_params[2]

                hidden_ens = nengo.Ensemble(
                    n_neurons=hidden_size,
                    # dimensions=36*4 + args.maze_id_dim,
                    dimensions=1,
                    neuron_type=neuron_type,
                    **localization_ens_params
                )

                out = nengo.Node(size_in=dim)

                if n_layers == 1:

                    conn_in = nengo.Connection(
                        inp, hidden_ens.neurons, synapse=None,
                        **localization_inp_params
                    )
                    conn_out = nengo.Connection(
                        hidden_ens.neurons, out, synapse=None,
                        # function=lambda x: np.zeros((args.dim,)),
                        **localization_out_params
                    )
                elif n_layers == 2:

                    localization_mid_params = localization_params[3]
                    localization_ens_two_params = localization_params[4]

                    hidden_ens_two = nengo.Ensemble(
                        n_neurons=hidden_size,
                        dimensions=1,
                        neuron_type=neuron_type,
                        **localization_ens_two_params
                    )

                    conn_in = nengo.Connection(
                        inp, hidden_ens.neurons, synapse=None,
                        **localization_inp_params
                    )
                    conn_mid = nengo.Connection(
                        hidden_ens.neurons, hidden_ens_two.neurons, synapse=None,
                        **localization_mid_params
                    )
                    conn_out = nengo.Connection(
                        hidden_ens_two.neurons, out, synapse=None,
                        **localization_out_params
                    )

            else:
                hidden_ens = nengo.Ensemble(
                    n_neurons=hidden_size,
                    dimensions=36 * 4 + maze_id_dim,
                    neuron_type=neuron_type
                )

                out = nengo.Node(size_in=dim)

                if n_layers == 2:
                    hidden_ens_two = nengo.Ensemble(
                        n_neurons=hidden_size,
                        dimensions=1,
                        # dimensions=36*4 + args.maze_id_dim,
                        neuron_type=neuron_type
                    )

                    conn_in = nengo.Connection(
                        inp, hidden_ens.neurons, synapse=None
                    )

                    conn_mid = nengo.Connection(
                        hidden_ens.neurons, hidden_ens_two.neurons, synapse=None
                    )

                    conn_out = nengo.Connection(
                        hidden_ens_two.neurons, out, synapse=None,
                    )

                else:

                    conn_in = nengo.Connection(
                        inp, hidden_ens.neurons, synapse=None
                    )
                    conn_out = nengo.Connection(
                        hidden_ens.neurons, out, synapse=None,
                    )

                # conn_in = nengo.Connection(inp, hidden_ens, synapse=None)
                # conn_out = nengo.Connection(hidden_ens, out, synapse=None, function=lambda x: np.zeros((args.dim,)))

            out_p = nengo.Probe(out, label="out_p")
            self.out_p_filt = nengo.Probe(out, synapse=0.1, label="out_p_filt")

        self.sim = nengo_dl.Simulator(self.net, minibatch_size=1)
        # self.sim.load_params(param_file)
        self.sim.compile(loss={self.out_p_filt: mse_loss})
        self.n_steps = n_steps

    def predict(self, inputs):
        tiled_input = np.tile(inputs[:, None, :], (1, self.n_steps, 1))

        pred_eval = self.sim.predict(tiled_input)
        return pred_eval[self.out_p_filt][:, 10:, :].mean(axis=1)
