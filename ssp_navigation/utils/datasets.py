import numpy as np
import torch
import torch.utils.data as data


class PathDataset(data.Dataset):

    def __init__(self, ssp_inputs, direction_outputs, coord_inputs):

        self.ssp_inputs = ssp_inputs.astype(np.float32)
        self.direction_outputs = direction_outputs.astype(np.float32)
        self.coord_inputs = coord_inputs.astype(np.float32)

    def __getitem__(self, index):
        return self.ssp_inputs[index], self.direction_outputs[index], self.coord_inputs[index]

    def __len__(self):
        return self.ssp_inputs.shape[0]


class MultiPathDataset(data.Dataset):

    def __init__(self, ssp_inputs, direction_outputs, coord_inputs, path_indices, n_paths):

        self.ssp_inputs = ssp_inputs.astype(np.float32)
        self.direction_outputs = direction_outputs.astype(np.float32)
        self.coord_inputs = coord_inputs.astype(np.float32)

        # convert to one-hot encoding
        self.path_choice = np.zeros((path_indices.shape[0], n_paths)).astype(np.float32)
        # Note that the transpose on path_indices is required for this to work
        self.path_choice[np.arange(path_indices.shape[0]), path_indices.T] = 1

        # check that the correct number of 1s are in the array
        assert(np.sum(self.path_choice) == path_indices.shape[0])

        # Concatenate input
        self.combined_input = np.hstack([self.ssp_inputs, self.path_choice])

    def __getitem__(self, index):
        return self.combined_input[index], self.direction_outputs[index], self.coord_inputs[index]

    def __len__(self):
        return self.ssp_inputs.shape[0]


class MazeDataset(data.Dataset):

    def __init__(self, maze_ssp, loc_ssps, goal_ssps, locs, goals, direction_outputs):

        if maze_ssp is not None:
            if len(maze_ssp.shape) > 1:
                self.maze_ssp = maze_ssp.astype(np.float32)
            else:
                # NOTE: this is currently assuming only one fixed maze
                self.maze_ssp = np.zeros((loc_ssps.shape[0], maze_ssp.shape[0])).astype(np.float32)
                for i in range(loc_ssps.shape[0]):
                    self.maze_ssp[i, :] = maze_ssp
        self.loc_ssps = loc_ssps.astype(np.float32)
        self.goal_ssps = goal_ssps.astype(np.float32)
        self.locs = locs.astype(np.float32)
        self.goals = goals.astype(np.float32)
        self.direction_outputs = direction_outputs.astype(np.float32)

        # Concatenate input
        if maze_ssp is None:
            self.combined_input = np.hstack([self.loc_ssps, self.goal_ssps])
        else:
            self.combined_input = np.hstack([self.maze_ssp, self.loc_ssps, self.goal_ssps])

    def __getitem__(self, index):
        # return self.maze_ssp, self.loc_ssps[index], self.goal_ssps[index], self.direction_outputs[index], self.locs[index], self.goals[index]
        return self.combined_input[index], self.direction_outputs[index], self.locs[index], self.goals[index]

    def __len__(self):
        return self.goals.shape[0]


class SingleMazeDataset(data.Dataset):
    """
    Just one maze, no ID vector
    """

    def __init__(self, loc_ssps, goal_ssps, locs, goals, direction_outputs):

        self.loc_ssps = loc_ssps.astype(np.float32)
        self.goal_ssps = goal_ssps.astype(np.float32)
        self.locs = locs.astype(np.float32)
        self.goals = goals.astype(np.float32)
        self.direction_outputs = direction_outputs.astype(np.float32)

        # Concatenate input
        self.combined_input = np.hstack([self.loc_ssps, self.goal_ssps])

    def __getitem__(self, index):

        return self.combined_input[index], self.direction_outputs[index], self.locs[index], self.goals[index]

    def __len__(self):
        return self.goals.shape[0]


class GenericDataset(data.Dataset):

    def __init__(self, inputs, outputs):

        self.inputs = inputs.astype(np.float32)
        self.outputs = outputs.astype(np.float32)

    def __getitem__(self, index):
        return self.inputs[index], self.outputs[index]

    def __len__(self):
        return self.inputs.shape[0]
