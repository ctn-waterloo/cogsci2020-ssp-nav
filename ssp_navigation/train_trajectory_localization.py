import matplotlib
import os
# allow code to work on machines without a display or in a screen session
display = os.environ.get('DISPLAY')
if display is None or 'localhost' in display:
    matplotlib.use('agg')

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from tensorboardX import SummaryWriter
import json
from spatial_semantic_pointers.utils import get_heatmap_vectors, ssp_to_loc, ssp_to_loc_v
from ssp_navigation.utils.training import TrajectoryValidationSet, localization_train_test_loaders, LocalizationModel, pc_to_loc_v


parser = argparse.ArgumentParser('Run 2D supervised localization experiment with trajectories using pytorch')

parser.add_argument('--seed', type=int, default=13)
parser.add_argument('--n-epochs', type=int, default=250)
parser.add_argument('--n-samples', type=int, default=5000)
parser.add_argument('--encoding', type=str, default='ssp', choices=['ssp', '2d'])
parser.add_argument('--eval-period', type=int, default=50)
parser.add_argument('--load-saved-model', type=str, default='', help='Saved model to load from')
parser.add_argument('--loss-function', type=str, default='cosine', choices=['cosine', 'mse'])
parser.add_argument('--trajectory-length', type=int, default=100)
parser.add_argument('--batch-size', type=int, default=10)
parser.add_argument('--learning-rate', type=float, default=1e-5, help='Step size multiplier in the RMSProp algorithm')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum parameter of the RMSProp algorithm')

parser.add_argument('--lstm-hidden-size', type=int, default=128)
parser.add_argument('--hidden-size', type=int, default=512)
parser.add_argument('--grad-clip-thresh', type=float, default=1e-5, help='Gradient clipping threshold')
parser.add_argument('--dropout', type=float, default=0.5, help='Probability of dropout')

parser.add_argument('--dataset-dir', type=str,
                    default='datasets/mixed_style_20mazes_50goals_64res_13size_13seed/36sensors_360fov/500t_250s/')
parser.add_argument('--variant-subfolder', type=str, default='',
                    help='Optional custom subfolder')
# parser.add_argument('--logdir', type=str, default='traj_network',
#                     help='Directory for saved model and tensorboard log, within dataset-dir')

args = parser.parse_args()

dataset_file = os.path.join(args.dataset_dir, 'trajectory_dataset.npz')

variant_folder = '{}_rec{}_lin{}_trajlen{}_samples{}_dropout{}'.format(
    args.encoding, args.lstm_hidden_size, args.hidden_size, args.trajectory_length, args.n_samples, args.dropout,
)

if args.variant_subfolder != '':
    variant_folder = os.path.join(variant_folder, args.variant_subfolder)

logdir = os.path.join(args.dataset_dir, 'trajectory_network', variant_folder)

torch.manual_seed(args.seed)
np.random.seed(args.seed)

use_cosine_loss = False
if args.loss_function == 'cosine':
    use_cosine_loss = True

if use_cosine_loss:
    print("Using Cosine Loss")
else:
    print("Using MSE Loss")

current_time = datetime.now().strftime('%b%d_%H-%M-%S')
save_dir = os.path.join(logdir, current_time)
writer = SummaryWriter(log_dir=save_dir)

data = np.load(dataset_file)

x_axis_vec = data['x_axis_vec']
y_axis_vec = data['y_axis_vec']
ssp_scaling = data['ssp_scaling']
ssp_offset = data['ssp_offset']

# shape of coarse maps is (n_maps, env_size, env_size)
coarse_maps = data['coarse_mazes']
n_maps = coarse_maps.shape[0]
env_size = coarse_maps.shape[1]

# shape of dist_sensors is (n_maps, n_trajectories, n_steps, n_sensors)
n_sensors = data['dist_sensors'].shape[3]

if args.encoding == 'ssp':
    # shape of ssps is (n_maps, n_trajectories, n_steps, dim)
    dim = data['ssps'].shape[3]
elif args.encoding == '2d':
    dim = 2
else:
    raise NotImplementedError


limit_low = -ssp_offset * ssp_scaling
limit_high = (env_size - ssp_offset) * ssp_scaling
res = 256

#TEMP FIXME DEBUGGING
print("limit_low", limit_low)
print("limit_high", limit_high)
# limit_low = -5
# limit_high = 5

xs = np.linspace(limit_low, limit_high, res)
ys = np.linspace(limit_low, limit_high, res)

# Used for visualization of test set performance using pos = ssp_to_loc(sp, heatmap_vectors, xs, ys)
heatmap_vectors = get_heatmap_vectors(xs, ys, x_axis_vec, y_axis_vec)

n_samples = args.n_samples
rollout_length = args.trajectory_length
batch_size = args.batch_size
n_epochs = args.n_epochs

# Input is x and y velocity plus the distance sensor measurements, plus map ID
model = LocalizationModel(
    input_size=2 + n_sensors + n_maps,
    unroll_length=rollout_length,
    lstm_hidden_size=args.lstm_hidden_size,
    linear_hidden_size=args.hidden_size,
    dropout_p=args.dropout,
    sp_dim=dim
)

if args.load_saved_model:
    model.load_state_dict(torch.load(args.load_saved_model), strict=False)

cosine_criterion = nn.CosineEmbeddingLoss()
mse_criterion = nn.MSELoss()

optimizer = optim.RMSprop(model.parameters(), lr=args.learning_rate, momentum=args.momentum)

trainloader, testloader = localization_train_test_loaders(
    data,
    n_train_samples=n_samples,
    n_test_samples=n_samples,
    rollout_length=rollout_length,
    batch_size=batch_size,
    encoding=args.encoding,
)

validation_set = TrajectoryValidationSet(
    dataloader=testloader,
    heatmap_vectors=heatmap_vectors,
    xs=xs,
    ys=ys,
    ssp_scaling=ssp_scaling,
    spatial_encoding=args.encoding,
)

params = vars(args)
with open(os.path.join(save_dir, "params.json"), "w") as f:
    json.dump(params, f)

print("Training")
for epoch in range(n_epochs):
    print("Epoch {} of {}".format(epoch + 1, n_epochs))

    # Every 'eval_period' epochs, create a test loss and image
    if epoch % args.eval_period == 0:

        print("Evaluating")
        validation_set.run_eval(
            model=model,
            writer=writer,
            epoch=epoch,
        )

        if epoch > 0:
            # Save a copy of the model at this stage
            if args.encoding == 'ssp':
                torch.save(
                    model.state_dict(),
                    os.path.join(save_dir, 'ssp_trajectory_localization_model_epoch_{}.pt'.format(epoch))
                )
            elif args.encoding == '2d':
                torch.save(
                    model.state_dict(),
                    os.path.join(save_dir, '2d_trajectory_localization_model_epoch_{}.pt'.format(epoch))
                )

    avg_mse_loss = 0
    avg_cosine_loss = 0
    n_batches = 0
    for i, data in enumerate(trainloader):
        # velocity_inputs, sensor_inputs, ssp_inputs, ssp_outputs = data
        combined_inputs, ssp_inputs, ssp_outputs = data

        if ssp_inputs.size()[0] != batch_size:
            continue  # Drop data, not enough for a batch
        optimizer.zero_grad()
        # model.zero_grad()

        ssp_pred = model(combined_inputs, ssp_inputs)

        # NOTE: need to permute axes of the targets here because the output is
        #       (sequence length, batch, units) instead of (batch, sequence_length, units)
        #       could also permute the outputs instead
        # NOTE: for cosine loss the input needs to be flattened first
        cosine_loss = cosine_criterion(
            ssp_pred.reshape(ssp_pred.shape[0] * ssp_pred.shape[1], ssp_pred.shape[2]),
            ssp_outputs.permute(1, 0, 2).reshape(ssp_pred.shape[0] * ssp_pred.shape[1], ssp_pred.shape[2]),
            torch.ones(ssp_pred.shape[0] * ssp_pred.shape[1])
        )
        mse_loss = mse_criterion(ssp_pred, ssp_outputs.permute(1, 0, 2))

        if use_cosine_loss:
            cosine_loss.backward()
        else:
            mse_loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_thresh)

        optimizer.step()

        avg_mse_loss += mse_loss.data.item()
        avg_cosine_loss += cosine_loss.data.item()
        n_batches += 1

    avg_mse_loss /= n_batches
    avg_cosine_loss /= n_batches
    print("mse loss:", avg_mse_loss)
    print("cosine loss:", avg_cosine_loss)
    writer.add_scalar('avg_mse_loss', avg_mse_loss, epoch + 1)
    writer.add_scalar('avg_cosine_loss', avg_cosine_loss, epoch + 1)


print("Testing")
validation_set.run_eval(
    model=model,
    writer=writer,
    epoch=n_epochs,
)

if args.encoding == 'ssp':
    torch.save(model.state_dict(), os.path.join(save_dir, 'ssp_trajectory_localization_model.pt'))
elif args.encoding == '2d':
    torch.save(model.state_dict(), os.path.join(save_dir, '2d_trajectory_localization_model.pt'))
