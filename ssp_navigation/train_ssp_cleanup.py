# Given a 'noisy' spatial semantic pointer that is the result of a query, clean it up to the best pure SSP
# One use of this function is a component in sliding single objects in a memory
import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from ssp_navigation.utils.models import FeedForward
from ssp_navigation.utils.datasets import GenericDataset
import argparse
import json
from datetime import datetime
import os.path as osp
import os
import nengo
# from spatial_semantic_pointers.utils import encode_point, make_good_unitary
from ssp_navigation.utils.encodings import get_encoding_function


def generate_cleanup_dataset(
        # x_axis_sp,
        # y_axis_sp,
        encoding_func,
        n_samples,
        dim,
        n_items,
        item_set=None,
        allow_duplicate_items=False,
        limits=(-1, 1, -1, 1),
        seed=13,
        normalize_memory=True):
    """
    TODO: fix this description
    Create a dataset of memories that contain items bound to coordinates

    :param n_samples: number of memories to create
    :param dim: dimensionality of the memories
    :param n_items: number of items in each memory
    :param item_set: optional list of possible item vectors. If not supplied they will be generated randomly
    :param allow_duplicate_items: if an item set is given, this will allow the same item to be at multiple places
    # :param x_axis_sp: optional x_axis semantic pointer. If not supplied, will be generated as a unitary vector
    # :param y_axis_sp: optional y_axis semantic pointer. If not supplied, will be generated as a unitary vector
    :param encoding_func: function for generating the encoding
    :param limits: limits of the 2D space (x_low, x_high, y_low, y_high)
    :param seed: random seed for the memories and axis vectors if not supplied
    :param normalize_memory: if true, call normalize() on the memory semantic pointer after construction
    :return: memory, items, coords, x_axis_sp, y_axis_sp, z_axis_sp
    """
    # This seed must match the one that was used to generate the model
    np.random.seed(seed)

    # Memory containing n_items of items bound to coordinates
    memory = np.zeros((n_samples, dim))

    # SP for the item of interest
    items = np.zeros((n_samples, n_items, dim))

    # Coordinate for the item of interest
    coords = np.zeros((n_samples * n_items, 2))

    # Clean ground truth SSP
    clean_ssps = np.zeros((n_samples * n_items, dim))

    # Noisy output SSP
    noisy_ssps = np.zeros((n_samples * n_items, dim))

    for i in range(n_samples):
        memory_sp = nengo.spa.SemanticPointer(data=np.zeros((dim,)))

        # If a set of items is given, choose a subset to use now
        if item_set is not None:
            items_used = np.random.choice(item_set, size=n_items, replace=allow_duplicate_items)
        else:
            items_used = None

        for j in range(n_items):

            x = np.random.uniform(low=limits[0], high=limits[1])
            y = np.random.uniform(low=limits[2], high=limits[3])

            # pos = encode_point(x, y, x_axis_sp=x_axis_sp, y_axis_sp=y_axis_sp)
            pos = nengo.spa.SemanticPointer(data=encoding_func(x, y))

            if items_used is None:
                item = nengo.spa.SemanticPointer(dim)
            else:
                item = nengo.spa.SemanticPointer(data=items_used[j])

            items[i, j, :] = item.v
            coords[i * n_items + j, 0] = x
            coords[i * n_items + j, 1] = y
            clean_ssps[i * n_items + j, :] = pos.v
            memory_sp += (pos * item)

        if normalize_memory:
            memory_sp.normalize()

        memory[i, :] = memory_sp.v

        # Query for each item to get the noisy SSPs
        for j in range(n_items):
            noisy_ssps[i * n_items + j, :] = (memory_sp * ~nengo.spa.SemanticPointer(data=items[i, j, :])).v

    return clean_ssps, noisy_ssps, coords
    # return memory, items, coords, x_axis_sp, y_axis_sp


def run_evaluation(testloader, model, writer, epoch, mse_criterion, cosine_criterion, name='test'):
    with torch.no_grad():

        # Everything is in one batch, so this loop will only happen once
        for i, data in enumerate(testloader):

            noisy, clean = data

            outputs = model(noisy)

            mse_loss = mse_criterion(outputs, clean)
            # Modified to use CosineEmbeddingLoss
            cosine_loss = cosine_criterion(outputs, clean, torch.ones(len(outputs)))

            loss = cosine_loss + mse_loss

            print(cosine_loss.data.item())

        if writer is not None:
            # TODO: get a visualization of the performance
            writer.add_scalar('{}_cosine_loss'.format(name), cosine_loss.data.item(), epoch)
            writer.add_scalar('{}_mse_loss'.format(name), mse_loss.data.item(), epoch)
            writer.add_scalar('{}_combined_loss'.format(name), loss.data.item(), epoch)


def main():
    parser = argparse.ArgumentParser('Train a network to clean up a noisy spatial semantic pointer')

    parser.add_argument('--loss-function', type=str, default='cosine',
                        choices=['cosine', 'mse', 'combined', 'scaled'])
    parser.add_argument('--noise-type', type=str, default='memory', choices=['memory', 'gaussian', 'both'])
    parser.add_argument('--sigma', type=float, default=0.005, help='sigma on the gaussian noise if noise-type==gaussian')
    parser.add_argument('--spatial-encoding', type=str, default='sub-toroid-ssp',
                        choices=[
                            'ssp', 'hex-ssp', 'random', '2d', '2d-normalized', 'one-hot', 'hex-trig',
                            'sub-toroid-ssp', 'var-sub-toroid-ssp', 'proj-ssp', 'orth-proj-ssp',
                            'trig', 'random-trig', 'random-proj', 'learned', 'frozen-learned',
                            'pc-gauss', 'pc-dog', 'tile-coding'
                        ],
                        help='coordinate encoding')
    parser.add_argument('--hex-freq-coef', type=float, default=2.5,
                        help='constant to scale frequencies by for hex-trig')
    parser.add_argument('--pc-gauss-sigma', type=float, default=0.25, help='sigma for the gaussians')
    parser.add_argument('--pc-diff-sigma', type=float, default=0.5, help='sigma for subtracted gaussian in DoG')
    parser.add_argument('--n-tiles', type=int, default=8, help='number of layers for tile coding')
    parser.add_argument('--n-bins', type=int, default=0, help='number of bins for tile coding')
    parser.add_argument('--ssp-scaling', type=float, default=1.0)
    parser.add_argument('--phi', type=float, default=0.5, help='phi as a fraction of pi for orth-proj-ssp')
    parser.add_argument('--n-proj', type=int, default=3, help='projection dimension for sub toroids')
    parser.add_argument('--scale-ratio', type=float, default=0, help='ratio between sub toroid scales')

    parser.add_argument('--val-period', type=int, default=10, help='number of epochs before a test/validation set run')
    parser.add_argument('--train-fraction', type=float, default=.5, help='proportion of the dataset to use for training')
    parser.add_argument('--n-samples', type=int, default=10000,
                        help='Number of memories to generate. Total samples will be n-samples * n-items')
    parser.add_argument('--n-items', type=int, default=12, help='number of items in memory. Proxy for noisiness')
    parser.add_argument('--dim', type=int, default=256, help='Dimensionality of the semantic pointers')
    parser.add_argument('--hidden-size', type=int, default=512, help='Hidden size of the cleanup network')
    # parser.add_argument('--limits', type=str, default="0,13,0,13", help='The limits of the space')
    parser.add_argument('--limits', type=str, default="-5,5,-5,5", help='The limits of the space')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--seed', type=int, default=13)
    parser.add_argument('--logdir', type=str, default='trained_models/ssp_cleanup',
                        help='Directory for saved model and tensorboard log')
    parser.add_argument('--load-model', type=str, default='', help='Optional model to continue training from')
    # parser.add_argument('--name', type=str, default='',
    #                     help='Name of output folder within logdir. Will use current date and time if blank')
    parser.add_argument('--weight-histogram', action='store_true', help='Save histograms of the weights if set')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['rmsprop', 'adam', 'sgd'])

    parser.add_argument('--dataset-seed', type=int, default=14)

    args = parser.parse_args()

    args.limits = tuple(float(v) for v in args.limits.split(','))

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset_name = 'data/ssp_cleanup_dataset_dim{}_seed{}_items{}_samples{}.npz'.format(
        args.dim, args.seed, args.n_items, args.n_samples
    )

    if not os.path.exists('data'):
        os.makedirs('data')

    rng = np.random.RandomState(seed=args.seed)
    # x_axis_sp = make_good_unitary(args.dim, rng=rng)
    # y_axis_sp = make_good_unitary(args.dim, rng=rng)


    limit_low = args.limits[0]
    limit_high = args.limits[1]

    encoding_func, repr_dim = get_encoding_function(args, limit_low=limit_low, limit_high=limit_high)

    if os.path.exists(dataset_name):
        print("Loading dataset")
        data = np.load(dataset_name)
        clean_ssps = data['clean_ssps']
        noisy_ssps = data['noisy_ssps']
    else:
        print("Generating SSP cleanup dataset")
        # TODO: save the dataset the first time it is created, so it can be loaded the next time
        clean_ssps, noisy_ssps, coords = generate_cleanup_dataset(
            # x_axis_sp=x_axis_sp,
            # y_axis_sp=y_axis_sp,
            encoding_func=encoding_func,
            n_samples=args.n_samples,
            dim=args.dim,
            n_items=args.n_items,
            limits=args.limits,
            seed=args.dataset_seed,
        )
        print("Dataset generation complete. Saving dataset")
        np.savez(
            dataset_name,
            clean_ssps=clean_ssps,
            noisy_ssps=noisy_ssps,
            coords=coords,
            # x_axis_vec=x_axis_sp.v,
            # y_axis_vec=x_axis_sp.v,
        )

    # Add gaussian noise if required
    if args.noise_type == 'gaussian':
        noisy_ssps = clean_ssps + np.random.normal(loc=0, scale=args.sigma, size=noisy_ssps.shape)
    elif args.noise_type == 'both':
        noisy_ssps += np.random.normal(loc=0, scale=args.sigma, size=noisy_ssps.shape)

    n_samples = clean_ssps.shape[0]
    n_train = int(args.train_fraction * n_samples)
    n_test = n_samples - n_train
    assert(n_train > 0 and n_test > 0)
    train_clean = clean_ssps[:n_train, :]
    train_noisy = noisy_ssps[:n_train, :]
    test_clean = clean_ssps[n_train:, :]
    test_noisy = noisy_ssps[n_train:, :]

    dataset_train = GenericDataset(inputs=train_noisy, outputs=train_clean)
    dataset_test = GenericDataset(inputs=test_noisy, outputs=test_clean)

    trainloader = torch.utils.data.DataLoader(
        dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=0,
    )

    # For testing just do everything in one giant batch
    testloader = torch.utils.data.DataLoader(
        dataset_test, batch_size=len(dataset_test), shuffle=False, num_workers=0,
    )

    model = FeedForward(input_size=args.dim, hidden_size=args.hidden_size, output_size=args.dim)

    # Open a tensorboard writer if a logging directory is given
    if args.logdir != '':
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        save_dir = osp.join(args.logdir, current_time)
        writer = SummaryWriter(log_dir=save_dir)
        if args.weight_histogram:
            # Log the initial parameters
            for name, param in model.named_parameters():
                writer.add_histogram('parameters/' + name, param.clone().cpu().data.numpy(), 0)

    mse_criterion = nn.MSELoss()
    cosine_criterion = nn.CosineEmbeddingLoss()

    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for e in range(args.epochs):
        print('Epoch: {0}'.format(e + 1))

        if e % args.val_period == 0:
            run_evaluation(
                testloader, model, writer, e,
                mse_criterion, cosine_criterion,
                name='test'
            )

        avg_mse_loss = 0
        avg_cosine_loss = 0
        avg_combined_loss = 0
        n_batches = 0
        for i, data in enumerate(trainloader):

            noisy, clean = data

            if noisy.size()[0] != args.batch_size:
                continue  # Drop data, not enough for a batch
            optimizer.zero_grad()

            outputs = model(noisy)

            mse_loss = mse_criterion(outputs, clean)
            # Modified to use CosineEmbeddingLoss
            cosine_loss = cosine_criterion(outputs, clean, torch.ones(args.batch_size))
            # print(loss.data.item())

            loss = cosine_loss + mse_loss

            if args.loss_function == 'cosine':
                cosine_loss.backward()
            elif args.loss_function == 'mse':
                mse_loss.backward()
            elif args.loss_function == 'combined':
                loss.backward()

            avg_cosine_loss += cosine_loss.data.item()
            avg_mse_loss += mse_loss.data.item()
            avg_combined_loss += loss.data.item()
            n_batches += 1

            optimizer.step()

        print(avg_cosine_loss / n_batches)

        if args.logdir != '':
            if n_batches > 0:
                avg_cosine_loss /= n_batches
                writer.add_scalar('avg_cosine_loss', avg_cosine_loss, e + 1)
                writer.add_scalar('avg_mse_loss', avg_mse_loss, e + 1)
                writer.add_scalar('avg_combined_loss', avg_mse_loss, e + 1)

            if args.weight_histogram and (e + 1) % 10 == 0:
                for name, param in model.named_parameters():
                    writer.add_histogram('parameters/' + name, param.clone().cpu().data.numpy(), e + 1)

    print("Testing")
    run_evaluation(testloader, model, writer, e, mse_criterion, cosine_criterion, name='test')
    # with torch.no_grad():
    #
    #     # Everything is in one batch, so this loop will only happen once
    #     for i, data in enumerate(testloader):
    #
    #         noisy, clean = data
    #
    #         outputs = model(noisy)
    #
    #         mse_loss = mse_criterion(outputs, clean)
    #         # Modified to use CosineEmbeddingLoss
    #         cosine_loss = cosine_criterion(outputs, clean, torch.ones(len(dataset_test)))
    #
    #         print(cosine_loss.data.item())
    #
    #     if args.logdir != '':
    #         # TODO: get a visualization of the performance
    #         writer.add_scalar('test_cosine_loss', cosine_loss.data.item())
    #         writer.add_scalar('test_mse_loss', mse_loss.data.item())

    # Close tensorboard writer
    if args.logdir != '':
        writer.close()

        torch.save(model.state_dict(), osp.join(save_dir, 'model.pt'))

        params = vars(args)
        # # Additionally save the axis vectors used
        # params['x_axis_vec'] = list(x_axis_sp.v)
        # params['y_axis_vec'] = list(y_axis_sp.v)
        with open(osp.join(save_dir, "params.json"), "w") as f:
            json.dump(params, f)


if __name__ == '__main__':
    main()
