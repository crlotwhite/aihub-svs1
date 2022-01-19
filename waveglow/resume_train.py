import argparse
import torch
import preference
from train import train_process, setup, dataset_process, model_process


def load_checkpoint(path, model, optimizer):
    checkpoint_dict = torch.load(path, map_location='cpu')
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
    model.load_state_dict(checkpoint_dict['model'].state_dict())

    return model, optimizer


if __name__ == '__main__':
    print('Initializing Resume Training Process..')

    parser = argparse.ArgumentParser()

    parser.add_argument('config', help='JSON file for configuration')
    parser.add_argument('--epoch', '-e', type=int, help='num of train loop')
    parser.add_argument('--batch', '-b', type=int, help='batch size for training')
    parser.add_argument('--dir', '-d', help='directory for save checkpoint file')
    parser.add_argument('--interval', '-t', type=int, help='check point save interval time(sec)')
    parser.add_argument('--gpu', '-g', type=int, help='gpu device index')
    parser.add_argument('--log', '-l', help='log directory for tensorboard')

    args = parser.parse_args()

    # argument parsing
    training_epoch = args.epoch
    batch_size = args.batch
    checkpoint_dir = args.dir
    interval = args.interval
    gpu = args.gpu
    log_dir = args.log

    # configuration setting
    config = preference.load(args.config)
    if training_epoch is not None:
        config.training_epoch = training_epoch
    if batch_size is not None:
        config.batch_size = batch_size
    if checkpoint_dir is not None:
        config.checkpoint_dir = checkpoint_dir
    if interval is not None:
        config.checkpoint_interval = interval
    if gpu is not None:
        config.gpu_index = gpu
    if log_dir is not None:
        config.log = log_dir

    # setup
    device, logger = setup(config)

    # dataset
    dataloader = dataset_process(config)

    # model
    model, loss, optimizer = model_process(config, device)

    # load model/optimizer
    model, optimizer = load_checkpoint(config.last_checkpoint_file, model, optimizer)
    print('load checkpoint file : ', config.last_checkpoint_file)

    print('{s:{c}^{n}}\n'.format(s='complete: load model step', n=50, c='-'))

    # train
    train_process(dataloader, model, loss, optimizer, logger, config, config.last_epoch)
