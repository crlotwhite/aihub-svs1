import argparse
import torch
import preference
import os
import tqdm
import pickle
from data import Dataset
from torch.utils.data import DataLoader
from glow import WaveGlow, WaveGlowLoss
import time
from tensorboardX import SummaryWriter
import numpy as np
from preference import save, load
from stft import MelSpectrogram
import logging
import sys


def save_checkpoint(model, optimizer, filepath, config: preference.Config):
    model_for_saving = WaveGlow(
        n_mel_channels=config.num_mel,
        n_flows=config.n_flow,
        n_group=config.n_group,
        n_early_every=config.n_early_every,
        n_early_size=config.n_early_size,
        wn_n_layer=config.wn_n_layer,
        wn_n_channel=config.wn_n_channel,
        wn_kernel_size=config.wn_kernel_size,
        win_size=config.win_length,
        hop_size=config.hop_length
    ).cpu()
    model_for_saving.load_state_dict(model.state_dict())
    torch.save({'model': model_for_saving,
                'optimizer': optimizer.state_dict()}, filepath)


def setup(config: preference.Config):
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(config.gpu_index)

    if torch.cuda.is_available() and config.gpu_index >= 0:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    os.makedirs(config.checkpoint_dir, exist_ok=True)
    print("checkpoints directory : ", config.checkpoint_dir)

    os.makedirs(config.log_dir, exist_ok=True)
    print('log directory : ', config.log_dir)

    path = os.path.join(config.log_dir, 'train')
    os.makedirs(path, exist_ok=True)
    print('train log directory : ', path)
    config.train_log = path

    path = os.path.join(config.log_dir, 'valid')
    os.makedirs(path, exist_ok=True)
    print('valid log directory : ', path)
    config.valid_log = path

    if config.use_log is True:
        sw_train = SummaryWriter(config.train_log)
        sw_valid = SummaryWriter(config.valid_log)
    else:
        sw_train = None
        sw_valid = None

    print('{s:{c}^{n}}\n'.format(s='complete: setup step', n=50, c='-'))

    return device, sw_train, sw_valid


def build_dataloader(files, config: preference.Config, desc):
    audio_list = []
    for file in tqdm.tqdm(files, desc=desc):
        with open(file, 'rb') as f:
            data = pickle.load(f)
            audio_list.append(np.array(data['audio']))
    dataset = Dataset(audio_list, config.audio_segment_size)
    dataloader = DataLoader(
        dataset,
        shuffle=config.shuffle,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=config.drop_last)
    return dataloader


def build_dataset(config: preference.Config):
    train_loader = build_dataloader(config.train_file, config, 'load train dataset')

    valid_loader = None
    if config.use_valid and len(config.valid_file) > 0:
        valid_loader = build_dataloader(config.valid_file, config, 'load valid dataset')

    print('shuffle: ', config.shuffle)
    print('batch size: ', config.batch_size)
    print('workers: ', config.num_workers)
    print('pin memory: ', config.pin_memory)
    print('drop last: ', config.drop_last)

    print('{s:{c}^{n}}\n'.format(s='complete: dataset step', n=50, c='-'))

    return train_loader, valid_loader


def train_model(
        dataloader: DataLoader,
        valid_loader: DataLoader,
        model: torch.nn.Module,
        loss: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        sw_train,
        sw_valid,
        config: preference.Config,
        logger,
        start_epoch: int = 0
):

    mel_spectrogram = MelSpectrogram(config.sampling_rate, config.filter_length, config.num_mel, config.win_length, config.hop_length, config.mel_min, config.mel_max)

    mel_spectrogram = mel_spectrogram.to(device)

    model.train()

    print('max epoch: ', config.training_epoch)

    for epoch in range(start_epoch, config.training_epoch):
        print('========================================')
        print('epoch: ', epoch + 1)
        print('========================================')
        start = time.time()
        count = 0
        loss_total = 0
        for audio in tqdm.tqdm(dataloader):
            audio = audio.to(device, non_blocking=True)

            mel = mel_spectrogram(audio)

            model.zero_grad()

            outputs = model((mel, audio))

            loss_value = loss(outputs)
            reduced_loss = loss_value.item()
            loss_value.backward()

            optimizer.step()

            count += config.batch_size
            loss_total += reduced_loss
        train_loss = loss_total / count

        if valid_loader is not None:
            print('========================================')
            print('Validation')
            print('========================================')
            model.eval()
            count = 0
            loss_total = 0
            with torch.no_grad():
                for audio in tqdm.tqdm(valid_loader):
                    audio = audio.to(device, non_blocking=True)
                    mel = mel_spectrogram(audio)
                    outputs = model((mel, audio))
                    loss_value = loss(outputs)
                    reduced_loss = loss_value.item()
                    count += config.batch_size
                    loss_total += reduced_loss
                valid_loss = loss_total / count
            model.train()

            print('epoch: {}, train loss: {:4.3f}, valid loss: {:4.3f}, s/e: {:4.3f}'.format(
                epoch + 1, train_loss, valid_loss, time.time() - start
            ))
            logger.info('epoch: {}, train loss: {:4.3f}, valid loss: {:4.3f}, s/e: {:4.3f}'.format(
                epoch + 1, train_loss, valid_loss, time.time() - start
            ))
        else:
            print('epoch: {}, train loss: {:4.3f}, s/e: {:4.3f}'.format(
                epoch + 1, train_loss, time.time() - start
            ))
        config.last_epoch = epoch

        # log
        if config.use_log:
            sw_train.add_scalar('loss', train_loss, epoch + 1)
            if valid_loader is not None:
                sw_valid.add_scalar('loss', valid_loss, epoch + 1)

        # check point
        if (epoch + 1) % config.checkpoint_interval == 0:
            path = os.path.join(config.checkpoint_dir, 'waveglow_{}.tar'.format(epoch))
            save_checkpoint(model, optimizer, path, config)
            config.last_checkpoint_file = path
            print('save checkpoint : ', path)

            path = os.path.join(config.checkpoint_dir, 'config.json')
            preference.save(config, path)
            print('save config : ', path)


def build_fastspeech(config: preference.Config):
    model = WaveGlow(
        n_mel_channels=config.num_mel,
        n_flows=config.n_flow,
        n_group=config.n_group,
        n_early_every=config.n_early_every,
        n_early_size=config.n_early_size,
        wn_n_layer=config.wn_n_layer,
        wn_n_channel=config.wn_n_channel,
        wn_kernel_size=config.wn_kernel_size,
        win_size=config.win_length,
        hop_size=config.hop_length
    )

    return model


def build_model(config: preference.Config, device: torch.device):
    model = build_fastspeech(config)
    print('--Model')
    print('mel channel: ', config.num_mel)
    print('flow: ', config.n_flow)
    print('group: ', config.n_group)
    print('early every: ', config.n_early_every)
    print('early size: ', config.n_early_size)
    print('wn layer: ', config.wn_n_layer)
    print('wn channel: ', config.wn_n_channel)
    print('wn kernel size: ', config.wn_kernel_size)
    print('win size', config.win_length)
    print('hop size', config.hop_length)

    if len(config.model_path) > 0:
        checkpoint_dict = torch.load(config.model_path)
        model.load_state_dict(checkpoint_dict['model'].state_dict())
        model = model.to(device)
    else:
        model = model.to(device)

    loss = WaveGlowLoss(config.sigma)
    print('--Loss')
    print('sigma: ', config.sigma)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    print('--Adam optimizer')
    print('learning rate: ', config.learning_rate)

    print('{s:{c}^{n}}\n'.format(s='complete: model step', n=50, c='-'))

    return model, loss, optimizer


if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s', '%Y-%m-%d %H:%M:%S %Z%z')

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler('log.txt')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    cmd = ' '.join(sys.argv)
    logger.info('python ' + cmd)

    print('Initializing Training Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='JSON file for configuration')
    parser.add_argument('--epoch', '-e', type=int, help='num of train loop')
    parser.add_argument('--batch', '-b', type=int, help='batch size for training')
    parser.add_argument('--rate', type=float, help='learning rate for training')
    parser.add_argument('--checkpoint', '-c', help='directory for save checkpoint file')
    parser.add_argument('--interval', '-t', type=int, help='check point save interval time(sec)')
    parser.add_argument('--model', '-m', help='model path for fine tuning')
    parser.add_argument('--gpu', '-g', type=int, help='gpu device index')
    parser.add_argument('--log', help='log directory for tensorboard')
    parser.add_argument('--workers', '-w', type=int, help='num of dataloader workers')
    parser.add_argument('--shuffle', '-s', type=bool, help='dataset shuffle use')
    parser.add_argument('--pin', '-p', type=bool, help='dataloader use pin memory')
    parser.add_argument('--drop', '-d', type=bool, help='dataloader use drop last')
    parser.add_argument('--use_valid', '-v', action='store_true', help='use validation dataset')
    parser.add_argument('--use_log', '-l', action='store_true', help='use logger')

    # argument parsing
    args = parser.parse_args()
    training_epoch = args.epoch
    batch_size = args.batch
    learning_rate = args.rate
    checkpoint_dir = args.checkpoint
    interval = args.interval
    model_path = args.model
    gpu = args.gpu
    log_dir = args.log
    workers = args.workers
    shuffle = args.shuffle
    pin = args.pin
    drop = args.drop
    use_valid = args.use_valid
    use_log = args.use_log

    # configuration setting
    config = load(args.config)
    print('config : ', args.config)
    if training_epoch is not None:
        config.training_epoch = training_epoch
    if batch_size is not None:
        config.batch_size = batch_size
    if learning_rate is not None:
        config.learning_rate = learning_rate
    if checkpoint_dir is not None:
        config.checkpoint_dir = checkpoint_dir
    if interval is not None:
        config.checkpoint_interval = interval
    if model_path is not None:
        config.model_path = model_path
    if gpu is not None:
        config.gpu_index = gpu
    if log_dir is not None:
        config.log_dir = log_dir
    if workers is not None:
        config.num_workers = workers
    if shuffle is not None:
        config.shuffle = shuffle
    if pin is not None:
        config.pin_memory = pin
    if drop is not None:
        config.drop_last = drop
    config.use_valid = use_valid
    config.use_log = use_log

    # setup
    device, sw_train, sw_valid = setup(config)

    # dataset
    train_loader, valid_loader = build_dataset(config)

    # model
    model, loss, optimizer = build_model(config, device)
    
    # load checkpoint
    start_epoch = 0
    if config.get('last_checkpoint_file', None) is not None:
        model = torch.load(config['last_checkpoint_file'])
        start_epoch = config['last_epoch'] + 1

    print('========================================')
    print('start training')
    print('========================================')
    # train
    train_model(train_loader, valid_loader, model, loss, optimizer, sw_train, sw_valid, config, logger, start_epoch=start_epoch)
