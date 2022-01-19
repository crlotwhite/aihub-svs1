import os
import random
import shutil
import tqdm
from scipy.io.wavfile import write, read
import torch
import argparse
import preference
import pickle
from tacotronstft import TacotronSTFT
from stft import MelSpectrogram

MAX_WAV_VALUE = 32768.0


def inference(model: torch.nn.Module, mel_spectogram: MelSpectrogram, files, cf:preference.Config, dir, device, desc):
    for file in tqdm.tqdm(files, desc=desc):
        basename = os.path.splitext(os.path.basename(file))[0]

        sr, audio = read(file)
        audio = audio / MAX_WAV_VALUE
        audio = torch.FloatTensor(audio).to(device)
        mel = mel_spectogram(audio.unsqueeze(0))
        with torch.no_grad():
            audio = model.infer(mel, sigma=cf.sigma)
        audio = audio * MAX_WAV_VALUE
        audio = audio.squeeze()
        audio = audio.cpu().numpy()
        audio = audio.astype('int16')
        filepath = os.path.join(dir, '{}.wav'.format(basename))
        write(filepath, config.sampling_rate, audio)


if __name__ == "__main__":
    print('Initializing Inference Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='JSON file for configuration')
    parser.add_argument('--dir', '-d', default='generated_file', help='directory for saving generated file')
    parser.add_argument('--file', '-f', help='checkpoint file to load generator model')
    parser.add_argument('--sample', '-s', type=int, help='sampling number for inference train/valid data')
    parser.add_argument('--remove', '-r', action='store_true', help='remove previous generated file')
    parser.add_argument('--gpu', '-g', type=int, help='gpu device index')

    args = parser.parse_args()

    # argument parsing
    config_file = args.config
    output_dir = args.dir
    checkpoint_file = args.file
    sampling_num = args.sample
    remove = args.remove
    gpu = args.gpu

    # configuration setting
    config = preference.load(config_file)
    if checkpoint_file is None:
        checkpoint_file = config.last_checkpoint_file
    if gpu is None:
        gpu = config.gpu_index

    # setup
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available() and gpu >= 0:
        device = torch.device('cuda:{:d}'.format(gpu))
    else:
        device = torch.device('cpu')

    os.makedirs(output_dir, exist_ok=True)
    if remove is True:
        if os.path.isdir(os.path.join(output_dir, 'train')) is True:
            shutil.rmtree(os.path.join(output_dir, 'train'))
        if os.path.isdir(os.path.join(output_dir, 'valid')) is True:
            shutil.rmtree(os.path.join(output_dir, 'valid'))
    os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'valid'), exist_ok=True)
    print('train directory : ', os.path.join(output_dir, 'train'))
    print('valid directory : ', os.path.join(output_dir, 'valid'))

    if sampling_num is not None:
        train_file = config.train_wave_file.copy()
        if len(train_file) > sampling_num:
            random.shuffle(train_file)
            train_file = train_file[:sampling_num]

        valid_file = config.valid_wave_file.copy()
        if len(valid_file) > sampling_num:
            random.shuffle(valid_file)
            valid_file = valid_file[:sampling_num]
    else:
        train_file = config.train_wave_file
        valid_file = config.valid_wave_file
    print('inference train file: ', len(train_file))
    print('inference valid file: ', len(valid_file))

    print('{s:{c}^{n}}\n'.format(s='complete: setup step', n=50, c='-'))

    # model
    waveglow = torch.load(checkpoint_file)['model']
    waveglow = waveglow.remove_weightnorm(waveglow)
    if torch.cuda.is_available() and gpu >= 0:
        waveglow.to(device).eval()
    else:
        waveglow.to(device).eval()

    print('{s:{c}^{n}}\n'.format(s='complete: model step', n=50, c='-'))

    mel_spectrogram = MelSpectrogram(config.sampling_rate, config.filter_length, config.num_mel, config.win_length,
                                     config.hop_length, config.mel_min, config.mel_max)
    mel_spectrogram = mel_spectrogram.to(device)

    inference(waveglow, mel_spectrogram, train_file, config, os.path.join(output_dir, 'train'), device, 'inference train dataset')
    inference(waveglow, mel_spectrogram, valid_file, config, os.path.join(output_dir, 'valid'), device,
              'inference valid dataset')
