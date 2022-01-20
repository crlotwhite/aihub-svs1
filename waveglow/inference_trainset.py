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
        if os.path.isdir(output_dir) is True:
            shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    print('inference directory : ', output_dir, 'train')

    if sampling_num is not None:
        wave_file = config.wave_file.copy()
        if len(wave_file) > sampling_num:
            random.shuffle(wave_file)
            wave_file = wave_file[:sampling_num]
    else:
        wave_file = config.wave_file
    print('inference file: ', len(wave_file))

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

    inference(waveglow, mel_spectrogram, wave_file, config, output_dir, device, 'inference dataset')
