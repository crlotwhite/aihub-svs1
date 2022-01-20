import argparse
import tqdm
from preference import Config, save, load
import json
import os
import pickle
from scipy.io.wavfile import read
import math
import random
import logging
import sys

MAX_WAV_VALUE = 32768.0


def match_textgrid(wav, tgt_dir):
    tgt_list = list()
    for path in wav:
        basename = os.path.basename(path).split('.')[0]
        tgt_list.append(os.path.join(tgt_dir, '{}.json'.format(basename)))
    return tgt_list


def parse_filepath(path: str):
    path_list = []
    with open(path) as f:
        path_list = f.readlines()
        # remove '\n'
        path_list = [path.split('\n')[0] for path in path_list]
    return path_list


def split_segment(notes, config: Config):
    segment_duration = config.audio_segment_size / float(config.sampling_rate)
    segment_start = segment_end = 0
    segment_list = list()

    for note in notes:
        if len(note['lyric']) == 0:
            continue

        start = float(note['start_time'])
        end = float(note['end_time'])

        if segment_start == 0 and segment_end == 0:
            segment_start = start
            segment_end = end
        elif segment_end + segment_duration > start:
            segment_end = end
        else:
            start_idx = math.floor(segment_start * config.sampling_rate)
            end_idx = math.ceil(segment_end * config.sampling_rate)

            segment_list.append((start_idx, end_idx))

            segment_start = start
            segment_end = end
    start_idx = math.floor(segment_start * config.sampling_rate)
    end_idx = math.ceil(segment_end * config.sampling_rate)

    segment_list.append((start_idx, end_idx))

    return segment_list


def precess(wav_file, tgt_file, out_file, config: Config, desc, logger):
    save_file = list()

    for i in tqdm.tqdm(range(len(wav_file)), desc=desc):
        logger.info('preprocess ' + wav_file[i])

        basename = os.path.basename(wav_file[i]).split('.')[0]

        sampling_rate, audio = read(wav_file[i])
        audio = audio / MAX_WAV_VALUE

        with open(tgt_file[i], 'r', encoding='UTF-8') as f:
            obj = json.load(f)
        notes = obj['notes']

        segment_list = split_segment(notes, config)

        for start, end in segment_list:
            wave = audio[start: end]
            save_path = os.path.join(config.preprocessed_dir, '{}_{}.pkl'.format(basename, start))

            with open(save_path, 'wb') as f:
                pickle.dump(
                    {'audio': wave.tolist()},
                    f
                )
            save_file.append(save_path)

    random.shuffle(save_file)
    train_size = int(len(save_file) * 0.9)
    config.train_file = save_file[:train_size]
    config.valid_file = save_file[train_size:]


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

    print('Initializing PreProcess..')

    cmd = ' '.join(sys.argv)
    logger.info('python ' + cmd)

    parser = argparse.ArgumentParser()
    parser.add_argument('file', help='file that have train audio file path')
    parser.add_argument('textgrid_dir', help='text grid directory')
    parser.add_argument('--output_dir', '-o', help='save directory for preprocessed file')
    parser.add_argument('--config', '-c', help='preprocess configuration file')

    # argument parsing
    args = parser.parse_args()
    config_file = args.config
    file = args.file
    textgrid_dir = args.textgrid_dir
    output_dir = args.output_dir

    # configuration setting
    if config_file is not None:
        config = load(config_file)
    else:
        config = Config()
    if output_dir is not None:
        config.preprocessed_dir = output_dir
    else:
        output_dir = config.preprocessed_dir
    config.wave_file = []
    config.tgt_file = []
    config.train_file = []
    config.valid_file = []

    # parsing train wave file
    config.wave_file = parse_filepath(file)
    config.tgt_file = match_textgrid(config.wave_file, textgrid_dir)

    # make preprocess result folder
    os.makedirs(output_dir, exist_ok=True)
    print('make preprocess directory: ' + output_dir)

    precess(config.wave_file, config.tgt_file, config.train_file, config, 'wave preprocess', logger)

    path = os.path.join(config.preprocessed_dir, 'config.json')
    save(config, path)
