from dataclasses import dataclass, field
from typing import List
import json


@dataclass
class Config:
    # preprocess
    wave_file: List[str] = field(default_factory=list)
    tgt_file: List[str] = field(default_factory=list)

    # train dataset
    preprocessed_dir: str = ''
    train_file: List[str] = field(default_factory=list)
    valid_file: List[str] = field(default_factory=list)

    # audio parameter
    mel_segment_size: int = 0
    audio_segment_size: int = 16000
    num_mel: int = 80
    filter_length: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    sampling_rate: int = 44100
    mel_min: float = 0
    mel_max: float = 8000

    # dataset
    shuffle: bool = True
    batch_size: int = 16
    num_workers: int = 4
    pin_memory: bool = True
    drop_last: bool = True

    # train
    gpu_index: int = 0
    learning_rate: float = 0.0001
    training_epoch: int = 400
    last_epoch: int = 0
    checkpoint_dir: str = './checkpoint'
    checkpoint_interval: int = 3600  # 3600s = 1hours
    last_checkpoint_file: str = ''
    model_path: str = ''
    use_valid: bool = False
    use_log: bool = False
    log_dir: str = './log'
    train_log: str = ''
    valid_log: str = ''
    remove_checkpoint: bool = False

    # model
    sigma: float = 1.0
    n_flow: int = 12
    n_group: int = 8
    n_early_every: int = 4
    n_early_size: int = 2
    wn_n_layer: int = 8
    wn_n_channel: int = 256
    wn_kernel_size: int = 3


def save(cf: Config, path: str):
    with open(path, 'w') as f:
        json.dump(cf.__dict__, f, indent=2)


def load(path: str):
    with open(path, 'r') as f:
        json_obj = json.load(f)
        config = Config(**json_obj)
    return config
