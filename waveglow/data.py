from torch.utils import data
import random
import torch
from dataclasses import dataclass


@dataclass
class Segment:
    index: int
    start: int
    end: int


class Dataset(data.Dataset):
    def __init__(self, audio_list, segment_size):
        self.audio_list = audio_list
        self.segment_size = segment_size

        self.segment_list = list()
        for i, audio in enumerate(audio_list):
            for start in range(0, audio.shape[0] - segment_size, segment_size //2):
                self.segment_list.append(Segment(i, start, start + segment_size))

    def __getitem__(self, index):
        segment = self.segment_list[index]
        return torch.from_numpy(self.audio_list[segment.index][segment.start:segment.end]).float()

    def __len__(self):
        return len(self.segment_list)
