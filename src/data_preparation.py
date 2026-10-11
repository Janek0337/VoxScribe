from src.config import DATA_DIR

import torch
import torchaudio
import torch.nn.functional as nnF
import torchaudio.transforms as T
from torch import tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import sys

torch.random.manual_seed(0)

class CharTokenizer:
    def __init__(self):
        self.pad_char = '_'
        self.blank_char = '.'

        self.chars = [self.pad_char, self.blank_char, 'a', 'ą', 'b', 'c', 'ć', 'd', 'e', 'ę', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'ł', 'm', 'n', 'ń', 'o', 'ó',
                      'p', 'q', 'r', 's', 'ś', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'ż', 'ź', ' ']
        self.char_to_idx = {char: idx for idx, char in enumerate(self.chars)}
        self.idx_to_char = {idx: char for idx, char in enumerate(self.chars)}

    def encode_text(self, text: str):
        try:
            return [self.char_to_idx[char] for char in text]
        except KeyError:
            print("Input text must only contain small letters of polish alphabet.", file=sys.stderr)

    def decode_text(self, encoded: list[int]):
        return "".join([self.idx_to_char[idx] for idx in encoded if idx > 1])

class AudioDataset(Dataset):
    def __init__(self):
        super().__init__()

        self.audio_dir = DATA_DIR
        self.audio_data = pd.read_csv(DATA_DIR / 'audio_data.csv')
        self.audio_data = self.audio_data[self.audio_data['duration[ms]'] <= 9000]

        n_fft = 1024
        win_length = None
        hop_length = 512
        n_mels = 128
        sample_rate = 16000

        self.transformer = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            center=True,
            pad_mode="reflect",
            power=2.0,
            norm="slaney",
            n_mels=n_mels,
            mel_scale="htk",
        )

        waveform_size = 9 * 16000
        dummy_waveform = torch.zeros(1, waveform_size) 
        self.max_frames = self.transformer(dummy_waveform).shape[-1]

    def __len__(self):
        return self.audio_data.shape[0]

    def __getitem__(self, idx):
        filename = self.audio_data['path'].iloc[idx]
        filepath = self.audio_dir / filename
        sentence = self.audio_data['sentence'].iloc[idx]
        spectrogram = self._make_mel_spectrogram(filepath)
        return spectrogram, sentence

    def _make_mel_spectrogram(self, audio_path):
        SPEECH_WAVEFORM, _ = torchaudio.load(audio_path)
        spectrogram = self.transformer(SPEECH_WAVEFORM)
        current_frames = spectrogram.shape[-1]
    
        if current_frames < self.max_frames:
            missing_frames = self.max_frames - current_frames
            spectrogram = nnF.pad(spectrogram, (0, missing_frames))
        elif current_frames > self.max_frames:
            spectrogram = spectrogram[:, :, :self.max_frames]

        return spectrogram

class AudioCollator:
    def __init__(self):
        self.tokenizer = CharTokenizer()

    def collate_fn(self, input: list[tuple[tensor, str]]):
        spectrograms = [tupl[0] for tupl in input]
        ready_spectrograms = torch.stack(spectrograms)

        texts = [tupl[1] for tupl in input]
        lengths = tensor([len(text) for text in texts], dtype=torch.long)

        encoded_text = [tensor(self.tokenizer.encode_text(text)) for text in texts]
        ready_text = pad_sequence(encoded_text, padding_value=self.tokenizer.encode_text(self.tokenizer.pad_char)[0], batch_first=True)

        return (ready_spectrograms, ready_text, lengths)

    def __call__(self, input):
        return self.collate_fn(input)
