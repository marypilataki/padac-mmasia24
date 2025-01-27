from pathlib import Path
import torch
import numpy as np
from pretty_midi import PrettyMIDI
import librosa
import csv
import json

DAC_SAMPLE_RATE = 87
INPUT_DURATION = 1.0
N_SAMPLES = int(DAC_SAMPLE_RATE * INPUT_DURATION)
N_NOTES = 128

def load_json(path):
    """Loads a json option file

    Parameters:
        path (str) -- Path to the json file

    Returns:
        Option dictionary
    """
    path = Path(path)
    if path.exists():
        with open(path, 'r') as f:
            config_text = f.read()
        opt = json.loads(config_text)
        return opt
    else:
        print(f'json file {path} could not be found')
        exit(0)
def load_sol_labels(metadata_path):
    pitch_dict = {}
    with open(metadata_path) as csvfile:
        reader = csv.reader(csvfile)
        for i, row in enumerate(reader):
            if i>0:
                pitch_dict[row[0]] = int(row[8])-21 # index
    return pitch_dict


def resample_label(label, original_sr, new_sr):
    """Function to resample a ground truth roll.
    ----------
    label (np array): ground truth roll
    original_sr (int): sampling rate of input label
    new_sr (int): sampling rate for resampling label

    Returns
    -------
    cod_label (np array): codified ground truth roll
    """
    max_sample = label.shape[0]
    t = librosa.samples_to_time(max_sample, sr=original_sr)
    new_max_sample = librosa.time_to_samples(t, sr=new_sr)

    # resampled label dimensionality
    resampled_label_dim = list(label.shape)
    resampled_label_dim[0] = new_max_sample

    if isinstance(label, np.ndarray):
        resampled_label = np.zeros(shape=(resampled_label_dim))
    elif isinstance(label, torch.Tensor):
        resampled_label = torch.Tensor(resampled_label_dim)
    else:
        raise ValueError("Input label must be either a NumPy array or a PyTorch tensor!")

    assert label.shape[-1] == resampled_label.shape[-1], 'Labels must match other than T dim.'

    for n in range(max_sample):
        # time stamp for this sample
        t = librosa.samples_to_time(n, sr=original_sr)
        # get codified audio frame given time stamp
        new_n = librosa.time_to_samples(t, sr=new_sr)

        try:
            resampled_label[new_n, ...] = label[n, ...]
        except:
            print(f'Error at sample {new_n}/{new_max_sample} with time stamp {t}')

    return resampled_label

def get_label_from_midi(midi_file_or_data, n_samples, sampling_rate, n_notes=N_NOTES):
    """Function to extract one-hot label vectors given a midi file or pretty midi object

    Parameters
    ----------
    midi_file (string or Path object or pretty midi object): path to the midi file or midi data
    n_samples (int): number of samples of the audio file corresponding to this midi
    sampling_rate (int): sampling rate
    n_notes (int): number of notes in the dataset. Defaults to 128.
    offset (float): offset in seconds. Defaults to None.
    duration (float): duration in seconds. Defaults to 1.0. Used only if offset is specified.

    Returns
    -------
    pitch_label (np array): pitch one-hot label [n samples, n notes]
    """
    pitch_label = np.zeros((n_samples, n_notes), dtype=np.uint8)

    # if input is midi data it is basic pitch prediction, corresponds to 1 sec of audio
    # if input is path then it is ground truth midi, extract the label for the whole audio file first, use dac sample rate
    if isinstance(midi_file_or_data, str) or isinstance(midi_file_or_data, Path):
        midi_data = PrettyMIDI(str(midi_file_or_data))
    elif isinstance(midi_file_or_data, PrettyMIDI):
        midi_data = midi_file_or_data
    else:
        raise ValueError(f"midi_file_or_data must be a string, Path object, or pretty midi object but is {type(midi_file_or_data)}")

    for instrument in midi_data.instruments:
        # only consider non-percussive instruments
        if not instrument.is_drum:
            for note in instrument.notes:
                start_time = note.start
                end_time = note.end
                pitch_index = note.pitch
                assert pitch_index >= 0, f"Pitch index must be non-negative! Got {pitch_index}"
                assert pitch_index < n_notes, f"Pitch index must be less than n_notes! Got {pitch_index}"

                start_sample = librosa.time_to_samples(start_time, sr=sampling_rate)
                end_sample = librosa.time_to_samples(end_time, sr=sampling_rate)

                assert end_sample >= start_sample, "End sample must be later than start sample!"
                if start_sample == end_sample:
                    end_sample = end_sample + 1

                pitch_label[start_sample:end_sample, pitch_index] = 1

    return pitch_label

def get_label_excerpt_from_midi(midifile, n_samples, sampling_rate, n_notes=N_NOTES, offset=None, duration=1.0):
    pitch_label = np.zeros((n_samples, n_notes), dtype=np.uint8)

    # if input is midi data it is basic pitch prediction, corresponds to 1 sec of audio
    # if input is path then it is ground truth midi, extract the label for the whole audio file first, use dac sample rate
    if isinstance(midifile, str) or isinstance(midifile, Path):
        midi_data = PrettyMIDI(str(midifile))
    elif isinstance(midifile, PrettyMIDI):
        midi_data = midifile
    else:
        raise ValueError(f"midi_file_or_data must be a string, Path object, or pretty midi object but is {type(midi_file_or_data)}")

    for instrument in midi_data.instruments:
        # only consider non-percussive instruments
        if not instrument.is_drum:
            for note in instrument.notes:
                start_time = note.start
                end_time = note.end
                midi_pitch = note.pitch
                pitch_index = midi_pitch - 21

                # get the whole label first
                start_sample = librosa.time_to_samples(start_time, sr=sampling_rate)
                end_sample = librosa.time_to_samples(end_time, sr=sampling_rate)

                assert end_sample >= start_sample, "End sample must be later than start sample!"
                if start_sample == end_sample:
                    end_sample = end_sample + 1
                pitch_label[start_sample:end_sample, pitch_index] = 1

    # extract the duration sec excerpt
    excerpt_start_sample = librosa.time_to_samples(offset, sr=sampling_rate)
    excerpt_end_sample = librosa.time_to_samples(offset + duration, sr=sampling_rate)
    pitch_label = pitch_label[excerpt_start_sample:excerpt_end_sample, :]
    return pitch_label