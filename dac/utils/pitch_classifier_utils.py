from pathlib import Path
import sys
import torch
import numpy as np
from pretty_midi import PrettyMIDI
import librosa
import audioread
import math
import json

DAC_SAMPLE_RATE = 87
INPUT_DURATION = 1.0
N_SAMPLES = int(DAC_SAMPLE_RATE * INPUT_DURATION)
N_NOTES = 128

# tensorflow imports for basic pitch
#sys.path.append('/homes/mpm30/Dev/dac_for_mir/dac-for-mir/basic-pitch/') if Path(
#    '/homes/mpm30/Dev/dac_for_mir/dac-for-mir/basic-pitch/').exists() else sys.path.append(
#    r'C:\Dev\descript_from_git\dac-for-mir\basic-pitch')
#from basic_pitch import ICASSP_2022_MODEL_PATH
#from basic_pitch.inference import predict
#import tensorflow as tf
import csv

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

"""
Ground truth dicts
"""
# SOL_METADATA_PATH = '/import/c4dm-datasets/TinySOL/TinySOL_metadata.csv'
# NSYNTH_METADATA_PATH_TRAIN = '/import/c4dm-datasets/nsynth/nsynth-train/examples.json'
# NSYNTH_METADATA_PATH_VALID = '/import/c4dm-datasets/nsynth/nsynth-valid/examples.json'
# SOL_PITCH_DICT = load_sol_labels(metadata_path=SOL_METADATA_PATH)
# nsynth_metadata_train = load_json(NSYNTH_METADATA_PATH_TRAIN)
# nsynth_metadata_valid = load_json(NSYNTH_METADATA_PATH_VALID)
# NSYNTH_METADATA = nsynth_metadata_train | nsynth_metadata_valid
# del nsynth_metadata_train
# del nsynth_metadata_valid
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

# def add_pitch_labels_to_batch(batch, use_basic_pitch=False, labels_path=None):
#     """Function to add pitch labels to a batch of audio data
#
#     Parameters
#     ----------
#     batch (torch tensor): batch of audio data [batch size, n samples]
#
#     Returns
#     -------
#     batch (torch tensor): batch of audio data with pitch labels [batch size, n samples, n notes]
#     """
#     batch_size = batch['signal'].shape[0]
#     n_samples = int(batch['signal'].duration * DAC_SAMPLE_RATE)
#     pitch_labels = torch.zeros((batch_size, n_samples, N_NOTES), dtype=torch.float32)
#
#     # use trained basic pitch model to assign pitch labels
#     if use_basic_pitch==True:
#         #basic_pitch_model = tf.saved_model.load(str(ICASSP_2022_MODEL_PATH))
#         for batch_idx in range(batch_size):
#             _, midi_data, _ = predict(batch["signal"][batch_idx].audio_data.squeeze().squeeze().detach().cpu().numpy(), sample_rate=batch["signal"].sample_rate, model_or_model_path=ICASSP_2022_MODEL_PATH)
#             pitch_labels[batch_idx] = torch.from_numpy(get_label_from_midi(midi_data, n_samples=n_samples, sampling_rate=DAC_SAMPLE_RATE, n_notes=N_NOTES))
#         batch['pitch_labels'] = pitch_labels.to(batch['signal'].device)
#         #print('pitch labels ', batch['pitch_labels'].shape) batch, 87, 128
#         tf.keras.backend.clear_session()
#     # use ground truth midi files to assign pitch labels
#     elif use_basic_pitch==False:
#         for batch_idx in range(batch_size):
#             audiofile = Path(batch['path'][batch_idx])
#             #parents = list(audiofile.parents)
#             #split = parents[0].name
#             #with audioread.audio_open(str(audiofile)) as f:
#             #    audio_duration = f.duration
#             #    sample_rate = f.samplerate
#             #    #assert sample_rate == batch['signal'].sample_rate, f"Sample rate {sample_rate} does not match {batch['signal'].sample_rate}!"
#             #num_samples = math.ceil(sample_rate * audio_duration)
#             #midifile = parents[1] / f'{split}_labels' / (audiofile.name[:-4] + '.mid')
#             #if not midifile.exists():
#             #    midifile = parents[1] / f'{split}_labels' / (audiofile.name[:-4] + '.midi')
#             #assert midifile.exists(), f"Ground truth midi file {midifile} does not exist!"
#
#             #label = get_label_from_midi(midifile, n_samples=num_samples, sampling_rate=sample_rate, n_notes=N_NOTES)
#             # get excerpt of label corresponding to audio
#             #start_sample = librosa.time_to_samples(batch['offset'][batch_idx].item(), sr=sample_rate)
#             #end_sample = librosa.time_to_samples(batch['offset'][batch_idx].item() + 1.0, sr=sample_rate) # duration = 1.0
#             #label = label[start_sample:end_sample, :]
#             #label = resample_label(label, original_sr=sample_rate, new_sr=DAC_SAMPLE_RATE)
#             #pitch_labels[batch_idx] = torch.from_numpy(label)
#             label_file = labels_path / (str(audiofile.stem)+'.npz')
#             start_sample = librosa.time_to_samples(batch['offset'][batch_idx].item(), sr=DAC_SAMPLE_RATE)
#             end_sample = librosa.time_to_samples(batch['offset'][batch_idx].item() + 1.0, sr=DAC_SAMPLE_RATE)
#             label = np.load(label_file, allow_pickle=True)['pitch_label']
#             pitch_labels[batch_idx] = torch.from_numpy(label[start_sample:end_sample, :])
#         batch['pitch_labels'] = pitch_labels.to(batch['signal'].device)
#     elif use_basic_pitch=='single_note':
#         # load tinysol metadata
#         for batch_idx in range(batch_size):
#             dataset = Path(batch['source'][batch_idx]).parent.stem
#             key = Path(batch['path'][batch_idx]).stem
#             label = np.zeros(shape=(87, 128))
#             if dataset=='TinySOL':
#                 k = [p for p in SOL_PITCH_DICT.keys() if key in p][0]
#                 label[:, SOL_PITCH_DICT[k]] = 1
#             elif dataset=='NSynth':
#                 label[:, NSYNTH_METADATA[key]['pitch']-21] = 1
#             else:
#                 raise ValueError(f"Dataset {dataset} not supported!")
#
#             pitch_labels[batch_idx] = torch.from_numpy(label)
#         batch['pitch_labels'] = pitch_labels.to(batch['signal'].device)
#     elif use_basic_pitch=='gtzan_only':
#         basic_pitch_model = tf.saved_model.load(str(ICASSP_2022_MODEL_PATH))
#         for batch_idx in range(batch_size):
#             # use basic pitch for gtzan
#             if Path(batch['source'][batch_idx]).parent.name == 'gtzan':
#                 _, midi_data, _ = predict(
#                     batch["signal"][batch_idx].audio_data.squeeze().squeeze().detach().cpu().numpy(),
#                     sample_rate=batch["signal"].sample_rate, model_or_model_path=basic_pitch_model)
#                 pitch_labels[batch_idx] = torch.from_numpy(
#                     get_label_from_midi(midi_data, n_samples=n_samples, sampling_rate=DAC_SAMPLE_RATE, n_notes=N_NOTES))
#             elif Path(batch['source'][batch_idx]).parent.name == 'maestro': # get ground truth for maestro
#                 audiofile = Path(batch['path'][batch_idx])
#                 parents = list(audiofile.parents)
#                 split = parents[0].name
#                 with audioread.audio_open(str(audiofile)) as f:
#                     audio_duration = f.duration
#                 num_samples = math.ceil(DAC_SAMPLE_RATE * audio_duration)
#
#                 midifile = parents[1] / f'{split}_labels' / (audiofile.name[:-4] + '.mid')
#                 if not midifile.exists():
#                     midifile = parents[1] / f'{split}_labels' / (audiofile.name[:-4] + '.midi')
#                 assert midifile.exists(), f"Ground truth midi file {midifile} does not exist!"
#
#                 pitch_labels[batch_idx] = torch.from_numpy(
#                     get_label_from_midi(midifile, n_samples=num_samples, sampling_rate=DAC_SAMPLE_RATE, n_notes=N_NOTES,
#                                         offset=batch['offset'][batch_idx].item(),
#                                         duration=batch['signal'][batch_idx].duration))
#         batch['pitch_labels'] = pitch_labels.to(batch['signal'].device)
#         tf.keras.backend.clear_session()
#     else:
#         raise ValueError(f"use_basic_pitch not supported: {use_basic_pitch}!")
#     return batch

"""
todo: add in  audiotools data datasets
        # todo: add ground truth label
        item = {
            "signal": signal,
            "source_idx": source_idx,
            "item_idx": item_idx,
            "source": str(self.sources[source_idx]),
            "path": str(path),
            "offset": signal.metadata["offset"]
        }
        item['label'] = get_midi_label(item)
        if self.transform is not None:
            item["transform_args"] = self.transform.instantiate(state, signal=signal)
        return item


def get_midi_label(item):
    import librosa
    import torch
    from pretty_midi import PrettyMIDI
    dac_rate = 87
    start_time = item["offset"]
    end_time = start_time + item["signal"].duration
    label_path = Path(item["path"]).parent / (Path(item["path"]).stem + '.midi')
    num_samples = int(item["signal"].duration * dac_rate)
    num_notes = 128
    label = torch.zeros(num_samples, num_notes)

    midi_data = PrettyMIDI(str(label_path))
    for instrument in midi_data.instruments:
        if not instrument.is_drum:
            for note in instrument.notes:
                if note.start >= start_time:
                    note_start = librosa.time_to_samples(note.start - start_time, sr=dac_rate)
                    pitch_index = note.pitch - 21
                    if note.end <= end_time:
                        note_end = librosa.time_to_samples(end_time - note.end, sr=dac_rate)
                        assert note_end => note_start, "End sample must be later than start sample!"
                        note_end = note_end + 1 if note_end == note_start else note_end
                        label[note_start:note_end, pitch_index] = 1
                    else:
                        label[note_start: , pitch_index] = 1
    return label
"""