import json
import os
import librosa
import torch
import numpy as np
import pretty_midi
from typing import DefaultDict, Dict, List, Optional, Tuple, Union
from collections import defaultdict

DAC_SAMPLE_RATE = 87
MIDI_OFFSET = 21
MAX_FREQ_IDX = 127

def load_json(path):
    """Loads a json option file

    Parameters:
        path (str) -- Path to the json file

    Returns:
        Option dictionary
    """
    assert os.path.exists(path), f"The config file could not be found: {path}"
    with open(path, 'r') as f:
        config_text = f.read()
    opt = json.loads(config_text)
    return opt


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
