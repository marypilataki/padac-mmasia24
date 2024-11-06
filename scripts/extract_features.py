import argparse
from pathlib import Path
import torch
import dac
from madmom.audio.signal import FramedSignal
import resampy
import soundfile as sf

SAMPLE_RATE = 44100
EXCERPT_DURATION = 1.0
HOP_DURATION = 0.8 # 0.2 sec overlapping

def main(dac_path, output_dir, dataset_path):
    """
    Args:
        dac_path (str): full path to the trained padac model.
        output_dir (str): full path to the output directory where to save features.
        dataset_path (str): full path to the audio dataset.
    """

    # load dataset
    audiofilenames = []
    for ext in ['.wav', '.flac']:
        audiofilenames.extend(list(dataset_path.rglob(f'*{ext}')))
    assert len(audiofilenames) > 0, f'No audio files found in {dataset_path}'
    output_dir.mkdir(parents=True, exist_ok=True)

    dac_model = dac.DAC(sample_rate= 44100,
                        encoder_dim = 64,
                        encoder_rates = [2, 4, 8, 8],
                        decoder_dim = 1536,
                        decoder_rates = [8, 8, 4, 2],
                        add_conditioner = True,
                        n_codebooks = 9,
                        codebook_size = 1024,
                        codebook_dim = 8,
                        quantizer_dropout = 0.5)
    dac_model.load_state_dict(torch.load(str(dac_path), map_location='cpu')['state_dict'])
    dac_model.eval()

    for param in dac_model.parameters():
        assert param.data.all !=0, "DAC state not loaded!"
    print('DAC model successfully loaded in eval mode for feature extraction.')

    if torch.cuda.is_available():
        dac_model.cuda()

    print(f'Extracting features for {len(audiofilenames)} files.')

    # extract features
    for f in audiofilenames:
        track_id = f.stem
        audio, sample_rate = sf.read(f)
        if sample_rate != SAMPLE_RATE:
            audio = resampy.resample(audio, sample_rate, SAMPLE_RATE)
        audio = FramedSignal(audio, frame_size=SAMPLE_RATE*EXCERPT_DURATION, hop_size=SAMPLE_RATE*HOP_DURATION)

        for i, chunk in enumerate(audio):
            save_filename = output_dir / f'{track_id}_{i}.pt'
            if not save_filename.exists():
                chunk = torch.FloatTensor(chunk).unsqueeze(0).unsqueeze(0) # [1 x 1 x 44100]
                if torch.cuda.is_available():
                    chunk = chunk.cuda()

                with torch.no_grad():
                    latent_space = dac_model.encoder(chunk)

                # batch x feat x time => batch x time x feat
                latent_space = latent_space.permute(0, 2, 1)
                torch.save(latent_space, save_filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dac_path', type=str, default='./runs/padac/100k/dac/weights.pth',
                        help='Full path to the trained padac model.')
    parser.add_argument('--output_dir', type=str, default='./datasets/train_features/',
                        help='Full path to the output directory where to save features.')
    parser.add_argument('--dataset_path', type=str, default='/homes/mpm30/dataset/train_audio/',
                        help='Full path to the audio dataset.')
    args = parser.parse_args()
    main(Path(args.dac_path), Path(args.output_dir), Path(args.dataset_path))