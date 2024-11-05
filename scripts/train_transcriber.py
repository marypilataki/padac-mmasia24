"""
Script to train shallow transcriber on latent space embeddings extracted from PA-DAC.
Some code borrowed from https://github.com/rainerkelz/framewise_2016
"""
import os
from pathlib import Path
import argparse
import numpy as np
import time
import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import precision_recall_fscore_support as prfs

from transcriber.dataset import DacDataset
from transcriber.network import LinearModel
from transcriber.utils import load_json

def train_one_epoch(model, dataloader, optimizer, cuda_avail):
    """Function to train the model for one epoch.

    Parameters
    ----------
    model (PyTorch model): the model to be trained
    dataloader (PyTorch dataloader object): data loader
    optimizer (PyTorch optimizer object): optimizer algorithm to be use while training
    cuda_avail (bool): whether a gpu is available in the current device (True) or not (False)

    Returns
    -------
    smoothed_loss (float): loss value
    """
    model.train()
    loss_function = nn.BCEWithLogitsLoss(reduction='mean')
    smoothed_loss = 1.

    for data, label in dataloader:
        if cuda_avail:
            data = data.cuda()
            label = label.cuda()

        t = data.shape[2]  # usable time steps
        label = label[:, :t, :]

        optimizer.zero_grad()
        pred_pitch = model.forward(data)
        loss = loss_function(pred_pitch.squeeze(), label[:, :t, :])

        loss.backward()
        optimizer.step()
        smoothed_loss = smoothed_loss * 0.9 + loss.detach().cpu().item() * 0.1

        # bail if NaN or Inf is encountered
        if np.isnan(smoothed_loss) or np.isinf(smoothed_loss):
            print('encountered NaN/Inf in smoothed_loss "{}"'.format(smoothed_loss))
            exit(-1)

    return smoothed_loss


def evaluate(model, dataloader, threshold=0.5, cuda_avail=False):
    """Function to validate model.

    Parameters
    ----------
    model (Pytorch model): model to be validated
    dataloader (Pytorch dataloader): dataloader
    threshold (float): threshold above which a prediction probability is considered as 1
    cuda_avail (bool): whether there is available cpu on the system (True) or not (False). Defaults to False.

    Returns
    -------
    results (dict): dictionary containing loss, precision, recall and f1-score
    """
    model.eval()
    loss_function = nn.BCELoss(reduction='mean')
    smoothed_loss = 1.
    pitch_ground_truth = []
    pitch_predictions = []

    # for codec, onsets, pitch, instrument, multitrack in dataloader:
    for data, label in dataloader:
        t = data.shape[2]  # usable time steps
        label = label[:, :t, :]
        if cuda_avail:
            data = data.cuda()
            label = label.cuda()

        pred_pitch = model.predict(data).squeeze()
        valid_loss = loss_function(pred_pitch, label[:, :t, :])
        smoothed_loss = smoothed_loss * 0.9 + valid_loss.detach().cpu().item() * 0.1

        pitch_ground_truth.append(label.detach().cpu().numpy())
        pitch_predictions.append((pred_pitch.detach().cpu().numpy() > threshold) * 1)

    pitch_ground_truth = np.vstack(np.vstack(pitch_ground_truth))
    pitch_predictions = np.vstack(np.vstack(pitch_predictions))

    assert pitch_ground_truth.shape == pitch_predictions.shape, f'Ground truth and predictions shapes must match! labels are {pitch_ground_truth.shape} and predictions are {pitch_predictions.shape}'
    p_pitch, r_pitch, f_pitch, _ = prfs(pitch_ground_truth, pitch_predictions, average='micro')

    results = dict(
        loss=smoothed_loss,
        P=p_pitch,
        R=r_pitch,
        F=f_pitch
    )
    return results


def main(opt):
    """Main function to train a model based on user config.

    Parameters
    ----------
    opt (dict): User configuration options.
    """

    # prepare train splits
    train_dataset_path = Path(opt['data']['dataset_path']) / 'train_data'
    train_labels_path = Path(opt['data']['labels_path']) / 'train_labels'
    # prepare valid splits
    valid_dataset_path = Path(opt['data']['dataset_path']) / 'valid_data'
    valid_labels_path = Path(opt['data']['labels_path']) / 'valid_labels'

    train_dataset = DacDataset(train_dataset_path, train_labels_path)
    valid_dataset = DacDataset(valid_dataset_path, valid_labels_path)

    train_loader = DataLoader(
        train_dataset,
        batch_size=opt["train"]["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=opt["train"]["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=True
    )

    n_epochs = opt["train"]["n_epochs"]
    # keep track of best validation state
    best_valid_loss = 1000
    save_dir = opt["save_dir"]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # instantiate model and optimizer
    model = LinearModel(**opt["model"]["params"])
    model.init_weights()
    optimizer = torch.optim.Adam(model.parameters(), opt["train"]["lr"], weight_decay=opt["train"]["weight_decay"])

    # use cuda if available
    cuda_avail = torch.cuda.is_available()
    if cuda_avail:
        model = model.cuda()

    #
    # main training loop
    #
    torch.set_num_threads(1)
    for epoch in range(n_epochs):

        """Training phase"""
        start_time = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, cuda_avail=cuda_avail)
        print(f'Finished training epoch {epoch + 1} in {(time.time() - start_time) / 3600} hours.')

        """Validation phase"""
        results = evaluate(model, valid_loader, threshold=0.3, cuda_avail=cuda_avail)
        valid_loss = results['loss']

        """Print losses and metrics"""
        print()
        print(f'--------- Epoch {epoch + 1}-------------')
        print(f'Training Loss {train_loss}')
        print(f"Validation Loss {valid_loss}")
        print(f"Pitch P - R - F: {results['P']} - {results['R']} - {results['F']}")

        # keep track of best validation loss state
        epoch_number = epoch + 1
        if valid_loss < best_valid_loss:
            save_path = os.path.join(save_dir, f'{epoch_number}_model_state_best.pth')
            best_valid_loss = valid_loss
            torch.save({
                'epoch': epoch_number,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': valid_loss}, save_path)
            print(f"Saved model at best validation loss state {epoch_number}.")

        # save state every 10 epochs
        if (epoch_number) % 10 == 0:
            save_path = os.path.join(save_dir, f'{epoch_number}_model_state.pth')
            torch.save({
                'epoch': epoch_number,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': valid_loss}, save_path)
            print(f"Saved model at epoch {epoch_number}.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config_file', type=str, default='trancriber.json', help='Full path to the config file')
    args = parser.parse_args()
    opt = load_json(args.config_file)
    print(f"Training on {opt['data']['dataset_name']} dataset.")
    main(opt)