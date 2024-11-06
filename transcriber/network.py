import torch
from torch import nn
from torch.nn import init

class TrascriptionModel(nn.Module):

    def __init__(self):
        """TrascriptionModel class constructor. This class is intended to be used as a base model for all networks.
        """
        super().__init__()

    def init_weights(self):
        """Function to apply weight initialisation. Needs to be called once after the model is instantiated.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight, gain=1)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight, init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        """Fuction to run a forward pass. Returns logits.
        Parameters
        ----------
        x (torch Tensor): input feature

        Returns
        -------
        (torch Tensor): model's output
        """
        return x

    # returns pseudo probabilities
    def predict(self, x):
        """Function to return model's prediction for a given input feature. This function will return pseudo-probabilities
        which will then need to be transformed into a multi-instrument roll using suitable thresholds.

        Parameters
        ----------
        x (torch Tensor): input feature

        Returns
        -------
        (torch Tensor): model's prediction in the form of pseudo-probabilities
        """
        return torch.sigmoid(self.forward(x))


class LinearModel(TrascriptionModel):
    def __init__(self, in_features=1024, out_features=128, hidden_units=512, dropout_prob=0.2):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.hidden_units = hidden_units
        self.dropout_prob = dropout_prob

        self.linear = nn.Sequential(
            nn.Linear(self.in_features, self.hidden_units),
            nn.ReLU(),
            nn.Dropout(self.dropout_prob),
            nn.Linear(self.hidden_units, self.out_features)
        )


    def forward(self, x):
        """Function to run a forward pass.

        Parameters
        ----------
        x (torch Tensor): input feature [BATCH X TIME X FEATURES]

        Returns
        -------
        x (torch Tensor): predictions for this feature [BATCH x TIME X CLASSES]
        """
        output = self.linear(x)
        return output