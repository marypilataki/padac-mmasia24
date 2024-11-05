from torch import nn

class Conditioner(nn.Module):
    def __init__(self, input_size=1024, output_size=128, mid_size=512):
        super(Conditioner, self).__init__()
        net_ly = []
        net_ly += [nn.Linear(input_size, mid_size)]
        net_ly += [nn.ReLU()]
        net_ly += [nn.Linear(mid_size, mid_size)]
        net_ly += [nn.ReLU()]
        net_ly += [nn.Linear(mid_size, output_size)]
        net_ly += [nn.ReLU()]
        net_ly += [nn.Linear(output_size, output_size)]
        self.net = nn.Sequential(*net_ly)

    def forward(self, x):
        # do not pass model output through sigmoid when using BCEWithLogitsLoss
        # pass model output through sigmoid when using CrossEntropyLoss
        return self.net(x)