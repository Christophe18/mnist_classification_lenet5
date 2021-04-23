import torch.nn as nn


class LeNet5(nn.Module):
    """ LeNet-5 (LeCun et al., 1998)

        - For a detailed architecture, refer to the lecture note
        - Freely choose activation functions as you want
        - For subsampling, use max pooling with kernel_size = (2,2)
        - Output should be a logit vector
    """

    def __init__(self):
        super(LeNet5, self).__init__()

        self.conv_layers = nn.Sequential(
            # C1 (5*5*1+1)*6 = 156
            nn.Conv2d(in_channels=1,
                      out_channels=6,
                      kernel_size=5,
                      stride=1,
                      padding=2,
                      bias=True),
            nn.Tanh(),
            # S2
            nn.AvgPool2d(kernel_size=2,
                         stride=2),
            # C3 (5*5*6+1)*16 = 2416
            nn.Conv2d(in_channels=6,
                      out_channels=16,
                      kernel_size=5,
                      stride=1,
                      padding=0,
                      bias=True),
            nn.Tanh(),
            # S4
            nn.AvgPool2d(kernel_size=2,
                         stride=2)
        )
        self.fc_layers = nn.Sequential(
            # C5 (16*5*5+1)*120 = 48120
            nn.Linear(in_features=16 * 5 * 5,
                      out_features=120),
            nn.Tanh(),
            # F6 (120+1)*84 = 10164
            nn.Linear(in_features=120,
                      out_features=84),
            nn.Tanh(),
            # OUTPUT (84+1)*10 = 850
            nn.Linear(in_features=84,
                      out_features=10),
            nn.Softmax(dim=1)
        )

        # Total number of parameters = 123,412
        # 156+2,416+48,120+10,164+850 = 61,706
        # backpropagation 61,706

    def forward(self, img):
        x = self.conv_layers(img)
        x = x.view(-1, 16 * 5 * 5)
        output = self.fc_layers(x)

        return output


class CustomMLP(nn.Module):
    """ Your custom MLP model

        - Note that the number of model parameters should be about the same
          with LeNet-5
    """

    def __init__(self):
        super().__init__()

        self.layer1 = nn.Sequential(
            # L1 (28*28+1)*64 = 50240
            nn.Linear(28 * 28, 64),
            nn.Dropout(p=0.5),
            nn.ReLU()
        )

        self.layer2 = nn.Sequential(
            # L2 (64+1)*64 = 4160
            nn.Linear(64, 64),
            nn.Dropout(p=0.5),
            nn.ReLU()
        )

        self.layer3 = nn.Sequential(
            # L3 (64+1)*64 = 4160
            nn.Linear(64, 64),
            nn.Dropout(p=0.5),
            nn.ReLU()
        )

        self.layer4 = nn.Sequential(
            # L4 (64+1)*32 = 2080
            nn.Linear(64, 32),
            nn.Dropout(p=0.5),
            nn.ReLU()
        )

        self.layer5 = nn.Sequential(
            # L5 (32+1)*16 = 528
            nn.Linear(32, 16),
            nn.Dropout(p=0.5),
            nn.ReLU()
        )

        self.layer6 = nn.Sequential(
            # L6 (16+1)*16 = 272
            nn.Linear(16, 16),
            nn.Dropout(p=0.5),
            nn.ReLU()
        )

        self.layer7 = nn.Sequential(
            # L7 (16+1)*10 = 170
            nn.Linear(16, 10),
            nn.Dropout(p=0.5),
            nn.ReLU()
        )

        self.layer8 = nn.Sequential(
            # L8 (10+1)*10 = 110
            nn.Linear(10, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, img):
        x = img.view(-1, 28 * 28)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        output = self.layer8(x)

        return output

    # Total number of parameters = 123,440
    # 50,240+4,160+4,160+2,080+528+272+170+110 = 61,720
    # backpropagation 61,720


class RegularizedLenet5(nn.Module):

    def __init__(self):
        super(RegularizedLenet5, self).__init__()

        self.conv_layers = nn.Sequential(
            # C1 (5*5*1+1)*6 = 156
            nn.Conv2d(in_channels=1,
                      out_channels=6,
                      kernel_size=5,
                      stride=1,
                      padding=2,
                      bias=True),
            nn.ReLU(),
            nn.Dropout2d(0.5),
            # S2
            nn.MaxPool2d(kernel_size=2,
                         stride=2),
            # C3 (5*5*6+1)*16 = 2416
            nn.Conv2d(in_channels=6,
                      out_channels=16,
                      kernel_size=5,
                      stride=1,
                      padding=0,
                      bias=True),
            nn.ReLU(),
            nn.Dropout2d(0.5),
            # S4
            nn.MaxPool2d(kernel_size=2,
                         stride=2)
        )
        self.fc_layers = nn.Sequential(
            # C5 (16*5*5+1)*120 = 48120
            nn.Linear(in_features=16 * 5 * 5,
                      out_features=120),
            nn.ReLU(),
            nn.Dropout2d(0.5),
            # F6 (120+1)*84 = 10164
            nn.Linear(in_features=120,
                      out_features=84),
            nn.ReLU(),
            nn.Dropout2d(0.5),
            # OUTPUT (84+1)*10 = 850
            nn.Linear(in_features=84,
                      out_features=10),
            nn.Softmax(dim=1)
        )

        # Total number of parameters = 123,412
        # 156+2,416+48,120+10,164+850 = 61,706
        # backpropagation 61,706

    def forward(self, img):
        x = self.conv_layers(img)
        x = x.view(-1, 16 * 5 * 5)
        output = self.fc_layers(x)

        return output
