import torch
import torch.nn as nn

from monai.networks.blocks import Convolution


class MLTModel(nn.Module):
    def __init__(self, in_channels=2, dropout=0.1, hidden_dim=16, tasks=None, out_dim=None, kernel_size=5, strides=2, dem_feat_dim=3):
        super(MLTModel, self).__init__()
        self.conv1 = Convolution(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=16,
            adn_ordering="ADN",
            kernel_size=kernel_size,
            dropout=dropout,
            strides=1,
            conv_only=True
        )
        self.conv2 = Convolution(
            spatial_dims=2,
            in_channels=16,
            out_channels=32,
            adn_ordering="ADN",
            kernel_size=kernel_size,
            dropout=dropout,
            strides=strides,
            conv_only=True
        )
        self.conv3 = Convolution(
            spatial_dims=2,
            in_channels=32,
            out_channels=64,
            adn_ordering="ADN",
            kernel_size=kernel_size,
            dropout=dropout,
            strides=strides,
            conv_only=True
        )
        self.conv4 = Convolution(
            spatial_dims=2,
            in_channels=64,
            out_channels=64,
            adn_ordering="ADN",
            kernel_size=kernel_size,
            dropout=dropout,
            strides=strides,
            conv_only=True
        )

        self.mxpool = nn.MaxPool2d(4)
        self.mxpool2 = nn.MaxPool2d(2)
        self.task_layer_dict = nn.ModuleDict()
        self.act = nn.PReLU()
        self.dem_net = nn.Sequential(nn.Linear(3, 64), self.act)
        for task in tasks:
            self.task_layer_dict[task] = nn.Sequential(
                nn.Linear(64, 64),
                self.act,
                nn.Linear(64, out_dim[task]))
        self.tasks = tasks
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, Convolution):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, dem):
        out = self.mxpool(self.act(self.conv1(x)))
        out = self.mxpool(self.act(self.conv2(out)))
        out = self.mxpool2(self.act(self.conv3(out)))
        out = self.mxpool2(self.act(self.conv4(out)))
        # out = self.mxpool2(self.act(self.conv5(out)))
        out = nn.Flatten(start_dim=1)(out)
        # out = out + self.dem_net(dem.float())
        output_dict = {}
        for task in self.tasks:
            output_dict[task] = self.task_layer_dict[task](out).squeeze().float()
        return output_dict


class MLTModel2(nn.Module):
    def __init__(self, in_channels=2, dropout=0.1, hidden_dim=64, tasks=None, out_dim=None, kernel_size=5, strides=2, dem_feat_dim=3):
        super(MLTModel2, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=16,
            kernel_size=kernel_size,
            stride=strides
        )
        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=kernel_size,
            stride=strides
        )
        self.conv3 = nn.Conv2d(
            in_channels=32,
            out_channels=hidden_dim,
            kernel_size=kernel_size,
            stride=strides
        )
        self.conv4 = nn.Conv2d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=3,
            stride=1
        )


        self.mxpool = nn.MaxPool2d(4)
        self.mxpool2 = nn.MaxPool2d(2)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.act = nn.PReLU()
        self.task_layer_dict = nn.ModuleDict()
        self.skip = nn.Sequential(
            self.drop,
            nn.MaxPool2d(4),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=32,
                      kernel_size=kernel_size,
                      stride=strides),
            self.act,
            nn.MaxPool2d(4)
        )

        self.dem_net = nn.Sequential(nn.Linear(3, hidden_dim), self.act)
        for task in tasks:
            self.task_layer_dict[task] = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                self.act,
                nn.Linear(hidden_dim, hidden_dim),
                self.act,
                nn.Linear(hidden_dim, out_dim[task]))
        self.tasks = tasks
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, dem):
        out = self.mxpool(self.act(self.conv1(self.drop(x))))
        # out_skipped = self.skip(x)
        out = self.mxpool2(self.act(self.conv2(self.drop(out))))
        out = self.mxpool2(self.act(self.conv3(self.drop(out))))
        out = self.act(self.conv4(self.drop(out)))
        # out = self.mxpool2(self.act(self.conv5(out)))
        out = nn.Flatten(start_dim=1)(out)
        # out = out + self.dem_net(dem.float())
        output_dict = {}
        for task in self.tasks:
            output_dict[task] = self.task_layer_dict[task](out).squeeze().float()
        return output_dict