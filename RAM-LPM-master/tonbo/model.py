"""
This module contains models for image classification used in experiments
in scripts.

"""
import pdb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

import torchex.nn as exnn
import torchvision

from .modules import (
    action_network,
    baseline_network,
    core_network,
    glimpse_network,
    location_network,
    retina_polar,
    CNN_in_polar_coords,
    retina_rectangular,
)
from tonbo.resnet_polar import ResNet, BasicBlock


class RecurrentAttention(nn.Module):
    """
    A Recurrent Model of Visual Attention (RAM) [1].
    RAM is a recurrent neural network that processes
    inputs sequentially, attending to different locations
    within the image one at a time, and incrementally
    combining information from these fixations to build
    up a dynamic internal representation of the image.
    This is a modification based on the following paper:
    Minh et. al., https://arxiv.org/abs/1406.6247
    """

    def __init__(
        self, g, k, s, c, h_g, h_l, std, hidden_size, num_classes, sampling_interval,
    ):
        """
        Initialize the recurrent attention model and its
        different components.
        Args
        ----
        - g: size of the square patches in the glimpses extracted
          by the retina.
        - k: number of patches to extract per glimpse. ignored if glimpse_polar==True.
        - s: scaling factor that controls the size of successive patches. ignored if glimpse_polar==True.
        - c: number of channels in each image.
        - h_g: hidden layer size of the fc layer for `phi`.
        - h_l: hidden layer size of the fc layer for `l`.
        - std: standard deviation of the Gaussian policy.
        - num_classes: number of classes in the dataset.
        """
        super(RecurrentAttention, self).__init__()
        self.std = std
        self.glimpse_size = g
        self.hidden_size = hidden_size
        self.retina = retina_rectangular(sampling_interval, g, k=k, s=s)
        self.glimpse_network = glimpse_network(h_g, h_l)
        self.rnn = core_network(hidden_size)
        self.locator = location_network(hidden_size, 2, std)
        self.classifier = action_network(hidden_size, num_classes)
        self.baseliner = baseline_network(hidden_size, 1)

    def reset(self, batch_size, device):
        """
        Initialize the hidden state of the core network
        and the location vector.

        This is called once every time a new minibatch
        `x` is introduced.
        """
        h_t = torch.zeros(batch_size, self.hidden_size).to(device)
        l_t = torch.Tensor(batch_size, 2).uniform_(-1, 1).to(device)
        return h_t, l_t

    def forward(self, x, l_t_prev, h_t_prev, last=False):
        """
        Run the recurrent attention model for 1 timestep
        on the minibatch of images `x`.
        Args
        ----
        - x: a 4D Tensor of shape (B, H, W, C). The minibatch
          of images.
        - l_t_prev: a 2D tensor of shape (B, 2). The location vector
          containing the glimpse coordinates [x, y] for the previous
          timestep `t-1`.
        - h_t_prev: a 2D tensor of shape (B, hidden_size). The hidden
          state vector for the previous timestep `t-1`.
        - last: a bool indicating whether this is the last timestep.
          If True, the action network returns an output probability
          vector over the classes and the baseline `b_t` for the
          current timestep `t`. Else, the core network returns the
          hidden state vector for the next timestep `t+1` and the
          location vector for the next timestep `t+1`.
        Returns
        -------
        - h_t: a 2D tensor of shape (B, hidden_size). The hidden
          state vector for the current timestep `t`.
        - mu: a 2D tensor of shape (B, 2). The mean that parametrizes
          the Gaussian policy.
        - l_t: a 2D tensor of shape (B, 2). The location vector
          containing the glimpse coordinates [x, y] for the
          current timestep `t`.
        - b_t: a vector of length (B,). The baseline for the
          current time step `t`.
        - log_probas: a 2D tensor of shape (B, num_classes). The
          output log probability vector over the classes.
        - log_pi: a vector of length (B,).
        """
        fov = self.retina(x, l_t_prev)
        fov_flattened = fov.view(fov.shape[0], -1)
        g_t = self.glimpse_network(fov_flattened, l_t_prev)
        h_t = self.rnn(g_t, h_t_prev)
        b_t = self.baseliner(h_t).squeeze()
        mu, l_t = self.locator(h_t)

        # we assume both dimensions are independent
        # 1. pdf of the joint is the product of the pdfs
        # 2. log of the product is the sum of the logs
        log_pi = Normal(mu, self.std).log_prob(l_t)
        log_pi = torch.sum(log_pi, dim=1)

        if last:
            log_probas = self.classifier(h_t)
            return h_t, l_t, b_t, log_probas, log_pi

        return h_t, l_t, b_t, log_pi


class RAM_LPM(nn.Module):
    """
    A Recurrent Model of Visual Attention (RAM) [1].
    RAM is a recurrent neural network that processes
    inputs sequentially, attending to different locations
    within the image one at a time, and incrementally
    combining information from these fixations to build
    up a dynamic internal representation of the image.
    This is a modification based on the following paper:
    Minh et. al., https://arxiv.org/abs/1406.6247
    """

    def __init__(
        self,
        std,
        num_classes,
        kernel_sizes_conv2d,
        kernel_sizes_pool,
        strides_pool,
        kernel_dims,
        kernel_sizes_conv2d_where,
        kernel_sizes_pool_where,
        strides_pool_where,
        kernel_dims_where,
        r_min,
        r_max,
        H,
        W,
        hidden_what,
        hidden_where,
        upsampling_factor_r,
        upsampling_factor_theta,
        log_r,
        H_linear,
        resnet,
    ):
        """
        Initialize the recurrent attention model and its
        different components.
        Args
        ----
        - h_g: hidden layer size of the fc layer for `phi`.
        - h_l: hidden layer size of the fc layer for `l`.
        - std: standard deviation of the Gaussian policy.
        - hidden_sizes: a list of hidden sizes of the core rnns. Usually
        two rnns are used. The first one is for the what pathway, and
        the second one is for where pathway.
        - num_classes: number of classes in the dataset.
        - kernel_sizes_conv2d: kernel sizes for conv in glimpse layer.
        - kernel_sizes_pool: kernel sizes for pooling in glimpse layer.
        - kernel_dims: the dims of inputs and ouputs for conv in glimpse
        layer.
        - r_min (float): radius of the inner circle of the FOV.
        - r_max (float): radius of the outer circle of the FOV.
        - H (int): The height of the tensor mapped from the retina.
        - W (int): The height of the tensor mapped from the retina.
        """
        super(RAM_LPM, self).__init__()
        self.std = std
        self.hidden_what = hidden_what
        self.hidden_where = hidden_where
        self.retina = retina_polar(
            r_min,
            r_max,
            H,
            W,
            upsampling_factor_r,
            upsampling_factor_theta,
            log_r,
            H_linear,
        )

        h_convlstm = H + H_linear
        w_convlstm = W
        # what path
        if resnet:
            self.cnn_what = ResNet(BasicBlock, [2, 2, 2, 2])
            lstm_in = 514
        else:
            self.cnn_what = CNN_in_polar_coords(
                kernel_sizes_conv2d, kernel_sizes_pool, kernel_dims, strides_pool
            )
            for conv, pool in zip(
                kernel_sizes_conv2d, kernel_sizes_pool
            ):  # todo: strides shoud be the same as pool.
                h_convlstm = h_convlstm // pool[0]
                w_convlstm = w_convlstm // pool[1]
            lstm_in = h_convlstm * w_convlstm * kernel_dims[-1] + 2

        self.rnn_what = nn.LSTMCell(lstm_in, hidden_what)  # core_network(hidden_what)
        self.bn_what = nn.BatchNorm1d(hidden_what, momentum=0.01)
        self.fc_before_classifier = nn.Linear(hidden_what, 1024)
        self.bn_before_classifier = nn.BatchNorm1d(1024, momentum=0.01)
        self.classifier = action_network(1024, num_classes)
        self.baseliner = baseline_network(hidden_what, 1)

        # where path
        self.cnn_where = CNN_in_polar_coords(
            kernel_sizes_conv2d_where,
            kernel_sizes_pool_where,
            kernel_dims_where,
            strides_pool_where,
            pool_type="avg",
        )
        h_convlstm_where = H + H_linear
        w_convlstm_where = W
        for conv, pool in zip(
            kernel_sizes_conv2d_where, kernel_sizes_pool_where
        ):  # todo: strides shoud be the same as pool.
            h_convlstm_where = h_convlstm_where // pool[0]
            w_convlstm_where = w_convlstm_where // pool[1]
        self.rnn_where = nn.LSTMCell(
            h_convlstm_where * w_convlstm_where * kernel_dims_where[-1] + 2,
            hidden_where,
        )
        self.bn_where = nn.BatchNorm1d(hidden_where, momentum=0.01)
        self.locator = location_network(hidden_where, 2, std)

    def reset(self, batch_size, device):
        """
        Initialize the hidden state of the core network
        and the location vector.

        This is called once every time a new minibatch
        `x` is introduced.
        """
        # h_t = [self.rnn_what.init_hidden(batch_size, device), torch.zeros(batch_size, self.hidden_where).to(device)]
        h_t = [
            [
                torch.zeros(batch_size, self.hidden_what).to(device),
                torch.zeros(batch_size, self.hidden_what).to(device),
            ],
            [
                torch.zeros(batch_size, self.hidden_where).to(device),
                torch.zeros(batch_size, self.hidden_where).to(device),
            ],
        ]
        l_t = torch.Tensor(batch_size, 2).uniform_(-0.25, 0.25).to(device)
        return h_t, l_t

    def forward(self, x, l_t_prev, h_t_prev, last=False):
        """
        Run the recurrent attention model for 1 timestep
        on the minibatch of images `x`.
        Args
        ----
        - x: a 4D Tensor of shape (B, H, W, C). The minibatch
          of images.
        - l_t_prev: a 2D tensor of shape (B, 2). The location vector
          containing the glimpse coordinates [x, y] for the previous
          timestep `t-1`.
        - h_t_prev: a 2D tensor of shape (B, hidden_size). The hidden
          state vector for the previous timestep `t-1`.
        - last: a bool indicating whether this is the last timestep.
          If True, the action network returns an output probability
          vector over the classes and the baseline `b_t` for the
          current timestep `t`. Else, the core network returns the
          hidden state vector for the next timestep `t+1` and the
          location vector for the next timestep `t+1`.
        Returns
        -------
        - h_t: a 2D tensor of shape (B, hidden_size). The hidden
          state vector for the current timestep `t`.
        - mu: a 2D tensor of shape (B, 2). The mean that parametrizes
          the Gaussian policy.
        - l_t: a 2D tensor of shape (B, 2). The location vector
          containing the glimpse coordinates [x, y] for the
          current timestep `t`.
        - b_t: a vector of length (B,). The baseline for the
          current time step `t`.
        - log_probas: a 2D tensor of shape (B, num_classes). The
          output log probability vector over the classes.
        - log_pi: a vector of length (B,).
        """
        g_t = self.retina(x, l_t_prev)

        # what path
        cnn_what = self.cnn_what(g_t)
        flatten_what = cnn_what.view(cnn_what.size(0), -1)
        flatten_what = torch.cat((flatten_what, l_t_prev), 1)
        h_t_what, c_t_what = self.rnn_what(flatten_what, h_t_prev[0])
        h_t_what_normalized = self.bn_what(h_t_what)
        b_t = self.baseliner(h_t_what_normalized).squeeze()

        # where path
        cnn_where = self.cnn_where(g_t)
        flatten_where = cnn_where.view(cnn_where.size(0), -1)
        flatten_where = torch.cat((flatten_where, l_t_prev), 1)
        h_t_where, c_t_where = self.rnn_where(flatten_where, h_t_prev[1])
        h_t = [[h_t_what, c_t_what], [h_t_where, c_t_where]]
        h_t_where_normalized = self.bn_where(h_t_where)
        mu, l_t = self.locator(h_t_where_normalized)
        # we assume both dimensions are independent
        # 1. pdf of the joint is the product of the pdfs
        # 2. log of the product is the sum of the logs
        log_pi = Normal(mu, self.std).log_prob(l_t)
        log_pi = torch.sum(log_pi, dim=1)

        if last:
            net = self.fc_before_classifier(h_t_what_normalized)
            net = F.relu(net)
            net = self.bn_before_classifier(net)
            log_probas = self.classifier(net)
            return h_t, l_t, b_t, log_probas, log_pi

        return h_t, l_t, b_t, log_pi


class SqueezeChannel(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SqueezeChannel, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.linear = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B * W * H, C)
        x = self.linear(x)
        x = x.view(B, self.out_channels, H, W)
        return x


class Sort(torch.nn.Module):
    def __init__(self, dim):
        super(Sort, self).__init__()
        self.dim = dim

    def forward(self, x):
        x, indices = torch.sort(x, self.dim)
        return x


class FFTCNN(torch.nn.Module):
    def __init__(self):
        super(FFTCNN, self).__init__()

        self.net = torch.nn.Sequential(
            exnn.PeriodicPad2d(5, 5, 5, 5),
            torch.nn.Conv2d(1, 10, 5),
            torch.nn.MaxPool2d((4, 1), (5, 1)),
            torch.nn.ReLU(),
            exnn.PeriodicPad2d(5, 5, 5, 5),
            torch.nn.Conv2d(10, 20, 3),
            torch.nn.MaxPool2d((3, 1), (1, 1)),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            SqueezeChannel(20, 1),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            Sort(3),
            exnn.Flatten(),
            torch.nn.Linear(552, 10),
            torch.nn.LogSoftmax(),
        )

    def forward(self, x):
        x = torch.fft(x, 2)
        x = self.net(x)
        x = torch.ifft(x, 2)
        return x


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=5),
            nn.Dropout2d(),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )
        self.linear = nn.Sequential(
            exnn.Flatten(),
            exnn.Linear(50),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(50, 10),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.linear(x)
        return F.log_softmax(x, dim=1)


class CNN2FC2V2(torch.nn.Module):
    def __init__(self):
        super(CNN2FC2V2, self).__init__()
        self.net = torch.nn.Sequential(
            exnn.PeriodicPad2d(5, 5, 5, 5),
            torch.nn.Conv2d(1, 10, 5),
            torch.nn.MaxPool2d((4, 1), (5, 1)),
            torch.nn.ReLU(),
            exnn.PeriodicPad2d(5, 5, 5, 5),
            torch.nn.Conv2d(10, 20, 3),
            torch.nn.MaxPool2d((3, 1), (1, 1)),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            SqueezeChannel(20, 1),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            Sort(3),
            exnn.Flatten(),
            nn.Linear(546, 10),
            torch.nn.LogSoftmax(),
        )

    def forward(self, x):
        return self.net(x)


class CNN2FC2(torch.nn.Module):
    def __init__(self):
        super(CNN2FC2, self).__init__()
        # C(10, 5),MP(2),R,C(20, 5),D,MP(2),R,OFC(100),R,D,FC(10),LSMAX(1)
        self.net = torch.nn.Sequential(
            exnn.PeriodicPad2d(5, 5, 5, 5),
            torch.nn.Conv2d(1, 10, 5),
            torch.nn.MaxPool2d(5, 4),
            torch.nn.ReLU(),
            exnn.PeriodicPad2d(5, 5, 5, 5),
            torch.nn.Conv2d(10, 20, 5),
            torch.nn.Dropout(0.5),
            torch.nn.MaxPool2d(5, 3),
            torch.nn.ReLU(),
            exnn.Flatten(),
            torch.nn.Linear(320, 100),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(100, 10),
            torch.nn.LogSoftmax(),
        )

    def forward(self, x):
        return self.net(x)


class PolarConv2D(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super(PolarConv2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

        self.b = torch.nn.Parameter(torch.Tensor(in_channels, kernel_size, kernel_size))
        self.w = torch.nn.Parameter(torch.Tensor(in_channels, kernel_size, kernel_size))
        if self.bias is not None:
            self.b.data.uniform_(-0.1, 0.1)
        self.w.data.uniform_(-0.1, 0.1)

    def _transform2polar(self, img, order=1):
        """
        Transform img to its polar coordinate representation.

        order: int, default 1
        Specify the spline interpolation order.
        High orders may be slow for large images.
        """
        # max_radius is the length of the diagonal
        # from a corner to the mid-point of img.
        _img = img[0]
        max_radius = 0.5 * np.linalg.norm(_img.shape)

        def transform(coords):
            # Put coord[1] in the interval, [-pi, pi]
            theta = 2 * np.pi * coords[1] / (_img.shape[1] - 1.0)

            radius = max_radius * coords[0] / _img.shape[0]

            i = 0.5 * _img.shape[0] - radius * np.sin(theta)
            j = radius * np.cos(theta) + 0.5 * _img.shape[1]
            return i, j

        _img = _img.data.numpy()
        polar = geometric_transform(_img, transform, order=order)
        rads = max_radius * np.linspace(0, 1, _img.shape[0])
        angs = np.linspace(0, 2 * np.pi, _img.shape[1])
        img[0, :, :] = torch.FloatTensor(polar)
        return img, (rads, angs)

    def forward(self, x):
        """
        https://discuss.pytorch.org/t/custom-pooling-conv-layer/18916/3
        https://discuss.pytorch.org/t/convolution-that-takes-a-function-as-kernel/25588/7
        """
        x_unf = x.unfold(2, self.kernel_size, self.stride).unfold(
            3, self.kernel_size, self.stride
        )
        B, C, W, H, K1, K2 = x_unf.shape
        WC, WK1, WK2 = self.w.shape

        x_unf.matmul(self.w.transpose())

        return x


class LogPolarConv(nn.Module):
    def __init__(
        self,
        # num_classes,
        kernel_sizes_conv2d,
        kernel_sizes_pool,
        strides_pool,
        kernel_dims,
        r_min,
        r_max,
        H,
        W,
        upsampling_factor_r,
        upsampling_factor_theta,
        hidden_what=128,
    ):
        super(LogPolarConv, self).__init__()
        self.retina = retina_polar(
            r_min, r_max, H, W, upsampling_factor_r, upsampling_factor_theta
        )
        self.cnn = CNN_in_polar_coords(
            kernel_sizes_conv2d, kernel_sizes_pool, kernel_dims, strides_pool
        )
        self.flatten = exnn.Flatten()
        self.action = action_network(hidden_what, 10)
        self.register_buffer("center", torch.zeros([1, 2]))

    def forward(self, x):
        x = self.retina(x, self.center.repeat(x.shape[0], 1))
        x = self.cnn(x)
        x = self.flatten(x)
        logits = self.action(x[0])  # only uses the what info.
        return logits


class Res18(nn.Module):
    def __init__(self, num_classes=10):
        super(Res18, self).__init__()
        self.resmodel = torchvision.models.resnet18(num_classes=num_classes)

    def forward(self, x):
        x = self.resmodel(x)
        x = F.log_softmax(x, dim=1)
        return x


if __name__ == "__main__":
    x = torch.rand(5, 1, 28, 28)
    m = FFTCNN()
    y = m(x)
    print(y.shape)
    """
    x = torch.rand(5, 1, 128, 128)
    m = CircleKernelConv2D(1, 3)
    m(x)
    """
