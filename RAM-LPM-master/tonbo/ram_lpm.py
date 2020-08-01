import torch
from torch.distributions import Normal

from tonbo.modules import (
    retina_polar,
    action_network,
    baseline_network,
    CNN_in_polar_coords,
    location_network,
)


class RAM_LPM(torch.nn.Module):
    """
    A Recurrent Visual Attention Model with Log Polar Mapping (RAM-LPM).
    RAM-LPM is a recurrent neural network that processes
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
        mc_sample_num,
        num_glimpses,
        device,
        noise_level=None,
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
        self.cnn_what = CNN_in_polar_coords(
            kernel_sizes_conv2d, kernel_sizes_pool, kernel_dims, strides_pool
        )
        for conv, pool in zip(
            kernel_sizes_conv2d, kernel_sizes_pool
        ):  # todo: strides shoud be the same as pool.
            h_convlstm = h_convlstm // pool[0]
            w_convlstm = w_convlstm // pool[1]
        lstm_in = h_convlstm * w_convlstm * kernel_dims[-1] + 2

        self.rnn_what = torch.nn.LSTMCell(
            lstm_in, hidden_what
        )  # core_network(hidden_what)
        self.bn_what = torch.nn.BatchNorm1d(hidden_what, momentum=0.01)
        self.fc_before_classifier = torch.nn.Linear(hidden_what, 1024)
        self.bn_before_classifier = torch.nn.BatchNorm1d(hidden_what, momentum=0.01)
        self.classifier = action_network(hidden_what, num_classes)
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
        self.rnn_where = torch.nn.LSTMCell(
            h_convlstm_where * w_convlstm_where * kernel_dims_where[-1] + 2,
            hidden_where,
        )
        self.bn_where = torch.nn.BatchNorm1d(hidden_where, momentum=0.01)
        self.locator = location_network(hidden_where, 2, std)
        self.mc_sample_num = mc_sample_num
        self.num_glimpses = num_glimpses
        self.device = device
        self.noise_level = noise_level
        if noise_level:
            self.noise_added = torch.zeros([H + H_linear, W])
        else:
            self.noise_added = None

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

    def forward_one_t(self, x, l_t_prev, h_t_prev, last=False):
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
        if self.noise_added is not None:
            noise = (
                torch.stack(
                    [
                        self.noise_added.normal_(std=self.noise_level[0]),
                        self.noise_added.normal_(std=self.noise_level[1]),
                        self.noise_added.normal_(std=self.noise_level[2]),
                    ]
                )
                .unsqueeze(0)
                .expand(g_t.shape)
            )
            g_t = g_t + noise.to(self.device)

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
            # net = self.fc_before_classifier(h_t_what_normalized)
            # net = F.relu(net)
            net = self.bn_before_classifier(h_t_what_normalized)
            log_probas = self.classifier(net)
            return h_t, l_t, b_t, log_probas, log_pi

        return h_t, l_t, b_t, log_pi

    def forward(self, x):
        if self.training:
            h_t, l_t = self.reset(x.shape[0], self.device)
            locs = []
            log_pis = []
            baselines = []

            for i in range(self.num_glimpses - 1):
                h_t, l_t, b_t, log_pi = self.forward_one_t(x, l_t, h_t, last=False)
                locs.append(l_t)
                log_pis.append(log_pi)
                baselines.append(b_t)
            h_t, l_t, b_t, log_probas, log_pi = self.forward_one_t(
                x, l_t, h_t, last=True
            )
            locs = torch.stack(locs, dim=1)
            baselines = torch.stack(baselines, dim=1)
            log_pis = torch.stack(log_pis, dim=1)

            return log_probas, locs, log_pis, baselines

        else:
            log_probas_list = []
            for sample in range(self.mc_sample_num):
                h_t, l_t = self.reset(x.shape[0], self.device)
                for i in range(self.num_glimpses - 1):
                    h_t, l_t, b_t, log_pi = self.forward_one_t(x, l_t, h_t, last=False)
                h_t, l_t, b_t, log_probas, log_pi = self.forward_one_t(
                    x, l_t, h_t, last=True
                )
                log_probas_list.append(log_probas)
            return torch.mean(torch.stack(log_probas_list), dim=0)
