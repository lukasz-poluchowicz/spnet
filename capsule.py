import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

cuda = torch.cuda.is_available()

default_device = torch.device('cuda' if cuda else 'cpu')


class PrimaryLayer:
    """Layer of primary capsules, which take grid of pixels as input."""

    def __init__(self, *, kernel_size=None, stride=None, src_num_channels=None, device=default_device, state_dict=None):
        """
        If ``state_dict`` is provided, then the layer is recreated from this ``state_dict`` and the other parameters are
        ignored. If ``state_dict`` is ``None``, then the layer is created according to the other parameters.

        Args:
            kernel_size (int): Size of the input grid of pixels (``kernel_size`` by ``kernel_size``).
            stride (int): Stride of the convolution.
            src_num_channels (int): Number of channels per pixel.
            device (torch.device): Device of this layer.
            state_dict: State dictionary from which the layer can be recreated.
        """
        if state_dict:
            self._kernel_size = state_dict['kernel_size']
            self._stride = state_dict['stride']
            self._src_num_channels = state_dict['src_num_channels']
            self._vec_sizes = state_dict['vec_sizes']
            self._vec_sizes_sum = sum(self._vec_sizes)
            self._linear = create_linear_from_state(state_dict['linear'])
            self.to(state_dict['device'])

        else:
            assert kernel_size is not None and stride is not None and src_num_channels is not None

            self._kernel_size = kernel_size
            self._stride = stride
            self._src_num_channels = src_num_channels
            self._vec_sizes = []
            self._vec_sizes_sum = 0
            self._linear = None
            self._device = device

    def state_dict(self):
        """
        Returns:
            State dictionary from which the layer can be recreated.
        """
        return {'kernel_size': self._kernel_size,
                'stride': self._stride,
                'src_num_channels': self._src_num_channels,
                'vec_sizes': self._vec_sizes,
                'linear': self._linear.state_dict(),
                'device': self._device}

    @property
    def kernel_size(self):
        return self._kernel_size

    @property
    def stride(self):
        return self._stride

    def input_size(self, output_grid_size):
        """
        Args:
            output_grid_size (sequence of ints): height and width of the output for which the input size is returned.

        Returns:
            Size of the input which produces the output of height and width equal to ``output_grid_size``.
        """
        return _input_grid_size(output_grid_size, self._kernel_size, self._stride) + (self._src_num_channels,)

    @property
    def vec_sizes(self):
        return self._vec_sizes

    @property
    def vec_sizes_sum(self):
        return self._vec_sizes_sum

    def add_cap_channels(self, vec_sizes):
        """
        Args:
            vec_sizes (list of ints): List of output vectors sizes of new capsules channels.
        """
        self._vec_sizes += vec_sizes

        in_features = self._kernel_size ** 2 * self._src_num_channels

        np_vs = np.array(self._vec_sizes)
        assert np_vs.shape == (len(self._vec_sizes),)
        self._vec_sizes_sum = np_vs.sum()
        # Sum vector sizes and one activation per capsule channel.
        out_features = self._vec_sizes_sum + len(self._vec_sizes)

        self._linear = get_expanded_layer(self._linear, in_features, out_features, device=self._device)

    def __call__(self, inputs):
        """
        Args:
            inputs (tensor): Layer input of shape (num_samples, input_height, input_width, num_channels).

        Returns:
            Layer output of shape (num_samples, output_height, output_width, depth), where depth is the sum of vector
            sizes and activations of this layer capsule channels.
        """
        num_samples = len(inputs)
        grid_size = _output_grid_size(inputs.shape[1:3], self._kernel_size, self._stride)

        _assert_inputs(inputs, self._kernel_size, self._src_num_channels)

        outputs = torch.zeros(num_samples, *grid_size, self._linear.out_features, device=self._device)

        for y in range(grid_size[0]):
            ys = y * self._stride
            for x in range(grid_size[1]):
                xs = x * self._stride
                cut = inputs[:, ys:(ys + self._kernel_size), xs:(xs + self._kernel_size), :].reshape(num_samples, -1)
                outputs[:, y, x, :] = self._linear(cut)

        return outputs

    def parameters(self):
        """
        Returns:
            Parameters for optimizer.
        """
        return self._linear.parameters()

    def to(self, device):
        """
        Args:
            device (torch.device): New device of this layer.
        """
        self._device = device
        self._linear.to(device)


class Layer:
    """Layer of capsules, which take output of grid of capsules as input."""

    def __init__(self, *, kernel_size=None, stride=None, src_vec_sizes=None, device=default_device, state_dict=None):
        """
        If ``state_dict`` is provided, then the layer is recreated from this ``state_dict`` and the other parameters are
        ignored. If ``state_dict`` is ``None``, then the layer is created according to the other parameters.

        Args:
            kernel_size (int): Size of the input grid of capsules (``kernel_size`` by ``kernel_size``).
            stride (int): Stride of the convolution.
            src_vec_sizes (list of ints): List of output vector sizes of input capsule channels.
            device (torch.device): Device of this layer.
            state_dict: State dictionary from which the layer can be recreated.
        """
        if state_dict:
            self._kernel_size = state_dict['kernel_size']
            self._stride = state_dict['stride']
            self._src_vec_sizes = state_dict['src_vec_sizes']
            self._set_dvec_sizes(state_dict['dvec_sizes'])

            self._src_linears = [[[create_linear_from_state(linear_state)
                                   for linear_state in yxs] for yxs in ys] for ys in state_dict['src_linears']]

            self._activation_linears = [create_linear_from_state(linear_state)
                                        for linear_state in state_dict['activation_linears']]

            self._bonding_logits = state_dict['bonding_logits']
            self.to(state_dict['device'])

        else:
            assert kernel_size is not None and stride is not None and src_vec_sizes is not None

            self._kernel_size = kernel_size
            self._stride = stride
            self._src_vec_sizes = list(src_vec_sizes)
            self._dvec_sizes = []
            self._vec_sizes = []
            self._vec_sizes_sum = 0
            self._src_linears = [[[None] * len(src_vec_sizes) for _ in range(kernel_size)] for _ in range(kernel_size)]
            self._activation_linears = []
            self._bonding_logits = None
            self._device = device

    def state_dict(self):
        """
        Returns:
            State dictionary from which the layer can be recreated.
        """
        return {'kernel_size': self._kernel_size,
                'stride': self._stride,
                'src_vec_sizes': self._src_vec_sizes,
                'dvec_sizes': self._dvec_sizes,
                'src_linears': [[[linear.state_dict() for linear in yxs] for yxs in ys] for ys in self._src_linears],
                'activation_linears': [linear.state_dict() for linear in self._activation_linears],
                'bonding_logits': self._bonding_logits,
                'device': self._device}

    @property
    def kernel_size(self):
        return self._kernel_size

    @property
    def stride(self):
        return self._stride

    def input_size(self, output_grid_size):
        """
        Args:
            output_grid_size (sequence of ints): height and width of the output for which the input size is returned.

        Returns:
            Size of the input which produces the output of height and width equal to ``output_grid_size``.
        """
        input_depth = sum(self._src_vec_sizes) + len(self._src_vec_sizes)
        return _input_grid_size(output_grid_size, self._kernel_size, self._stride) + (input_depth,)

    def _set_dvec_sizes(self, dvec_sizes):
        self._dvec_sizes = dvec_sizes

        np_dvs = np.array(dvec_sizes)
        assert np_dvs.shape == (len(dvec_sizes), 2)
        # For each capsule channel: sum the size of rba and non-rba vector.
        vs = np_dvs.sum(axis=1)

        self._vec_sizes = list(vs)
        self._vec_sizes_sum = vs.sum()

    @property
    def dvec_sizes(self):
        return self._dvec_sizes

    @property
    def vec_sizes(self):
        return self._vec_sizes

    @property
    def vec_sizes_sum(self):
        return self._vec_sizes_sum

    def expand_src_vec_sizes(self, src_vec_sizes):
        """
        Expands input depth of this layer.

        Args:
            src_vec_sizes (list of ints): New list of output vector sizes of input capsule channels. Must be equal to
            the current list of output vector sizes with the vector sizes of new capsule channels appended to the end.
        """
        assert len(self._src_vec_sizes) < len(src_vec_sizes)
        for a, b in zip(self._src_vec_sizes, src_vec_sizes):
            assert a == b

        for ky in range(self._kernel_size):
            for kx in range(self._kernel_size):
                for src_vec_size in src_vec_sizes[len(self._src_vec_sizes):]:
                    linear = _create_linear(src_vec_size, self._vec_sizes_sum, device=self._device)
                    self._src_linears[ky][kx].append(linear)

                assert len(self._src_linears[ky][kx]) == len(src_vec_sizes)

        self._src_vec_sizes = list(src_vec_sizes)

        self._expand_bonding_logits()

    def _expand_bonding_logits(self):
        old_bonding_logits = self._bonding_logits
        self._bonding_logits = torch.randn(len(self._src_vec_sizes), len(self._dvec_sizes),
                                           device=self._device, requires_grad=True)

        if old_bonding_logits is not None:
            old_shape = old_bonding_logits.shape
            new_shape = self._bonding_logits.shape

            assert (old_shape[0] < new_shape[0] and old_shape[1] == new_shape[1]) or \
                   (old_shape[0] == new_shape[0] and old_shape[1] < new_shape[1])

            self._bonding_logits.data[:old_shape[0], :old_shape[1]] = old_bonding_logits

    def add_cap_channels(self, dvec_sizes):
        """
        Expands output depth of this layer.

        Args:
            dvec_sizes: List of tuples (``rba_size``, ``non_rba_size``) of new capsules channels. ``rba_size`` is the
            size of capsule channel output used to calculate activation.
        """
        self._set_dvec_sizes(self._dvec_sizes + dvec_sizes)

        for ky in range(self._kernel_size):
            for kx in range(self._kernel_size):
                for src_vec_index, src_vec_size in enumerate(self._src_vec_sizes):
                    src_linear = self._src_linears[ky][kx][src_vec_index]
                    src_linear = get_expanded_layer(src_linear, src_vec_size, self._vec_sizes_sum, device=self._device)
                    self._src_linears[ky][kx][src_vec_index] = src_linear

        self._expand_bonding_logits()

        for rba_size, non_rba_size in dvec_sizes:
            self._activation_linears.append(_create_linear(rba_size, 1, device=self._device))

        assert len(self._activation_linears) == len(self._dvec_sizes)

    def __call__(self, inputs):
        """
        Args:
            inputs (tensor): Layer input of shape (num_samples, input_height, input_width, input_depth), where depth is
                the sum of vector sizes and activations of input capsule channels.

        Returns:
            Layer output of shape (num_samples, output_height, output_width, output_depth), where depth is the sum of
            vector sizes and activations of this layer capsule channels.
        """
        num_samples = len(inputs)
        grid_size = _output_grid_size(inputs.shape[1:3], self._kernel_size, self._stride)
        input_depth = self.input_size(grid_size)[2]

        _assert_inputs(inputs, self._kernel_size, input_depth)

        outputs = torch.zeros(num_samples, *grid_size, self._vec_sizes_sum + len(self._vec_sizes), device=self._device)
        num_votes_per_sample = self._kernel_size ** 2 * len(self._src_vec_sizes)

        bondings = torch.sigmoid(self._bonding_logits)
        yx_votes = torch.zeros(num_samples, num_votes_per_sample, self._vec_sizes_sum, device=self._device)

        for y in range(grid_size[0]):
            for x in range(grid_size[1]):
                self._calculate_yx_outputs(y, x, inputs, outputs, bondings, yx_votes)

        return outputs

    def _calculate_yx_outputs(self, y, x, inputs, outputs, bondings, votes):
        num_samples = len(inputs)
        votes_powers = torch.zeros(*votes.shape[:2], len(self._dvec_sizes), device=self._device)
        votes_index = 0

        for ky in range(self._kernel_size):
            for kx in range(self._kernel_size):
                linears = self._src_linears[ky][kx]
                kyx_inputs = inputs[:, y * self._stride + ky, x * self._stride + kx, :]

                inputs_offset = 0

                for (src_vec_index, src_vec_size), linear in zip(enumerate(self._src_vec_sizes), linears):
                    votes[:, votes_index, :] = linear(kyx_inputs[:, inputs_offset:(inputs_offset + src_vec_size)])
                    inputs_offset += src_vec_size

                    src_vec_activations = kyx_inputs[:, inputs_offset].view(num_samples, 1)
                    inputs_offset += 1

                    src_vec_bondings = bondings[src_vec_index].view(1, len(self._dvec_sizes))

                    votes_powers[:, votes_index, :] = torch.mm(src_vec_activations, src_vec_bondings)
                    votes_index += 1

                assert inputs_offset == kyx_inputs.shape[1]

        assert votes_index == votes.shape[1]

        votes_powers /= votes_powers.sum(dim=1, keepdim=True)

        vec_offset = 0

        for dvec_index, (rba_size, non_rba_size) in enumerate(self._dvec_sizes):
            both_size = rba_size + non_rba_size
            votes[:, :, vec_offset:(vec_offset + both_size)] *= votes_powers[:, :, dvec_index, None]
            vec_offset += both_size

        assert vec_offset == votes.shape[2]

        votes_sum = votes.sum(dim=1)
        assert votes_sum.shape == (num_samples, self._vec_sizes_sum)

        vec_offset = 0
        outputs_offset = 0

        for dvec_index, (rba_size, non_rba_size) in enumerate(self._dvec_sizes):
            rba_vec = outputs[:, y, x, outputs_offset:(outputs_offset + rba_size)]
            rba_vec[:] = votes_sum[:, vec_offset:(vec_offset + rba_size)]
            variances = (votes[:, :, vec_offset:(vec_offset + rba_size)] - rba_vec[:, None, :]) ** 2
            variances *= votes_powers[:, :, dvec_index, None]
            variances = variances.sum(dim=1)
            assert variances.shape == (num_samples, rba_size)
            vec_offset += rba_size
            outputs_offset += rba_size

            non_rba_vec = outputs[:, y, x, outputs_offset:(outputs_offset + non_rba_size)]
            non_rba_vec[:] = F.leaky_relu(votes_sum[:, vec_offset:(vec_offset + non_rba_size)])
            vec_offset += non_rba_size
            outputs_offset += non_rba_size

            activations = outputs[:, y, x, outputs_offset]
            activations[:] = F.leaky_relu(self._activation_linears[dvec_index](variances).squeeze())
            outputs_offset += 1

        assert vec_offset == votes.shape[2]
        assert outputs_offset == outputs.shape[3]

    def parameters(self):
        """
        Returns:
            Parameters for optimizer.
        """
        for ky in range(self._kernel_size):
            for kx in range(self._kernel_size):
                for linear in self._src_linears[ky][kx]:
                    for param in linear.parameters():
                        yield param

        for linear in self._activation_linears:
            for param in linear.parameters():
                yield param

        yield self._bonding_logits

    def to(self, device):
        """
        Args:
            device (torch.device): New device of this layer.
        """
        self._device = device

        for ys in self._src_linears:
            for yxs in ys:
                for linear in yxs:
                    linear.to(device)

        for linear in self._activation_linears:
            linear.to(device)

        self._bonding_logits = self._bonding_logits.to(device).detach().requires_grad_(True)


def _create_linear(in_features, out_features, device):
    linear = nn.Linear(in_features, out_features)
    linear.to(device)
    return linear


def _assert_inputs(inputs, kernel_size, input_depth):
    assert len(inputs.shape) == 4
    assert inputs.shape[1] >= kernel_size and inputs.shape[2] >= kernel_size
    assert inputs.shape[3] == input_depth


def _input_grid_size(output_grid_size, kernel_size, stride):
    out_size = np.array(output_grid_size)
    assert out_size.shape == (2,)

    return tuple(kernel_size + (out_size - 1) * stride)


def _output_grid_size(input_grid_size, kernel_size, stride):
    in_size = np.array(input_grid_size)
    assert in_size.shape == (2,)
    assert np.all((in_size - kernel_size) % stride == 0)

    return tuple((in_size - kernel_size) // stride + 1)


def input_grid_size_thru(layers, output_grid_size):
    """
    Args:
        layers: List of layers that processes input of returned size into output of given size.
        output_grid_size (sequence of ints): Height and width of the ``layers`` output for which the ``layers`` input
            height and width is returned.

    Returns:
        Height and width (tuple of ints) of the ``layers`` input which produces the ``layers`` output of height and
        width equal to ``output_grid_size``.
    """
    size = output_grid_size

    if layers:
        for layer in reversed(layers):
            size = _input_grid_size(size, layer.kernel_size, layer.stride)

    return size


def output_grid_size_thru(layers, input_grid_size):
    """
    Args:
        layers: List of layers that processes input of given size into output of returned size.
        input_grid_size (sequence of ints): Height and width of the ``layers`` input for which the ``layers`` output
            height and width is returned.

    Returns:
        Height and width (tuple of ints) of the ``layers`` output which is produced from the ``layers`` input of height
        and width equal to ``input_grid_size``.
    """
    size = input_grid_size

    if layers:
        for layer in layers:
            size = _output_grid_size(size, layer.kernel_size, layer.stride)

    return size


def get_expanded_layer(old_layer, new_in_features, new_out_features, device):
    """
    Args:
        old_layer: Weights of that layer will be copied to returned layer to the common part of both layers.
        new_in_features (int): Number of input features of returned layer. Must not be smaller than
            ``old_layer.in_features``.
        new_out_features (int): Number of output features of returned layer. Must not be smaller than
            ``old_layer.out_features``.
        device (torch.device): Device of returned layer.

    Returns:
        New bigger linear layer (with more input and/or output features than ``old_layer``) containing weights of
        ``old_layer`` for the common part of both new and old layers.
    """
    new_layer = _create_linear(new_in_features, new_out_features, device)

    if old_layer:
        assert old_layer.in_features <= new_in_features
        assert old_layer.out_features <= new_out_features
        assert old_layer.in_features + old_layer.out_features < new_in_features + new_out_features

        old_weight_size = old_layer.weight.size()
        new_layer.weight.data[:old_weight_size[0], :old_weight_size[1]] = old_layer.weight
        new_layer.bias.data[:old_layer.bias.size()[0]] = old_layer.bias

    return new_layer


def create_layer_from_state(state_dict):
    """
    Args:
        state_dict: State dictionary of capsule layer from which the layer is recreated.

    Returns:
        New capsule layer recreated from ``state_dict``.
    """
    return (Layer if 'bonding_logits' in state_dict else PrimaryLayer)(state_dict=state_dict)


def create_linear_from_state(state_dict):
    """
    Args:
        state_dict: State dictionary of linear layer from which the layer is recreated.

    Returns:
        New linear layer recreated from ``state_dict``.
    """
    out_features, in_features = state_dict['weight'].shape

    linear = nn.Linear(in_features, out_features)
    linear.load_state_dict(state_dict)

    return linear
