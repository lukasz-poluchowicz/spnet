import math

import numpy as np
import torch
import torch.nn.functional as F

import capsule as cps

_default_learning_rate = 0.01

_decoder_vec_size_multiplier = 2.0
_decoder_vec_sizes_sum_multiplier = 1.5
_decoder_linear_size_multiplier = 1.0


class Trainer:
    """Capsule layer trainer."""

    def __init__(self, *, layer=None, decoder_kernel_size=None, learning_rate=_default_learning_rate, state_dict=None):
        """
        If ``state_dict`` is provided, then the trainer is recreated from this ``state_dict`` and the other parameters
        are ignored. If ``state_dict`` is ``None``, then the trainer is created according to the other parameters.

        Args:
            layer: Layer to train.
            decoder_kernel_size (int): Size of the input grid of decoder capsule layer (``kernel_size`` by
                ``kernel_size``).
            learning_rate (float): Learning rate (default: 0.01).
            state_dict: State dictionary from which the trainer can be recreated.
        """
        if state_dict:
            self._layer = cps.create_layer_from_state(state_dict['layer'])
            self._input_size = state_dict['input_size']
            self._decoder = Decoder(state_dict=state_dict['decoder'])
            self._last_num_cap_channels = len(self._layer.vec_sizes)
            self._learning_rate = state_dict['learning_rate']
            self._optimizer = torch.optim.Adam(self._optimizer_params())
            self._optimizer.load_state_dict(state_dict['optimizer'])

        else:
            assert layer is not None and decoder_kernel_size is not None

            self._layer = layer
            self._input_size = layer.input_size((decoder_kernel_size,) * 2)
            self._decoder = Decoder(kernel_size=decoder_kernel_size,
                                    num_linear_layers=2,
                                    output_size=self._input_size,
                                    output_stride=layer.stride,
                                    normalize_output=isinstance(layer, cps.PrimaryLayer))
            self._last_num_cap_channels = 0
            self._learning_rate = learning_rate
            self._fit_decoder_size()

    def state_dict(self):
        """
        Returns:
            State dictionary from which the trainer can be recreated.
        """
        return {'layer': self._layer.state_dict(),
                'input_size': self._input_size,
                'decoder': self._decoder.state_dict(),
                'learning_rate': self._learning_rate,
                'optimizer': self._optimizer.state_dict()}

    def _fit_decoder_size(self):
        if self._last_num_cap_channels < len(self._layer.vec_sizes):
            self._last_num_cap_channels = len(self._layer.vec_sizes)

            self._decoder.expand(self._layer)
            self._optimizer = torch.optim.Adam(self._optimizer_params(), lr=self._learning_rate)

    @property
    def learning_rate(self):
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, learning_rate):
        self._learning_rate = learning_rate
        self._optimizer.param_groups[0]['lr'] = learning_rate

    def _optimizer_params(self):
        for param in self._layer.parameters():
            yield param
        for param in self._decoder.parameters():
            yield param

    @property
    def layer(self):
        return self._layer

    @property
    def input_size(self):
        return self._input_size

    @property
    def decoder(self):
        self._fit_decoder_size()
        return self._decoder

    def predict(self, inputs):
        """
        Args:
            inputs (tensor): Trained layer input of shape (num_samples, height, width, depth).

        Returns:
            The result of processing ``inputs`` by trained layer and decoder. The shape of the result is the same as the
            shape of ``inputs``.
        """
        self._fit_decoder_size()
        x = self._layer(inputs)
        x = self._decoder(x)
        return x

    def train(self, train_inputs):
        """
        Args:
            train_inputs (tensor): Trained layer inputs of shape (num_samples, height, width, depth).

        Returns:
            Mean value of losses of all samples in ``train_inputs``.
        """
        self._optimizer.zero_grad()
        prediction = self.predict(train_inputs)
        loss = _loss(train_inputs, prediction).mean()
        loss.backward()
        self._optimizer.step()
        return loss.item()

    def test(self, test_inputs):
        """
        Args:
            test_inputs (tensor): Trained layer inputs of shape (num_samples, height, width, depth).

        Returns:
            Tuple (prediction, loss). Shape of prediction is the same as the shape of ``test_inputs``. Shape of loss is
            (num_samples,).
        """
        with torch.no_grad():
            prediction = self.predict(test_inputs)
            loss = _loss(test_inputs, prediction)
            return prediction, loss


class Decoder:
    """Decodes capsule layer output into its input."""

    def __init__(self, *, kernel_size=None, num_linear_layers=None, output_size=None, output_stride=None,
                 normalize_output=None, device=cps.default_device, state_dict=None):
        """
        If ``state_dict`` is provided, then the decoder is recreated from this ``state_dict`` and the other parameters
        are ignored. If ``state_dict`` is ``None``, then the decoder is created according to the other parameters.

        Args:
            kernel_size (int): Size of the input grid of decoder capsule layer (``kernel_size`` by ``kernel_size``).
            num_linear_layers (int): Number of linear layers used by decoder.
            output_size (tuple of ints): Size of decoder output (height, width, depth).
            output_stride (int): Stride used by the layer whose output is decoded.
            normalize_output (bool): If ``True``, then the values of the output will be from range [0, 1], otherwise
                the values can be any float values.
            device (torch.device): Device of this decoder.
            state_dict: State dictionary from which the decoder can be recreated.
        """
        if state_dict:
            self._kernel_size = state_dict['kernel_size']
            self._decoder_caps_layer = cps.create_layer_from_state(state_dict=state_dict['decoder_caps_layer'])
            self._linear_layers = [cps.create_linear_from_state(linear_state)
                                   for linear_state in state_dict['linear_layers']]

            self._output_size = state_dict['output_size']
            self._output_stride = state_dict['output_stride']
            self._normalize_output = state_dict['normalize_output']
            self.to(state_dict['device'])

        else:
            assert kernel_size is not None and num_linear_layers is not None and output_size is not None and \
                   output_stride is not None and normalize_output is not None

            self._kernel_size = kernel_size
            self._decoder_caps_layer = None
            self._linear_layers = [None] * num_linear_layers

            assert len(output_size) == 3
            assert output_size[0] == output_size[1]
            self._output_size = output_size

            self._output_stride = output_stride
            self._normalize_output = normalize_output
            self._device = device

    def state_dict(self):
        """
        Returns:
            State dictionary from which the decoder can be recreated.
        """
        return {'kernel_size': self._kernel_size,
                'decoder_caps_layer': self._decoder_caps_layer.state_dict(),
                'linear_layers': [linear.state_dict() for linear in self._linear_layers],
                'output_size': self._output_size,
                'output_stride': self._output_stride,
                'normalize_output': self._normalize_output,
                'device': self._device}

    @property
    def kernel_size(self):
        return self._kernel_size

    def expand(self, decoded_layer):
        """
        Expands the decoder so that it's matched to the expanded decoded layer.

        Args:
            decoded_layer: Expanded decoded layer.
        """
        self._expand_cap_layer(decoded_layer)
        self._expand_linear_layers()

    def _expand_cap_layer(self, decoded_layer):
        assert decoded_layer.input_size((self._kernel_size,) * 2) == self._output_size

        src_vec_sizes = decoded_layer.vec_sizes

        if self._decoder_caps_layer is None:
            self._decoder_caps_layer = cps.Layer(kernel_size=self._kernel_size, stride=1, src_vec_sizes=src_vec_sizes)
        else:
            self._decoder_caps_layer.expand_src_vec_sizes(src_vec_sizes)

        vec_size = math.ceil(sum(src_vec_sizes) / len(src_vec_sizes) * _decoder_vec_size_multiplier)
        vec_size_sum = min(np.array(self._output_size).prod(), sum(src_vec_sizes) * self._kernel_size ** 2)
        vec_size_sum *= _decoder_vec_sizes_sum_multiplier
        num_cap_channels = math.ceil(vec_size_sum / vec_size)

        num_new_cap_channels = num_cap_channels - len(self._decoder_caps_layer.dvec_sizes)

        if num_new_cap_channels > 0:
            new_dvec_sizes = [(vec_size, 0)] * num_new_cap_channels

            self._decoder_caps_layer.add_cap_channels(new_dvec_sizes)

    def _expand_linear_layers(self):
        caps_output_depth = self._decoder_caps_layer.vec_sizes_sum + len(self._decoder_caps_layer.vec_sizes)

        sizes = np.linspace(math.ceil(caps_output_depth * _decoder_linear_size_multiplier),
                            np.array(self._output_size).prod(),
                            len(self._linear_layers) + 1)

        sizes[1:-1] *= _decoder_linear_size_multiplier
        sizes = np.ceil(sizes).astype(int)

        if self._linear_layers[0] is None or self._linear_layers[0].in_features < sizes[0]:
            for idx, (in_features, out_features) in enumerate(zip(sizes, sizes[1:])):
                self._linear_layers[idx] = cps.get_expanded_layer(self._linear_layers[idx], in_features, out_features,
                                                                  device=self._device)

    def __call__(self, inputs):
        """
        Args:
            inputs (tensor): Decoder input (decoded layer output) of shape (num_samples, height, width, depth).

        Returns:
            Decoder output, the same shape as the decoded layer input.
        """
        caps_outputs = self._decoder_caps_layer(inputs)

        caps_grid_size = caps_outputs.shape[1:3]

        if caps_grid_size == (1, 1):
            return self._decode_linear(caps_outputs.squeeze(dim=2).squeeze(dim=1))

        big_output_size = np.array((len(inputs),) + self._output_size)
        big_output_size[1:3] += (np.array(caps_grid_size) - 1) * self._output_stride

        z_sum = torch.zeros(*big_output_size, device=self._device)
        z_count = torch.zeros(*big_output_size, device=self._device)

        for y in range(caps_outputs.shape[1]):
            ys = y * self._output_stride
            y_slice = slice(ys, ys + self._output_size[0])

            for x in range(caps_outputs.shape[2]):
                xs = x * self._output_stride
                x_slice = slice(xs, xs + self._output_size[1])

                z_sum[:, y_slice, x_slice, :] += self._decode_linear(caps_outputs[:, y, x, :])
                z_count[:, y_slice, x_slice, :] += 1

        return z_sum / z_count

    def _decode_linear(self, x):
        for layer in self._linear_layers[:-1]:
            x = layer(x)
            x = F.leaky_relu(x)

        x = self._linear_layers[-1](x)
        x = (torch.sigmoid if self._normalize_output else F.leaky_relu)(x)

        return x.view(len(x), *self._output_size)

    def parameters(self):
        """
        Returns:
            Parameters for optimizer.
        """
        for param in self._decoder_caps_layer.parameters():
            yield param

        for layer in self._linear_layers:
            for param in layer.parameters():
                yield param

    def to(self, device):
        """
        Args:
            device (torch.device): New device of this decoder.
        """
        self._device = device
        self._decoder_caps_layer.to(device)

        for linear in self._linear_layers:
            linear.to(device)


def _loss(inputs, prediction):
    return ((inputs - prediction) ** 2).view(len(inputs), -1).sum(dim=1)
