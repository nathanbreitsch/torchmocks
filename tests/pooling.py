import torch
from torchmocks.nn.pooling import mock_dict


def _test_pool_layer(real, mock, dim):
    conv = torch.nn.Conv2d(3, 8, 1)
    batch_size = 4
    channels = 3
    input_shape = tuple([batch_size, channels] + [12] * dim)
    x = torch.zeros(input_shape)
    # multiply by variable scalar so there is something to autograd
    scalar = torch.tensor(1.0, requires_grad=True)
    out_mock = mock(scalar * x)
    out_real = real(scalar * x)
    assert out_mock.shape == out_real.shape
    out_mock.sum().backward()


def test_max_pool_1d():
    real = torch.nn.MaxPool1d(kernel_size=3, padding=1, dilation=2)
    mock = mock_dict[torch.nn.MaxPool1d](
        torch.nn.MaxPool1d(kernel_size=3, padding=1, dilation=2)
    )
    _test_pool_layer(real, mock, 1)


def test_max_pool_2d():
    real = torch.nn.MaxPool2d(kernel_size=3, padding=1, dilation=2)
    mock = mock_dict[torch.nn.MaxPool2d](
        torch.nn.MaxPool2d(kernel_size=3, padding=1, dilation=2)
    )
    _test_pool_layer(real, mock, 2)


def test_max_pool_3d():
    real = torch.nn.MaxPool3d(kernel_size=3, padding=1, dilation=2)
    mock = mock_dict[torch.nn.MaxPool3d](
        torch.nn.MaxPool3d(kernel_size=3, padding=1, dilation=2)
    )
    _test_pool_layer(real, mock, 3)


def test_avg_pool_1d():
    real = torch.nn.AvgPool2d(kernel_size=3, padding=1)
    mock = mock_dict[torch.nn.AvgPool2d](torch.nn.AvgPool1d(kernel_size=3, padding=1))
    _test_pool_layer(real, mock, 2)


def test_avg_pool_2d():
    real = torch.nn.AvgPool2d(kernel_size=3, padding=1)
    mock = mock_dict[torch.nn.AvgPool2d](torch.nn.AvgPool1d(kernel_size=3, padding=1))
    _test_pool_layer(real, mock, 2)


def test_avg_pool_3d():
    real = torch.nn.AvgPool2d(kernel_size=3, padding=1)
    mock = mock_dict[torch.nn.AvgPool2d](torch.nn.AvgPool1d(kernel_size=3, padding=1))
    _test_pool_layer(real, mock, 2)
