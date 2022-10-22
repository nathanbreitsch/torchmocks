import torch
from torchmocks.nn.pooling import mock_dict


def test_max_pool_2d():
    x = torch.zeros((4, 3, 7, 8))
    conv = torch.nn.Conv2d(3, 8, 1)  # need something to test autograd
    mock = mock_dict[torch.nn.MaxPool2d](
        torch.nn.MaxPool2d(kernel_size=3, padding=1, dilation=2)
    )
    real = torch.nn.MaxPool2d(kernel_size=3, padding=1, dilation=2)
    out_mock = mock(conv(x))
    out_real = real(conv(x))
    assert out_real.shape == out_mock.shape
    out_mock.sum().backward()


def test_avg_pool_2d():
    return
    x = torch.zeros((4, 3, 7, 8))
    conv = torch.nn.Conv2d(3, 8, 1)
    mock = mock_dict[torch.nn.AvgPool2d](torch.nn.AvgPool2d(kernel_size=3, padding=1))
    real = torch.nn.AvgPool2d(kernel_size=3, padding=1)
    out_mock = mock(conv(x))
    out_real = real(conv(x))
    assert out_real.shape == out_mock.shape
    out_mock.sum().backward()
