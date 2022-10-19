import torch
from torchmocks.nn.conv import mock_dict


def test_conv_1d():
    x = torch.zeros((4, 3, 7))
    conv1d_mock = mock_dict[torch.nn.modules.conv.Conv1d]
    conv1d_real = torch.nn.Conv1d(3, 8, 3)
    out_real = conv1d_real(x)
    out_mock = conv1d_mock(torch.nn.Conv1d(3, 8, 3))(x)
    assert out_real.shape == out_mock.shape


def test_conv_2d():
    x = torch.zeros((4, 3, 7, 8))
    conv2d_mock = mock_dict[torch.nn.modules.conv.Conv2d]
    conv2d_real = torch.nn.Conv2d(3, 8, 3)
    out_real = conv2d_real(x)
    out_mock = conv2d_mock(torch.nn.Conv2d(3, 8, 3))(x)
    assert out_real.shape == out_mock.shape


def test_conv_3d():
    x = torch.zeros((4, 3, 7, 8, 9))
    conv3d_mock = mock_dict[torch.nn.modules.conv.Conv3d]
    conv3d_real = torch.nn.Conv3d(3, 8, 3)
    out_real = conv3d_real(x)
    out_mock = conv3d_mock(torch.nn.Conv3d(3, 8, 3))(x)
    assert out_real.shape == out_mock.shape


def test_transpose_conv_1d():
    x = torch.zeros((4, 3, 7))
    convtrans1d_mock = mock_dict[torch.nn.modules.conv.ConvTranspose1d]
    convtrans1d_real = torch.nn.ConvTranspose1d(3, 8, 3)
    out_real = convtrans1d_real(x)
    out_mock = convtrans1d_mock(torch.nn.ConvTranspose1d(3, 8, 3))(x)
    assert out_real.shape == out_mock.shape


def test_transpose_conv_2d():
    x = torch.zeros((4, 3, 7, 8))
    convtrans2d_mock = mock_dict[torch.nn.modules.conv.ConvTranspose2d]
    convtrans2d_real = torch.nn.ConvTranspose2d(3, 8, 3)
    out_real = convtrans2d_real(x)
    out_mock = convtrans2d_mock(torch.nn.ConvTranspose2d(3, 8, 3))(x)
    assert out_real.shape == out_mock.shape


def test_transpose_conv_3d():
    x = torch.zeros((4, 3, 7, 8, 9))
    convtrans3d_mock = mock_dict[torch.nn.modules.conv.ConvTranspose3d]
    convtrans3d_real = torch.nn.ConvTranspose3d(3, 8, 3)
    out_real = convtrans3d_real(x)
    out_mock = convtrans3d_mock(torch.nn.ConvTranspose3d(3, 8, 3))(x)
    assert out_real.shape == out_mock.shape
