import time

import torch
import vit_pytorch
from vit_pytorch import ViT
from vit_pytorch.max_vit import MaxViT
from torchmocks import mock
from torchvision.models import resnet152


def test_mock_resnet():
    net = resnet152()
    mock(net, debug=True)
    image_batch = torch.zeros(4, 3, 256, 256)
    start = time.time()
    output = net(image_batch)
    end = time.time()
    elapsed = end - start
    assert elapsed < 0.25
    assert output.shape == (4, 1000)


v = MaxViT(
    num_classes=1000,
    dim_conv_stem=64,  # dimension of the convolutional stem, would default to dimension of first layer if not specified
    dim=96,  # dimension of first layer, doubles every layer
    dim_head=32,  # dimension of attention heads, kept at 32 in paper
    depth=(
        2,
        2,
        5,
        2,
    ),  # number of MaxViT blocks per stage, which consists of MBConv, block-like attention, grid-like attention
    window_size=7,  # window size for block and grids
    mbconv_expansion_rate=4,  # expansion rate of MBConv
    mbconv_shrinkage_rate=0.25,  # shrinkage rate of squeeze-excitation in MBConv
    dropout=0.1,  # dropout
)


def test_mock_ViT():
    net = ViT(
        image_size=256,
        patch_size=32,
        num_classes=1000,
        dim=1024,
        depth=6,
        heads=16,
        mlp_dim=2048,
        dropout=0.1,
        emb_dropout=0.1,
    )

    mock(net, debug=True)
    image_batch = torch.zeros(4, 3, 256, 256)
    start = time.time()
    output = net(image_batch)
    end = time.time()
    elapsed = end - start
    print(elapsed)
    assert elapsed < 0.25
    assert output.shape == (4, 1000)


def test_mock_MaxViT():

    net = MaxViT(
        num_classes=1000,
        dim_conv_stem=64,
        dim=96,
        dim_head=32,
        depth=(
            2,
            2,
            5,
            2,
        ),
        window_size=7,
        mbconv_expansion_rate=4,
        mbconv_shrinkage_rate=0.25,
        dropout=0.1,
    )

    from torchmocks.torchmocks import ActivationMock

    mock(net, debug=True, extra_mocks={vit_pytorch.max_vit.Dropsample: ActivationMock})
    image_batch = torch.zeros(4, 3, 224, 224)
    start = time.time()
    output = net(image_batch)
    end = time.time()
    elapsed = end - start
    print(elapsed)
    assert elapsed < 0.25
    assert output.shape == (4, 1000)
