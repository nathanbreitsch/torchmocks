import time

import torch
import vit_pytorch
from vit_pytorch import ViT
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


def test_mock_transformer():
    net = torch.nn.Transformer(nhead=16, num_encoder_layers=12)
    mock(net, debug=True)
    src = torch.rand((10, 32, 512))
    tgt = torch.rand((20, 32, 512))
    out = net(src, tgt)


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
