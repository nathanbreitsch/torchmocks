import time

import torch
from torchmocks import mock
from torchvision.models import resnet152


def test_mock_resnet():
    net = resnet152()
    mock(net, debug=True)
    image_batch = torch.zeros(4, 3, 255, 255)
    start = time.time()
    output = net(image_batch)
    end = time.time()
    elapsed = end - start
    print(elapsed)
    assert elapsed < 0.25
    assert output.shape == (4, 1000)
