import torch


class BatchNormMock:
    def __init__(self, batch_norm):
        self.__class__ = type(
            batch_norm.__class__.__name__, (self.__class__, batch_norm.__class__), {}
        )
        self.__dict__ = batch_norm.__dict__
        self.num_features = batch_norm.num_features

    def forward(self, x):
        in_channels = x.shape[1]
        assert self.num_features == in_channels
        return x


mock_dict = {
    torch.nn.modules.batchnorm.BatchNorm1d: BatchNormMock,
    torch.nn.modules.batchnorm.BatchNorm2d: BatchNormMock,
    torch.nn.modules.batchnorm.BatchNorm3d: BatchNormMock,
    # torch.nn.modules.batchnorm.SyncBatchNorm,
    # torch.nn.modules.batchnorm.LazyBatchNorm1d,
    # torch.nn.modules.batchnorm.LazyBatchNorm2d,
    # torch.nn.modules.batchnorm.LazyBatchNorm3d,
    # torch.nn.modules.instancenorm.InstanceNorm1d,
    # torch.nn.modules.instancenorm.InstanceNorm2d,
    # torch.nn.modules.instancenorm.InstanceNorm3d,
    # torch.nn.modules.instancenorm.LazyInstanceNorm1d,
    # torch.nn.modules.instancenorm.LazyInstanceNorm2d,
    # torch.nn.modules.instancenorm.LazyInstanceNorm3d,
    # torch.nn.modules.normalization.LocalResponseNorm,
    # torch.nn.modules.normalization.CrossMapLRN2d,
    # torch.nn.modules.normalization.LayerNorm,
    # torch.nn.modules.normalization.GroupNorm,
}
