import pytorch_lightning
import torch
import torchvision
from torchmocks import mock
from torchvision.datasets import FakeData
from torchvision.models import feature_extraction, resnet152


class ExamplePytorchLightningModule(pytorch_lightning.LightningModule):
    def __init__(self):
        super().__init__()
        self.feature_extractor = feature_extraction.create_feature_extractor(
            resnet152(), ["avgpool"]
        )
        # self.feature_extractor = lambda x: {'avgpool': x}
        self.flatten = torch.nn.Flatten()
        self.linear = torch.nn.Linear(2048, 10)

    def forward(self, x):
        features = self.feature_extractor(x)["avgpool"]
        logits = self.linear(self.flatten(features))
        return torch.nn.functional.softmax(x, dim=-1)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        features = self.feature_extractor(x)["avgpool"]
        logits = self.linear(self.flatten(features))
        loss = torch.nn.functional.cross_entropy(logits, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        features = self.feature_extractor(x)["avgpool"]
        logits = self.linear(self.flatten(features))
        loss = torch.nn.functional.cross_entropy(logits, y)
        self.log("val_loss", loss)
        return loss


class MockDataset:
    def __init__(self, x_shape=(3, 40, 40), length=10):
        self.x_shape = x_shape
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return torch.zeros(self.x_shape), torch.tensor(index, dtype=int)


def test_training():
    dataset = MockDataset()
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=3)
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=3)
    model = ExamplePytorchLightningModule()
    mock(model, debug=True)
    trainer = pytorch_lightning.Trainer(fast_dev_run=2)
    trainer.fit(model, train_loader, val_loader)
