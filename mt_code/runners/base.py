# Code based on the implementation of the ODE-LSTM Authors Mathias Lechner ad Ramin Hasani
import pytorch_lightning as pl
import torch
import torchmetrics
from torch import nn


class Learner(pl.LightningModule):
    def __init__(
        self,
        model,
        lr=0.005,
        timestamps=True,
        num_classes=7,
        class_weights=None,
        log_to="training.csv",
        eps=1e-8,
    ):
        super().__init__()
        self.model = model
        self.lr = lr
        self.timestamps = timestamps
        self.num_classes = num_classes
        self.class_weights = class_weights
        self.accuracy = torchmetrics.Accuracy()
        self.f1 = torchmetrics.F1(num_classes, average="weighted")

    def training_step(self, batch, batch_idx):
        self.model.train()
        if self.timestamps:
            x, t, y = batch
            y_hat = self.model.forward(x, t)

        else:
            x, y = batch
            y_hat = self.model.forward(x)

        y_hat = y_hat.view(-1, y_hat.size(-1))
        y = y.view(-1)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        preds = torch.argmax(y_hat.detach(), dim=-1)
        acc = self.accuracy(preds, y)
        f1_score = self.f1(preds, y)

        self.log("train_acc", acc, prog_bar=True)
        self.log("train_f1", f1_score, prog_bar=True)
        self.log("train_loss", loss, prog_bar=True)
        return {"loss": loss, "acc": acc, "f1": f1_score}

    def validation_step(self, batch, batch_idx):
        self.model.eval()
        if self.timestamps:
            x, t, y = batch
            y_hat = self.model.forward(x, t)
        else:
            x, y = batch
            y_hat = self.model.forward(x)

        y_hat = y_hat.view(-1, y_hat.size(-1))
        y = y.view(-1)

        loss = nn.CrossEntropyLoss(weight=self.class_weights)(y_hat, y)

        preds = torch.argmax(y_hat, dim=1)
        acc = self.accuracy(preds, y)
        f1_score = self.f1(preds, y)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_f1", f1_score, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return {"loss": loss, "acc": acc, "f1": f1_score}

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)
