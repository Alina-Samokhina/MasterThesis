import os
import time

import pytorch_lightning as pl
import torch
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint

from ..datasets import load_dataset
from .base import Learner


def evaluate(
    model,
    timestamps=False,
    coeffs=False,
    irregular=False,
    dataset="p300",
    batch_size=128,
    data_dir="../data/demons/nery_demons_dataset",
    lr=0.05,
    logs_dir="./logs_pl",
    chkpt_dir="./logs/models/demons/cde/",
    epochs=5,
    grad_clip=0,
    log_every_n_steps=1,
    val_check_interval=0.01,
    print_best_quality=True,
):
    (
        trainloader,
        testloader,
        in_features,
        _num_classes,
        _return_sequences,
        class_balance,
    ) = load_dataset(
        dataset,
        timestamps=timestamps,
        coeffs=coeffs,
        irregular=irregular,
        batch_size=batch_size,
        data_dir=data_dir,
    )
    print(in_features)

    learn = Learner(
        model, lr=lr, timestamps=timestamps, class_weights=1 / class_balance
    )
    tb_logger = pl_loggers.TensorBoardLogger(logs_dir)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_f1", mode="max", dirpath=chkpt_dir, save_top_k=3,
    )

    trainer = pl.Trainer(
        max_epochs=epochs,
        progress_bar_refresh_rate=1,
        gradient_clip_val=grad_clip,
        callbacks=[checkpoint_callback],
        logger=tb_logger,
        log_every_n_steps=log_every_n_steps,
        val_check_interval=val_check_interval,
    )

    trainer.fit(learn, trainloader, val_dataloaders=testloader)

    if print_best_quality:
        best_path = checkpoint_callback.best_model_path
        checkpoint = torch.load(best_path)
        states = {}
        for k_new, k_old in zip(
            model.state_dict().keys(), checkpoint["state_dict"].keys()
        ):
            states[k_new] = checkpoint["state_dict"].get(k_old)
        model.load_state_dict(state_dict=states)

        learn = Learner(
            model, lr=lr, timestamps=timestamps, class_weights=1 / class_balance
        )

        results = trainer.test(learn, testloader)
        base_path = "results/{}".format("person")
        os.makedirs(base_path, exist_ok=True)
        with open(f"{chkpt_dir}/results{time.time()}.csv", "a") as f:
            f.write(f"acc: {results[0]['val_acc']:.4f}, f1: {results[0]['val_f1']}")
