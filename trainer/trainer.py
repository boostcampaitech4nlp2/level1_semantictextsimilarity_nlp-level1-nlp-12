import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint


class Trainer(pl.Trainer):
    def __init__(self, config, wandb_logger):
        if not os.path.exists(config.model.saved_checkpoint):
            os.makedirs(config.model.saved_checkpoint)

        checkpoint_callback = ModelCheckpoint(
            dirpath=config.model.saved_checkpoint,
            filename=config.wandb.info + "--{epoch:02d}--{val_pearson:.3f}",
            verbose=True,
            save_last=False,
            save_top_k=1,
            monitor="val_pearson",
            mode="max",
        )

        early_stop_callback = EarlyStopping(
            monitor="val_loss", min_delta=0.00, patience=3, verbose=False, mode="min"
        )

        super(Trainer, self).__init__(
            accelerator="gpu",
            devices=config.train.gpus,
            max_epochs=config.train.max_epoch,
            logger=wandb_logger,
            log_every_n_steps=1,
            callbacks=[early_stop_callback, checkpoint_callback],
        )


class ContrastiveTrainer(pl.Trainer):
    def __init__(self, config, wandb_logger):
        if not os.path.exists(config.model.saved_contrastive_checkpoint):
            os.makedirs(config.model.saved_contrastive_checkpoint)

        checkpoint_callback = ModelCheckpoint(
            dirpath=config.model.saved_contrastive_checkpoint,
            filename=config.wandb.info + "--{epoch:02d}--{val_triplet_loss:.3f}",
            verbose=True,
            save_last=False,
            save_top_k=1,
            monitor="val_triplet_loss",
            mode="min",
        )

        early_stop_callback = EarlyStopping(
            monitor="val_triplet_loss",
            min_delta=0.00,
            patience=3,
            verbose=False,
            mode="min",
        )

        super(ContrastiveTrainer, self).__init__(
            accelerator="gpu",
            devices=config.train.gpus,
            max_epochs=config.train.max_epoch,
            logger=wandb_logger,
            log_every_n_steps=1,
            callbacks=[early_stop_callback, checkpoint_callback],
        )
