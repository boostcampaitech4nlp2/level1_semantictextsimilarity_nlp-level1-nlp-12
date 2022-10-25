import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


class Trainer:
    def __init__(self, config, wandb_logger):

        early_stop_callback = EarlyStopping(
            monitor="val_loss", min_delta=0.00, patience=3, verbose=False, mode="min"
        )

        self.trainer = pl.Trainer(
            accelerator="gpu",
            devices=1,
            max_epochs=config.max_epoch,
            logger=wandb_logger,
            log_every_n_steps=1,
            callbacks=[early_stop_callback],
        )
