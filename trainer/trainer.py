import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


class Trainer(pl.Trainer):
    def __init__(self, accelerator, devices, max_epochs, log_every_n_steps, logger):
        early_stop_callback = EarlyStopping(
            monitor="val_pearson",
            min_delta=0.00,
            patience=3,
            verbose=True,
            mode="max",
        )
        super(Trainer, self).__init__(
            accelerator=accelerator,
            devices=devices,
            max_epochs=max_epochs,
            log_every_n_steps=log_every_n_steps,
            logger=logger,
            callbacks=early_stop_callback,
        )
