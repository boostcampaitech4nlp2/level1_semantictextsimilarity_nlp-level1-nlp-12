import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


class Trainer(pl.Trainer):
    def __init__(
        self, accelerator, devices, max_epochs, log_every_n_steps, logger, callbacks
    ):

        super(Trainer, self).__init__(
            accelerator=accelerator,
            devices=devices,
            max_epochs=max_epochs,
            log_every_n_steps=log_every_n_steps,
            logger=logger,
            callbacks=callbacks,
        )
