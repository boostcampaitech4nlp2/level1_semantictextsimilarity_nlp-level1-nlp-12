import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


class Trainer(pl.Trainer):
    def __init__(self, config, wandb_logger):

        early_stop_callback = EarlyStopping(
            monitor="val_loss", min_delta=0.00, patience=3, verbose=False, mode="min"
        )
        
        # finding best model
        checkpoint_callback = ModelCheckpoint(
                monitor= "val_pearson",
                mode='max',
                dirpath="./checkpoint/"
                #save_top_k=1
        )
        '''
        checkpoint_callback = ModelCheckpoint(
                monitor= "val_loss",
                mode='min',
                save_top_k=2
        )
        '''

        super(Trainer, self).__init__(
            accelerator="gpu",
            devices=config.train.gpus,
            max_epochs=config.train.max_epoch,
            logger=wandb_logger,
            log_every_n_steps=1,
            callbacks=[early_stop_callback, checkpoint_callback],
        )
