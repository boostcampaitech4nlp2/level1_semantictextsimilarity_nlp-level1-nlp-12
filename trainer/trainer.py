import pytorch_lightning as pl


class PL_Trainer(pl.Trainer):
    def __init__(self, gpus, accelerator, devices, max_epochs, logger, callbacks, log_every_n_steps, enable_checkpointing, default_root_dir):
        ''' 
        !Warning :  gpus=x has been deprecated in v1.7 and will be removed in v2.0. Please use accelerator='gpu' and devices=x instead.
        
        '''
        super().__init__(
            gpus=gpus,
            accelerator=accelerator,
            devices=devices,
            max_epochs=max_epochs,
            logger=logger,
            callbacks=callbacks,
            log_every_n_steps=log_every_n_steps,
            enable_checkpointing=enable_checkpointing,
            default_root_dir=default_root_dir
        )
