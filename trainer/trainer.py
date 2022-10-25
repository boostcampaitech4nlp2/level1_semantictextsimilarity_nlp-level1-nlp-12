import pytorch_lightning as pl

class Trainer(pl.Trainer):
    def __init__(self, gpus, max_epochs, log_every_n_steps):
        ''' 
        !Warning :  gpus=x has been deprecated in v1.7 and will be removed in v2.0. Please use accelerator='gpu' and devices=x instead.
        
        '''
        super().__init__(gpus=gpus, max_epochs=max_epochs, log_every_n_steps=log_every_n_step)