import pytorch_lightning as pl


class PL_Trainer(pl.Trainer):
    def __init__(self, **kwargs):
        ''' 
        !Warning :  gpus=x has been deprecated in v1.7 and will be removed in v2.0. \
            Please use accelerator='gpu' and devices=x instead.
        
        '''
        super().__init__(**kwargs)
