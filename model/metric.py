import torchmetrics

def pearson(logits, y):
    return torchmetrics.functional.pearson_corrcoef(logits, y)