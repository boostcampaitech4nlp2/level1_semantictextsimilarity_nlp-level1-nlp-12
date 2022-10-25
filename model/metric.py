import torchmetrics


def pearson_corrcoef(logits, y):
    return torchmetrics.functional.pearson_corrcoef(logits, y)
