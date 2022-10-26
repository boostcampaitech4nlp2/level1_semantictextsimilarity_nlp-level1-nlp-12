from transformers import AutoTokenizer


def PretrainedTokenizer(checkpoint, **kwargs):
    """
    for more info: https://huggingface.co/docs/transformers/v4.23.1/en/main_classes/tokenizer
    """
    return AutoTokenizer.from_pretrained(
        checkpoint,
        **kwargs
    )