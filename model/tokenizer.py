from transformers import AutoTokenizer


def PretrainedTokenizer(**kwargs):
    #for more info: https://huggingface.co/docs/transformers/v4.23.1/en/main_classes/tokenizer
    return AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=kwargs.pop("checkpoint")
        )
