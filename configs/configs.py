import argparse

parser = argparse.ArgumentParser(description="pstage")

### WandB ###
parser.add_argument(
    "--wandb_project", default="Text_aug", type=str, help="wandb 레포지토리 이름"
)
parser.add_argument("--name", default="kjh", type=str, help="작성자 이름")
parser.add_argument("--info", default="데이터증강", type=str, help="추가한 작업 정보")

### Model ###
parser.add_argument(
    "--model_name",
    type=str,
    default="beomi/KcELECTRA-base",
    help="klue/roberta-small | monologg/koelectra-base-v3-discriminator | beomi/KcELECTRA-base",
)

### HyperParmeters ###
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--max_epoch", default=10, type=int)
parser.add_argument("--shuffle", default=True)

### optimizer ###
parser.add_argument("--optimizer", default="AdamW", type=str, help="SGD | AdamW")
parser.add_argument("--learning_rate", default=1e-5, type=float)

### scheduler ###
parser.add_argument(
    "--scheduler",
    default="StepLR",
    type=str,
    help="StepLR | ...",
)

### Data Path ###
parser.add_argument("--train_path", default="./data/train.csv")
parser.add_argument("--aug_train_path", default="./data/aug_train.csv")
parser.add_argument("--dev_path", default="./data/dev.csv")
parser.add_argument("--test_path", default="./data/dev.csv")
parser.add_argument("--predict_path", default="./data/test.csv")

config = parser.parse_args()
