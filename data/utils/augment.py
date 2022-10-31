from koeda import AEDA
import pandas as pd
import numpy as np
import os
import copy
import numpy as np


def get_preprocessed_label(df):
    """label값을 반올림한 preprocessed_label 칼럼을 추가

    Args:
        df (DataFrame)_
    Returns:
        preprocessed_label 칼럼이 추가된 데이터프레임
    """
    for i in range(len(df)):
        df.loc[i, "preprocessed_label"] = round(df.loc[i, "label"])

    return df


def concat_AEDA_sent(df_label, aug_nums):
    """라벨 x만 있는 데이터프레임에 aug_nums 만큼 AEAD된 데이터를 추가

    Args:
        df_label (DataFrame): 라벨 x만 있는 데이터프레임
        aug_nums (int): 증강하고자 하는 수
    Returns:
        AEDA가 추가된 라벨 x의 데이터프레임
    """
    np.random.seed(12)
    aug_idx = np.random.randint(0, 3, size=aug_nums)
    dataset_idx = 0
    aug_df_label = copy.deepcopy(df_label)

    aeda = AEDA(
        morpheme_analyzer="Okt",
        punc_ratio=0.3,
        punctuations=[".", ",", "!", "?", ";", ":"],
    )

    for i in range(aug_nums):
        if dataset_idx >= len(df_label):
            dataset_idx = 0

        origin_id = df_label.iloc[dataset_idx]["id"]
        origin_source = df_label.iloc[dataset_idx]["source"]
        origin_sentence_1 = df_label.iloc[dataset_idx]["sentence_1"]
        origin_sentence_2 = df_label.iloc[dataset_idx]["sentence_2"]
        origin_label = df_label.iloc[dataset_idx]["label"]
        origin_binarylabel = df_label.iloc[dataset_idx]["binary-label"]
        origin_preprocessed_label = df_label.iloc[dataset_idx]["preprocessed_label"]

        if aug_idx[i] == 0:  # sent_1만 aug
            sent = aug_df_label.iloc[dataset_idx]["sentence_1"]
            aug_sent = aeda(sent)
            new_df = pd.DataFrame(
                {
                    "id": [origin_id],
                    "source": [origin_source],
                    "sentence_1": [aug_sent],
                    "sentence_2": [origin_sentence_2],
                    "label": [origin_label],
                    "binary-label": [origin_binarylabel],
                    "preprocessed_label": [origin_preprocessed_label],
                }
            )
            aug_df_label = pd.concat([aug_df_label, new_df], ignore_index=True)

        elif aug_idx[i] == 1:  # sent_2만 aug
            sent = aug_df_label.iloc[dataset_idx]["sentence_2"]
            aug_sent = aeda(sent)
            new_df = pd.DataFrame(
                {
                    "id": [origin_id],
                    "source": [origin_source],
                    "sentence_1": [origin_sentence_1],
                    "sentence_2": [aug_sent],
                    "label": [origin_label],
                    "binary-label": [origin_binarylabel],
                    "preprocessed_label": [origin_preprocessed_label],
                }
            )
            aug_df_label = pd.concat([aug_df_label, new_df], ignore_index=True)

        else:  # sent_1과 2를 모두 aug
            sent_1 = aug_df_label.iloc[dataset_idx]["sentence_1"]
            sent_2 = aug_df_label.iloc[dataset_idx]["sentence_2"]
            aug_sent_1 = aeda(sent_1)
            aug_sent_2 = aeda(sent_2)
            new_df = pd.DataFrame(
                {
                    "id": [origin_id],
                    "source": [origin_source],
                    "sentence_1": [aug_sent_1],
                    "sentence_2": [aug_sent_2],
                    "label": [origin_label],
                    "binary-label": [origin_binarylabel],
                    "preprocessed_label": [origin_preprocessed_label],
                }
            )
            aug_df_label = pd.concat([aug_df_label, new_df], ignore_index=True)

        dataset_idx += 1

    return aug_df_label


def AEDA_data():

    train = pd.read_csv("data/raw_data/train.csv")
    train = get_preprocessed_label(train)

    train_0 = train[train["preprocessed_label"] == 0].reset_index(drop=True)
    train_1 = train[train["preprocessed_label"] == 1].reset_index(drop=True)
    train_2 = train[train["preprocessed_label"] == 2].reset_index(drop=True)
    train_3 = train[train["preprocessed_label"] == 3].reset_index(drop=True)
    train_4 = train[train["preprocessed_label"] == 4].reset_index(drop=True)
    train_5 = train[train["preprocessed_label"] == 5].reset_index(drop=True)

    train_1_aug = concat_AEDA_sent(train_1, 1323)
    train_2_aug = concat_AEDA_sent(train_2, 1906)
    train_3_aug = concat_AEDA_sent(train_3, 1647)
    train_4_aug = concat_AEDA_sent(train_4, 936)
    train_5_aug = concat_AEDA_sent(train_5, 2750)

    train_0 = pd.concat(
        [train_0, train_1_aug, train_2_aug, train_3_aug, train_4_aug, train_5_aug],
        ignore_index=True,
    )

    auged_df = train_0.sample(frac=1).reset_index(drop=True)

    return auged_df


if __name__ == "__main__":
    final_df = AEDA_data()
    final_df.to_csv("../aug_data/train_auged.csv", index=False)
