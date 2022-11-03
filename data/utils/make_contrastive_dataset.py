import pandas as pd


def contrastive_data(data_path):
    """
    binary-label==0(neg pair)인 문장을 기준으로 contrastive learning dataframe을 생성
    main sent의 pos pair는 main sent와 동일한 문장으로 사용
    """
    train = pd.read_csv(data_path)
    neg_pair = train[train["binary-label"] == 0]
    final_df = pd.DataFrame()

    for row in neg_pair.iterrows():

        new_df_row = pd.DataFrame(
            {
                "main_sentence": [row[1]["sentence_1"]],
                "pos_sentence": [row[1]["sentence_1"]],
                "neg_sentence": [row[1]["sentence_2"]],
            }
        )
        final_df = pd.concat([final_df, new_df_row], ignore_index=True)

    return final_df


if __name__ == "__main__":
    train_data_path = "data/raw_data/train.csv"
    dev_data_path = "data/raw_data/dev.csv"

    cl_train_df = contrastive_data(train_data_path)
    cl_dev_df = contrastive_data(dev_data_path)

    cl_train_df.to_csv("data/contrastive_data/contrastive_train.csv", index=False)
    cl_dev_df.to_csv("data/contrastive_data/contrastive_dev.csv", index=False)
