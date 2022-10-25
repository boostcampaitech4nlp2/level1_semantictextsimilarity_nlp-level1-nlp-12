import os
import random
import pandas as pd
import numpy as np

RANDOM_SEED = 12

def random_deletion(words, p):
    """
    해당 문장에서 p의 확률로 무작위 단어 삭제

    Args:
        words (list): 단어 리스트
        p (float): 한 단어당 삭제할 확률

    Returns:
        new_words (list): words 리스트의 원소를 삭제한 결과 나온 리스트
    """
    if len(words) == 1:
        return words

    np.random.seed(RANDOM_SEED)
    new_words = []
    for word in words:
        r = np.random.uniform(0, 1)
        if r > p:
            new_words.append(word)

    if len(new_words) == 0:
        rand_int = np.random.randint(0, len(words))
        return [words[rand_int]]

    return new_words

def random_swap(words, n):
    """
    문장의 두 단어를 무작위로 n번 교환

    Args:
        words (list): 단어 리스트
        n (int): 단어를 교환할 횟수

    Returns:
        new_words (list): words 리스트의 원소를 교환한 결과 나온 리스트
    """
    new_words = words.copy()
    for _ in range(n):
        new_words = swap_word(new_words)

    return new_words

def swap_word(new_words):
    """
    두 단어를 변경할 인덱스를 무작위로 선택 후 단어 변경

    Args:
        new_words (list): 단어 변경 전 리스트

    Returns:
        new_words (list): 단어 변경 후 리스트
    """
    np.random.seed(RANDOM_SEED)
    random_idx_1 = np.random.randint(0, len(new_words))
    random_idx_2 = random_idx_1
    counter = 0

    while random_idx_2 == random_idx_1:
        random_idx_2 = np.random.randint(0, len(new_words))
        counter += 1
        if counter > 3:
            return new_words

    new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1]
    return new_words


def random_deletion_df(df, p, key, cnt) :
    """
    해당 문장에서 p의 확률로 무작위 단어 삭제 (with pd.DataFrame)
    1개의 Series 데이터당 2개의 data augmentation

    Args:
        df (pd.DataFrame): RD를 진행할 데이터프레임
        p (float): 한 단어당 삭제할 확률
        key (str): RD를 진행할 label-binning 그룹
        cnt (int): Text augmentation으로 데이터를 생성할 개수

    Returns:
        aug_df (pd.DataFrame): Text augmentation 결과 나온 데이터프레임
    """
    src_df = df[(df['label-binning'] == key)]
    
    index_list = []
    np.random.seed(RANDOM_SEED)

    # augmentation은 1문장 당 2개씩 나오기 때문에 cnt // 2
    while len(index_list) < cnt // 2 :
        random_index = np.random.randint(len(src_df))

        if random_index not in index_list :
            index_list.append(random_index)
            
    tmp_df = src_df.iloc[index_list]
    
    aug_series_list = []

    for i in range(len(tmp_df)) :
        series = tmp_df.iloc[i].copy()
        sentence1 = series['sentence_1']
        sentence2 = series['sentence_2']

        words1 = [word for word in sentence1.split(' ') if word != ""]
        words2 = [word for word in sentence2.split(' ') if word != ""]

        sentence1 = ' '.join(random_deletion(words1, p))
        sentence2 = ' '.join(random_deletion(words2, p))

        series1 = series.copy()
        series2 = series.copy()

        series1['sentence_1'] = sentence1
        series2['sentence_2'] = sentence2

        aug_series_list.extend([series1, series2])
        
    aug_df = pd.DataFrame(aug_series_list)
    
    return aug_df


def random_swap_df(df, n, key, cnt) :
    """
    문장의 두 단어를 무작위로 n번 교환 (with pd.DataFrame)
    1개의 Series 데이터당 2개의 data augmentation

    Args:
        df (pd.DataFrame): RS를 진행할 데이터프레임
        n (int): 단어를 교환할 횟수
        key (str): RS를 진행할 label-binning 그룹
        cnt (int): Text augmentation으로 데이터를 생성할 개수

    Returns:
        aug_df (pd.DataFrame): Text augmentation 결과 나온 데이터프레임
    """
    src_df = df[(df['label-binning'] == key)]
    
    index_list = []
    np.random.seed(RANDOM_SEED)

    # augmentation은 1문장 당 2개씩 나오기 때문에 cnt // 2
    while len(index_list) < cnt//2 :
        random_index = np.random.randint(len(src_df))

        if random_index not in index_list :
            index_list.append(random_index)
            
    tmp_df = src_df.iloc[index_list]
    
    aug_series_list = []

    for i in range(len(tmp_df)) :
        series = tmp_df.iloc[i].copy()
        sentence1 = series['sentence_1']
        sentence2 = series['sentence_2']

        words1 = [word for word in sentence1.split(' ') if word != ""]
        words2 = [word for word in sentence2.split(' ') if word != ""]

        sentence1 = ' '.join(random_swap(words1, n))
        sentence2 = ' '.join(random_swap(words2, n))

        series1 = series.copy()
        series2 = series.copy()

        series1['sentence_1'] = sentence1
        series2['sentence_2'] = sentence2

        aug_series_list.extend([series1, series2])
        
    aug_df = pd.DataFrame(aug_series_list)
    
    return aug_df


def text_augmentation(train_path) :
    """
    Text augmentation를 진행하는 함수 (RD, RS)

    Args:
        train_path (str): train data 경로 

    *Create File:
        aug_train_df (.csv): Text augmentation 결과 나온 데이터프레임
    """
    train_df = pd.read_csv(train_path, engine='python')
    
    # data-binning (데이터 구간화)
    bins = [-0.1, 0.0, 0.9, 1.9, 2.9, 3.9, 4.9, 5]
    df_label_bins = pd.cut(train_df['label'], bins, labels=['0.0','0.1~0.9', '1.0~1.9', '2.0~2.9', 
                                                            '3.0~3.9', '4.0~4.9', '5.0'])

    # train_df 복사
    new_train_df = train_df.copy()
    new_train_df['label-binning'] = df_label_bins
    
    # text augmentation
    aug_del_df = random_deletion_df(df=new_train_df, p=0.25, key='2.0~2.9', cnt=100)
    aug_swap_df = random_swap_df(df=new_train_df, n=1, key='2.0~2.9', cnt=100) 
    
    # 원본 train_df와 aug_df 합치기
    aug_list = [new_train_df, aug_del_df, aug_swap_df]
    aug_train_df = pd.concat(aug_list)
    
    aug_train_df.to_csv('./data/aug_train.csv')


####################################################
if __name__ == "__main__":
    FILE_PATH = './data'
    train_path = os.path.join(FILE_PATH, 'train.csv')

    text_augmentation(train_path)

### 업데이트 예정 2022-10-25 ###
# 1. Back-Translation augmentation 추가 
# 2. random_deleton_df, random_swap_df 함수
#    key를 'str'이 아닌 'list'로 받는 형식으로 변경
# 

### 고민중 ###
# 1. synthetic noise 
# 2. synonym replacement, random insertion