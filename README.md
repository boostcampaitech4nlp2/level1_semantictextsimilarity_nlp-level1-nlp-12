# Semantic Text Similarity(문맥적 유사도 측정 : STS)

## Project Description

STS(Semantic Text Similarity)는 복수의 문장에 대한 유사도를 선형적 수치로 제시하는 NLP Task입니다.

본 프로젝트에서는 두 개의 문장을 입력하고, 이러한 문장쌍이 얼마나 의미적으로 서로 유사한지(0~5점)를 측정하는 AI모델을 구축합니다.


<br/>

## 데이터셋 
| Dataset            | train                    | dev | test |
| ------------------ | ----------------------- | --------------: | -----------: |
| **문장 수**        | 9324      |           550 |       1100 |
| **비율**        | 85      |           5 |       10 |

<br/>

### Columns
* **id** (문자열) : 문장 고유 ID : `데이터이름-버전-train/dev/test-번호`

* **source** (문자열) : 문장의 출처

    * **petition** (국민청원 게시판 제목 데이터)

    * **NSMC** (네이버 영화 감성 분석 코퍼스, Naver Sentiment Movie Corpus)

    * **slack** (업스테이지(Upstage) 슬랙 데이터)

* **sentence1** (문자열) : 문장 쌍의 첫번째 문장

* **sentence2** (문자열) : 문장 쌍의 두번째 문장

* **label** : 문장 쌍에 대한 유사도 0~5 점 사이의 실수 ; 소수 첫째 자리까지 존재
    * 5점 : 두 문장의 핵심 내용이 동일하며, 부가적인 내용들도 동일함

    * 4점 : 두 문장의 핵심 내용이 동등하며, 부가적인 내용에서는 미미한 차이가 있음

    * 3점 : 두 문장의 핵심 내용은 대략적으로 동등하지만, 부가적인 내용에 무시하기 어려운 차이가 있음

    * 2점 : 두 문장의 핵심 내용은 동등하지 않지만, 몇 가지 부가적인 내용을 공유함

    * 1점 : 두 문장의 핵심 내용은 동등하지 않지만, 비슷한 주제를 다루고 있음

    * 0점 : 두 문장의 핵심 내용이 동등하지 않고, 부가적인 내용에서도 공통점이 없음

* **binary-label** : 문장 쌍에 대한 유사도가 2점 이하 → 0, 3점 이상 → 1로 변환한 binary label


<br/>


## Set up

### 1. Requirements

```bash
$ pip install -r requirements.txt
```

### 2. prepare dataset

```bash
data/raw_data 폴더에 train.csv, dev.csv, test.csv 추가
```

<br/>

# How to Run

## How to train

```bash
$ sh train.sh
```

<br/>

## How to sweep hyperparameter tuning
```bash
$ sh sweep.sh

# Launch agents
## bayes나 random 탐색은 프로세스를 직접 종료하기 전까지 계속 탐색하므로 LIMIT_NUM으로 학습 횟수를 제한할 수 있다.
$ wandb agent --count [LIMIT_NUM] [SWEEPID] 
```

## How to sweep Contrastive Learning
```bash
$ sh cl_sweep.sh

# Launch agents
## bayes나 random 탐색은 프로세스를 직접 종료하기 전까지 계속 탐색하므로 LIMIT_NUM으로 학습 횟수를 제한할 수 있다.
$ wandb agent --count [LIMIT_NUM] [SWEEPID] 
```