## Datasets
* CHILDES datasets: https://talkbank.org/childes/ (여기서 회원 가입)
* ENNI: https://talkbank.org/childes/access/Clinical-Eng/ENNI.html
* Gillam: https://talkbank.org/childes/access/Clinical-Eng/Gillam.html

**학번이 짝수인 사람은 ENNI, 홀수인 사람은 Gillam**

### Split

데이터셋은 train/validation/test로 8:1:1 비율로 stratified split

**ENNI**
- Stratification 기준: `group` (SLI/TD)와 `sub_group` (A/B)
- 각 그룹 조합(SLI-A, SLI-B, TD-A, TD-B) 내에서 비율을 유지하며 분할
- `split_enni.py` 실행 
- 결과 `split/ENNI_train.csv`, `split/ENNI_dev.csv`, `split/ENNI_test.csv`

**Gillam**
- Stratification 기준: `group` (SLI/TD)와 `age` (5-11세)
- 각 그룹-나이 조합 내에서 비율을 유지하며 분할
- SLI 그룹의 11세 샘플이 부족하여 SLI-10으로 매핑하여 처리
- `split_gillam.py` 실행 
- 결과: `split/gillam_train.csv`, `split/gillam_dev.csv`, `split/gillam_test.csv`


## Helper functions ('utils.py') 사용법

### count_utterance_by_speaker
`utils.count_utterance_by_speaker("path/to/file.cha")` 를 호출하면 실제 발화가 존재하는 화자만 키로, 발화 수를 값으로 갖는 dict 를 돌려줍니다. 기본적인 사용 예시는 다음과 같습니다.

```
from utils import count_utterance_by_speaker

count_utterance = count_utterance_by_speaker("ENNI/SLI/A/413.cha")
print(count_utterance)  # {'CHI': 124, 'EXA': 103, ...}
```

### extract_utterances
특정 화자들의 발화를 순서대로 가져오고 싶으면 `utils.extract_utterances("file.cha", ["CHI", "EXA"])` 를 사용합니다. 반환값은 `Utterance(order, speaker, text, clean_text)` dataclass 리스트입니다.

```
from utils import extract_utterances

utterances = extract_utterances("ENNI/SLI/A/413.cha", ["CHI", "EXA"])
for utt in utterances[:3]:
    print(utt.order, utt.speaker, utt.clean_text)
```

## `.cha` 파일 읽는 예
```
python read_cha.py
```
```
📊 발화 분포: {'CHI': 72, 'EXA': 22}

📊 94개 발화 추출

1. EXA: okay Firstname .
2. EXA: look at these pictures and tell me what's happening .
3. CHI: I shopping .
4. CHI: and get cookie .
5. CHI: get . one cookie .
6. CHI: and fall off .
7. CHI: and then angry .
8. EXA: and ?
9. CHI: all done.
10. EXA: so .
```

## Goal
아동의 발화 텍스트를 기반으로 언어 발달 그룹을 예측하는 이진 분류(binary classification) 수행

- **입력**: `.cha` 파일
- **출력**: `group` 예측
  - **SLI** (Specific Language Impairment): 특정 언어 장애를 가진 아동
  - **TD** (Typically Developing): 정상적으로 언어 발달을 하는 아동

## 수행 방법

`split/` 폴더 아래에 제공된 train/dev/test split 파일들을 사용:

- **Hyperparameter Tuning**: `split/*_train.csv`와 `split/*_dev.csv`를 사용하여 모델의 하이퍼파라미터를 튜닝
- **Test Evaluation**: 최종 모델 성능 평가는 `split/*_test.csv`를 사용하여 수행
