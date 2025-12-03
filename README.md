## Datasets

### ENNI
* https://talkbank.org/childes/access/Clinical-Eng/ENNI.html

### Gillam
* https://talkbank.org/childes/access/Clinical-Eng/Gillam.html

## Utils 사용법

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

## read_cha.py 실행 예시

```
$ python read_cha.py
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
11. EXA: okay Firstname tell me the story that you see in these pictures .
```
