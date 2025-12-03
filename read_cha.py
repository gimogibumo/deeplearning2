# 🔥 테스트 + 사용 예시
from utils import count_utterance_by_speaker, extract_utterances


cha_filename = "ENNI/SLI/A/413.cha"

# 발화 분포 카운트
count_utterance = count_utterance_by_speaker(cha_filename)
print("📊 발화 분포:", count_utterance)

# 발화 추출
speakers = list(count_utterance.keys())
utts = extract_utterances(cha_filename, speakers)
print(f"📊 {len(utts)}개 발화 추출")
for i, utt in enumerate(utts):
    #print(f"{i+1}. {utt.speaker}: {utt.text}")
    print(f"{i+1}. {utt.speaker}: {utt.clean_text}")

    if i == 10:
        break

