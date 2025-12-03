import re
from pathlib import Path
from typing import List
from dataclasses import dataclass

def count_utterance_by_speaker(cha_file: str) -> dict:
    """
    실제 발화한 화자만 반환 (0 발화 제외)
    """
    content = Path(cha_file).read_text(errors='ignore')
    
    # 실제 *SPEAKER 발화만
    speaker_utts = re.findall(r'\*([A-Z][A-Za-z ]+?):\s*(.*?)(?=\n\*[A-Z][A-Za-z ]+?:|\n%|\n@|\Z)', 
                             content, re.DOTALL | re.I)
    
    active_speakers = {}
    for speaker, text in speaker_utts:
        # 클리닝 후 길이 체크
        text = re.sub(r'\d+_\d+|\[\w+\]|\b\d+\b|\bxxx\b', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        if len(text) > 1:
            if speaker not in active_speakers:
                active_speakers[speaker] = 0
            active_speakers[speaker] += 1
    
    # 발화수 순 정렬
    sorted_speakers = dict(sorted(active_speakers.items(), key=lambda x: x[1], reverse=True))
    
    return sorted_speakers

# # 테스트
# speakers = count_utterance_by_speaker("ENNI/SLI/A/413.cha")
# # print(f"👥 화자 {list(speakers.keys())}")
# print("📊 발화 분포:", speakers)


@dataclass
class Utterance:
    order: int
    speaker: str
    text: str
    clean_text: str

def clean(text: str) -> str:
    """타임스탬프 + 모든 특수기호 제거"""
    
    # 1. 타임스탬프 123_456
    text = re.sub(r'\d+_\d+', '', text)
    
    # 2. 나머지 특수기호
    text = re.sub(r'\[[^\]]*\]|\b(?:xxx|www|0)\b|[/]|[/=]|&=', '', text)
    text = re.sub(r'[^\w\s\.\,\!\?\-\']', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def extract_utterances(cha_file: str, speakers: List[str]) -> List[Utterance]:
    """
    지정 화자들의 발화만 순서대로 추출
    
    Args:
        cha_file: '413.cha'
        speakers: ['CHI', 'MOT'] 또는 ['CHI', 'EXA']
    
    Returns:
        List[Utterance]: 순서 유지된 발화 리스트
    """
    content = Path(cha_file).read_text(errors='ignore')
    
    # 지정 화자 패턴 (대소문자 무시)
    speaker_pattern = '|'.join([re.escape(s) for s in speakers])
    pattern = rf'\*({speaker_pattern}):\s*(.*?)(?=\n\*[A-Z][A-Za-z ]+?:|\n%|\n@|\Z)'
    
    matches = list(re.finditer(pattern, content, re.DOTALL | re.I))
    
    utterances = []
    for i, match in enumerate(matches, 1):
        speaker = match.group(1).strip()
        raw_text = match.group(2).strip()
        
        # 클리닝
        clean_text = clean(raw_text)
        
        if len(clean_text) > 1:  # 의미있는 발화만
            utterances.append(Utterance(
                order=i,
                speaker=speaker,
                text=raw_text,
                clean_text=clean_text
            ))
    
    return utterances

