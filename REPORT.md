# [프로젝트 #2] 아동 언어 발달 장애 분류 (Gillam)

## 요약

CHILDES TalkBank의 Gillam 데이터셋을 사용하여, 딥러닝 라이브러리 없이(NumPy만) 자가 구현한 MLP로 아동 언어 발달 장애(SLI vs TD) 이진 분류를 수행했다.

`.cha` 파일에서 아동(CHI) 발화 추출 → TF-IDF 벡터화(직접 구현) → 3층 MLP로 분류하는 파이프라인을 구축했다.

최종 검증(Test)에서 **Accuracy 80.88%, F1 Score 64.86%**를 달성했다. 주요 한계는 데이터 불균형(TD >> SLI)과 텍스트 특성만으로의 분류 한계이며, 발화 통계 특징 추가, 앙상블, 데이터 증강을 개선안으로 제시한다.

---

## 1. 서론

- **문제**: 아동의 발화 텍스트를 기반으로 언어 발달 그룹(SLI/TD)을 예측하는 이진 분류
  - **SLI** (Specific Language Impairment): 특정 언어 장애를 가진 아동
  - **TD** (Typically Developing): 정상적으로 언어 발달을 하는 아동
- **제약**: 딥러닝 라이브러리 사용 금지 → PyTorch/TensorFlow/JAX 없이 NumPy로만 모델/학습/추론 전 과정 구현
- **데이터 범위**: CHILDES TalkBank Gillam 데이터셋 (홀수 학번 지정)
- **평가 규칙**: 
  1. Train으로만 학습
  2. Dev로 하이퍼파라미터 튜닝
  3. Test는 최종 성능 측정 전용
- **목표**: 제약 하에서 동작하는 텍스트 분류 파이프라인을 구축하고, Accuracy/F1으로 성능 보고

---

## 2. 데이터 및 재현 절차

### 2.1 데이터셋 다운로드

1. [TalkBank](https://talkbank.org/childes/) 회원 가입
2. [Gillam 데이터셋](https://talkbank.org/childes/access/Clinical-Eng/Gillam.html) 다운로드
3. 프로젝트 폴더에 `gillam/` 디렉토리로 압축 해제

### 2.2 데이터 분할 (제공됨)

| Split | 전체 | SLI | TD | 비율 |
|-------|------|-----|-----|------|
| Train | 540 | 142 | 398 | 8 |
| Dev | 68 | 18 | 50 | 1 |
| Test | 68 | 19 | 49 | 1 |

- Stratification 기준: `group` (SLI/TD)와 `age` (5-11세)
- 각 그룹-나이 조합 내에서 비율을 유지하며 분할

### 2.3 학습/평가 명령

```bash
# 모델 학습
python train.py --epochs 50 --batch_size 32 --learning_rate 0.001

# 하이퍼파라미터 튜닝
python hyperparameter_tuning.py --mode quick

# 최종 평가 (Test 전용)
python evaluate.py --model_path tuning_results/best_tuned_model.npz
```

---

## 3. 방법

### 3.1 텍스트 추출 & 전처리

- `.cha` 파일에서 `*CHI:` 패턴으로 아동 발화만 추출
- `utils.py`의 `extract_utterances()` 함수 활용
- 클리닝: 타임스탬프(`123_456`), 특수기호(`[xxx]`, `&=`), 비언어 표현 제거

### 3.2 TF-IDF 벡터화 (NumPy 직접 구현)

- **어휘 사전 구축**: min_df=2, max_df=0.95 필터링
- **TF 계산**: 문서 내 단어 빈도 / 문서 길이
- **IDF 계산**: log(전체 문서 수 / 단어 포함 문서 수) + 1
- **L2 정규화**: 벡터 정규화로 문서 길이 영향 제거
- **N-gram**: unigram + bigram (1,2) 지원
- **최대 특성 수**: 3,000개

### 3.3 모델 (NumPy 전용 MLP)

**구조:**
```
Input(3000) → Linear(256) → ReLU → Dropout(0.5)
           → Linear(128) → ReLU → Dropout(0.5)
           → Linear(2) → Softmax
```

**구현 요소:**
- **레이어**: Linear, Dropout (직접 구현)
- **활성화**: ReLU, Softmax (직접 구현)
- **손실 함수**: CrossEntropyLoss (클래스 가중치 지원)
- **옵티마이저**: Adam (β1=0.9, β2=0.999, ε=1e-8)
- **초기화**: Xavier/Glorot 초기화

### 3.4 클래스 불균형 처리

- TD(398) vs SLI(142) → 약 2.8:1 불균형
- **클래스 가중치 적용**: 
  - TD: 0.678
  - SLI: 1.901
- CrossEntropyLoss에 가중치 반영

### 3.5 평가 지표

- **Accuracy**: 전체 정확도
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1 Score**: 2 × Precision × Recall / (Precision + Recall)

---

## 4. 실험 세부 & 결과

### 4.1 학습 데이터 통계

```
📊 데이터 로드 완료:
   Train: 540 (SLI: 142, TD: 398)
   Dev: 68 (SLI: 18, TD: 50)
   Test: 68 (SLI: 19, TD: 49)
   
📖 어휘 사전 구축: 3,000 특성
⚖️  클래스 가중치: [0.678, 1.901]
```

### 4.2 하이퍼파라미터 탐색

| 설정 | Dev F1 |
|------|--------|
| hidden=[256,128], lr=0.001, dropout=0.3 | 0.7333 |
| hidden=[256,128], lr=0.001, dropout=0.5 | 0.7333 |
| hidden=[256,128], lr=0.002, dropout=0.3 | 0.7429 |
| **hidden=[256,128], lr=0.002, dropout=0.5** | **0.8125** |
| hidden=[512,256], lr=0.001, dropout=0.3 | 0.7805 |
| hidden=[512,256], lr=0.001, dropout=0.5 | 0.7647 |
| hidden=[512,256], lr=0.002, dropout=0.3 | 0.7879 |
| hidden=[512,256], lr=0.002, dropout=0.5 | 0.7647 |

**Best Config**: hidden_dims=[256,128], learning_rate=0.002, dropout=0.5

### 4.3 최종 성능 (Test)

| 지표 | Train | Dev | Test |
|------|-------|-----|------|
| Accuracy | 94.63% | 86.76% | **80.88%** |
| Precision | 90.07% | 76.47% | **66.67%** |
| Recall | 89.44% | 72.22% | **63.16%** |
| F1 Score | 89.75% | 74.29% | **64.86%** |

### 4.4 Confusion Matrix (Test)

```
                 Predicted
                 TD    SLI
   Actual TD   [ 43     6]
   Actual SLI  [  7    12]
```

- True Negative (TD→TD): 43
- False Positive (TD→SLI): 6
- False Negative (SLI→TD): 7
- True Positive (SLI→SLI): 12

### 4.5 Classification Report (Test)

```
               Precision    Recall   F1-Score
---------------------------------------------
TD                0.8600    0.8776    0.8687
SLI               0.6667    0.6316    0.6486
---------------------------------------------
Accuracy          0.8088
Macro Avg         0.7633    0.7546    0.7587
```

---

## 5. 논의

### 5.1 무엇이 성능을 제한했나

1. **데이터 불균형**: TD가 SLI보다 2.8배 많음 → 클래스 가중치로 완화했으나 여전히 SLI 인식률 낮음
2. **텍스트만으로의 한계**: 발화 내용만으로는 SLI/TD 구분이 어려움 → 발화 패턴, 문법 오류 빈도 등 추가 특징 필요
3. **데이터 크기**: Train 540개는 딥러닝에 부족 → 간단한 MLP에도 과적합 경향
4. **NumPy 구현 제약**: 배치 정규화, 복잡한 아키텍처 구현 어려움

### 5.2 무엇이 효과적이었나

1. **TF-IDF 벡터화**: BoW 대비 문서 특성을 잘 반영
2. **클래스 가중치**: 불균형 완화에 기여 (가중치 없으면 SLI 거의 예측 못함)
3. **Dropout(0.5)**: 과적합 방지에 효과적, 0.3 대비 성능 향상
4. **Adam 옵티마이저**: SGD 대비 빠른 수렴

### 5.3 오류 분석

- **False Negative (SLI→TD) 7건**: SLI 아동이 정상 발화 패턴을 보이는 경우
- **False Positive (TD→SLI) 6건**: TD 아동의 비정형 발화가 SLI로 오분류

---

## 6. 결론

- 딥러닝 라이브러리 없이(NumPy 전용) TF-IDF + MLP 전 과정을 구축했고, Gillam Test 세트에서 **Accuracy 80.88%, F1 64.86%**를 달성했다.
- 핵심 한계는 **데이터 불균형**과 **텍스트 특성만으로의 분류 한계**다.
- 제약(라이브러리 금지) 내에서 파이프라인이 동작함을 확인했으며, 특징 공학 및 데이터 증강 시 추가 개선 여지가 크다.

---

## 7. 향후 개선 (우선순위 제안)

1. **발화 통계 특징 추가**: MLU(평균 발화 길이), TTR(어휘 다양성), 문법 오류 빈도 등
2. **데이터 증강**: 동의어 치환, 단어 순서 섞기, Back-translation
3. **앙상블**: 여러 모델(MLP, SVM, Random Forest) 결합
4. **언어모델 특징**: N-gram 확률, Perplexity 등 활용
5. **CNN/LSTM 직접 구현**: 시퀀스 패턴 포착을 위한 더 복잡한 아키텍처

---

## 8. 실험 재현 체크리스트

- [ ] Gillam 데이터셋 다운로드 및 `gillam/` 폴더에 압축 해제
- [ ] `split/gillam_*.csv` 파일로 Train/Dev/Test 분할 확인
- [ ] 학습/튜닝은 Train+Dev만, 최종 점수는 Test만
- [ ] NumPy 외 딥러닝 라이브러리 미사용 확인
- [ ] `evaluate.py`로 최종 성능 측정

---

## 9. 한계 및 리스크

- F1 Score는 클래스 불균형에 민감하여, 데이터 분포가 다르면 점수 변동이 큼
- 현재 수치(F1 ≈ 0.65)는 해당 Test 세트에 특화된 결과일 수 있음
- 실제 임상 적용 시 더 다양한 데이터셋에서 검증 필요

---

## 10. GitHub 저장소

- https://github.com/gimogibumo/deeplearning2

---

## 11. 참고

- CHILDES TalkBank: https://talkbank.org/childes/
- Gillam 데이터셋: https://talkbank.org/childes/access/Clinical-Eng/Gillam.html
- (내부 구현) NumPy MLP, TF-IDF 벡터화, Adam 옵티마이저
- (일반) Xavier 초기화, Dropout, CrossEntropyLoss

---

## 부록 A. 하이퍼파라미터 (최종)

| 항목 | 값 |
|------|-----|
| Hidden dims | [256, 128] |
| Dropout | 0.5 |
| Learning rate | 0.002 |
| Batch size | 32 |
| Epochs | 30 |
| Max features | 3,000 |
| Vectorizer | TF-IDF (unigram + bigram) |
| Optimizer | Adam (β1=0.9, β2=0.999) |
| Class weights | [0.678, 1.901] |

---

## 부록 B. 프로젝트 구조

```
slp/
├── numpy_nn.py              # NumPy 기반 신경망 라이브러리
│   ├── Layer (Linear, Embedding, Dropout)
│   ├── Activations (ReLU, Sigmoid, Tanh, Softmax)
│   ├── Loss (CrossEntropyLoss)
│   ├── Optimizers (SGD, Adam)
│   └── MLP, Sequential
├── data_preprocessing.py    # 데이터 전처리
│   ├── TfidfVectorizer (직접 구현)
│   ├── BagOfWordsVectorizer
│   └── GillamDataset
├── train.py                 # 모델 학습
├── hyperparameter_tuning.py # 하이퍼파라미터 튜닝
├── evaluate.py              # Test 세트 최종 평가
├── utils.py                 # .cha 파일 파싱
├── requirements.txt         # 의존성 (numpy, pandas)
├── split/                   # Train/Dev/Test 분할 파일
├── checkpoints/             # 학습된 모델
├── tuning_results/          # 튜닝 결과
└── results/                 # 최종 평가 결과
```

---

## 부록 C. 커밋 히스토리

```
9f82e58 Step 5: Test 세트 최종 평가
bdfe5d0 Step 4: 하이퍼파라미터 튜닝
ca81309 Step 3: 모델 학습 스크립트 작성
cb2bf24 Step 2: 데이터 전처리 파이프라인 구축
3b4e1fb Step 1: numpy 기반 신경망 라이브러리 구현
```

---

## 부록 D. 주요 구현 코드 설명

### D.1 TF-IDF 벡터화 (NumPy)

```python
# TF 계산
tf_matrix = word_counts / doc_lengths

# IDF 계산
idf = np.log(n_docs / doc_freq) + 1

# TF-IDF
tfidf = tf_matrix * idf

# L2 정규화
tfidf = tfidf / np.linalg.norm(tfidf, axis=1, keepdims=True)
```

### D.2 Adam 옵티마이저 (NumPy)

```python
# 1차 모멘트 (평균)
m = β1 * m + (1 - β1) * grad

# 2차 모멘트 (분산)
v = β2 * v + (1 - β2) * (grad ** 2)

# 편향 보정
m_hat = m / (1 - β1 ** t)
v_hat = v / (1 - β2 ** t)

# 파라미터 업데이트
param -= lr * m_hat / (sqrt(v_hat) + ε)
```

### D.3 CrossEntropyLoss (클래스 가중치)

```python
# Softmax
probs = exp(logits) / sum(exp(logits))

# Weighted NLL
loss = -mean(weights[targets] * log(probs[targets]))

# Gradient
grad = probs.copy()
grad[targets] -= 1
grad *= weights[targets]
```

