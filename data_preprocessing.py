"""
Gillam 데이터셋 전처리 모듈
- .cha 파일에서 아동 발화 텍스트 추출
- TF-IDF 또는 Bag-of-Words 벡터화
- Train/Dev/Test 데이터 준비
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from collections import Counter
import re

from utils import extract_utterances


# ============================================================================
# 텍스트 추출
# ============================================================================

def extract_child_text(cha_file: str, base_dir: str = ".") -> str:
    """
    .cha 파일에서 아동(CHI) 발화만 추출하여 하나의 텍스트로 결합
    
    Args:
        cha_file: 파일 경로 (예: gillam/SLI/5f/55697il-l.cha)
        base_dir: 기준 디렉토리
    
    Returns:
        아동 발화를 결합한 텍스트
    """
    file_path = Path(base_dir) / cha_file
    
    if not file_path.exists():
        print(f"⚠️  파일 없음: {file_path}")
        return ""
    
    try:
        utterances = extract_utterances(str(file_path), ["CHI"])
        texts = [utt.clean_text for utt in utterances if utt.clean_text.strip()]
        return " ".join(texts)
    except Exception as e:
        print(f"❌ 파일 처리 오류 ({cha_file}): {e}")
        return ""


def load_split_data(split_dir: str = "split") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train/Dev/Test split 파일 로드"""
    train_df = pd.read_csv(f"{split_dir}/gillam_train.csv")
    dev_df = pd.read_csv(f"{split_dir}/gillam_dev.csv")
    test_df = pd.read_csv(f"{split_dir}/gillam_test.csv")
    
    print(f"📊 데이터 로드 완료:")
    print(f"   Train: {len(train_df)} (SLI: {sum(train_df['group']=='SLI')}, TD: {sum(train_df['group']=='TD')})")
    print(f"   Dev: {len(dev_df)} (SLI: {sum(dev_df['group']=='SLI')}, TD: {sum(dev_df['group']=='TD')})")
    print(f"   Test: {len(test_df)} (SLI: {sum(test_df['group']=='SLI')}, TD: {sum(test_df['group']=='TD')})")
    
    return train_df, dev_df, test_df


def prepare_texts_and_labels(df: pd.DataFrame, base_dir: str = ".") -> Tuple[List[str], np.ndarray]:
    """
    DataFrame에서 텍스트와 레이블 추출
    
    Returns:
        texts: 아동 발화 텍스트 리스트
        labels: 레이블 배열 (SLI=1, TD=0)
    """
    texts = []
    labels = []
    
    for idx, row in df.iterrows():
        text = extract_child_text(row['filename'], base_dir)
        if text:
            texts.append(text)
            labels.append(1 if row['group'] == 'SLI' else 0)
    
    return texts, np.array(labels)


# ============================================================================
# 텍스트 벡터화 (TF-IDF)
# ============================================================================

class TfidfVectorizer:
    """
    TF-IDF 벡터화 (scikit-learn 없이 순수 구현)
    """
    
    def __init__(
        self,
        max_features: int = 5000,
        min_df: int = 2,
        max_df: float = 0.95,
        ngram_range: Tuple[int, int] = (1, 2)
    ):
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.ngram_range = ngram_range
        
        self.vocabulary_ = {}
        self.idf_ = None
        self.feature_names_ = []
    
    def _tokenize(self, text: str) -> List[str]:
        """텍스트를 토큰으로 분리"""
        # 소문자 변환 및 특수문자 제거
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        words = text.split()
        
        tokens = []
        # Unigrams
        if self.ngram_range[0] <= 1 <= self.ngram_range[1]:
            tokens.extend(words)
        
        # Bigrams
        if self.ngram_range[0] <= 2 <= self.ngram_range[1]:
            for i in range(len(words) - 1):
                tokens.append(f"{words[i]}_{words[i+1]}")
        
        return tokens
    
    def fit(self, texts: List[str]):
        """어휘 사전 및 IDF 계산"""
        n_docs = len(texts)
        
        # 문서 빈도 계산
        doc_freq = Counter()
        for text in texts:
            unique_tokens = set(self._tokenize(text))
            doc_freq.update(unique_tokens)
        
        # min_df, max_df 필터링
        max_doc_count = int(self.max_df * n_docs) if isinstance(self.max_df, float) else self.max_df
        min_doc_count = self.min_df
        
        filtered_tokens = {
            token: freq for token, freq in doc_freq.items()
            if min_doc_count <= freq <= max_doc_count
        }
        
        # 빈도순 정렬 후 max_features 개수만큼 선택
        sorted_tokens = sorted(filtered_tokens.items(), key=lambda x: -x[1])
        selected_tokens = sorted_tokens[:self.max_features]
        
        # 어휘 사전 구축
        self.vocabulary_ = {token: idx for idx, (token, _) in enumerate(selected_tokens)}
        self.feature_names_ = [token for token, _ in selected_tokens]
        
        # IDF 계산: log(N / df) + 1
        self.idf_ = np.zeros(len(self.vocabulary_))
        for token, idx in self.vocabulary_.items():
            df = doc_freq[token]
            self.idf_[idx] = np.log(n_docs / df) + 1
        
        print(f"📖 어휘 사전 구축: {len(self.vocabulary_)} 특성")
        return self
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """텍스트를 TF-IDF 벡터로 변환"""
        n_docs = len(texts)
        n_features = len(self.vocabulary_)
        
        # TF 계산
        tf_matrix = np.zeros((n_docs, n_features))
        
        for i, text in enumerate(texts):
            tokens = self._tokenize(text)
            token_counts = Counter(tokens)
            
            for token, count in token_counts.items():
                if token in self.vocabulary_:
                    idx = self.vocabulary_[token]
                    tf_matrix[i, idx] = count
        
        # TF 정규화 (문서 길이로 나누기)
        doc_lengths = tf_matrix.sum(axis=1, keepdims=True)
        doc_lengths[doc_lengths == 0] = 1  # 0 나눔 방지
        tf_matrix = tf_matrix / doc_lengths
        
        # TF-IDF 계산
        tfidf_matrix = tf_matrix * self.idf_
        
        # L2 정규화
        norms = np.linalg.norm(tfidf_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1
        tfidf_matrix = tfidf_matrix / norms
        
        return tfidf_matrix
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """fit과 transform을 한 번에"""
        self.fit(texts)
        return self.transform(texts)


class BagOfWordsVectorizer:
    """
    Bag-of-Words 벡터화 (더 간단한 방법)
    """
    
    def __init__(self, max_features: int = 5000, min_df: int = 2):
        self.max_features = max_features
        self.min_df = min_df
        self.vocabulary_ = {}
    
    def _tokenize(self, text: str) -> List[str]:
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        return text.split()
    
    def fit(self, texts: List[str]):
        word_counts = Counter()
        for text in texts:
            word_counts.update(self._tokenize(text))
        
        # min_df 필터링 및 상위 max_features 선택
        filtered = {w: c for w, c in word_counts.items() if c >= self.min_df}
        sorted_words = sorted(filtered.items(), key=lambda x: -x[1])[:self.max_features]
        
        self.vocabulary_ = {word: idx for idx, (word, _) in enumerate(sorted_words)}
        print(f"📖 어휘 사전 구축: {len(self.vocabulary_)} 특성")
        return self
    
    def transform(self, texts: List[str]) -> np.ndarray:
        n_docs = len(texts)
        n_features = len(self.vocabulary_)
        
        bow_matrix = np.zeros((n_docs, n_features))
        
        for i, text in enumerate(texts):
            for word in self._tokenize(text):
                if word in self.vocabulary_:
                    bow_matrix[i, self.vocabulary_[word]] += 1
        
        # 정규화
        norms = np.linalg.norm(bow_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1
        bow_matrix = bow_matrix / norms
        
        return bow_matrix
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        self.fit(texts)
        return self.transform(texts)


# ============================================================================
# 데이터셋 클래스
# ============================================================================

class GillamDataset:
    """Gillam 데이터셋 관리"""
    
    def __init__(
        self,
        vectorizer_type: str = "tfidf",
        max_features: int = 3000,
        base_dir: str = "."
    ):
        self.vectorizer_type = vectorizer_type
        self.max_features = max_features
        self.base_dir = base_dir
        
        if vectorizer_type == "tfidf":
            self.vectorizer = TfidfVectorizer(max_features=max_features)
        else:
            self.vectorizer = BagOfWordsVectorizer(max_features=max_features)
        
        self.X_train = None
        self.y_train = None
        self.X_dev = None
        self.y_dev = None
        self.X_test = None
        self.y_test = None
    
    def load_and_preprocess(self, split_dir: str = "split"):
        """데이터 로드 및 전처리"""
        
        print("\n📂 데이터 로드 중...")
        train_df, dev_df, test_df = load_split_data(split_dir)
        
        print("\n📝 텍스트 추출 중...")
        train_texts, self.y_train = prepare_texts_and_labels(train_df, self.base_dir)
        dev_texts, self.y_dev = prepare_texts_and_labels(dev_df, self.base_dir)
        test_texts, self.y_test = prepare_texts_and_labels(test_df, self.base_dir)
        
        print(f"   Train: {len(train_texts)} 샘플")
        print(f"   Dev: {len(dev_texts)} 샘플")
        print(f"   Test: {len(test_texts)} 샘플")
        
        if len(train_texts) == 0:
            raise ValueError("텍스트를 추출할 수 없습니다. gillam 데이터셋이 있는지 확인하세요.")
        
        print(f"\n🔧 {self.vectorizer_type.upper()} 벡터화 중...")
        self.X_train = self.vectorizer.fit_transform(train_texts)
        self.X_dev = self.vectorizer.transform(dev_texts)
        self.X_test = self.vectorizer.transform(test_texts)
        
        print(f"   특성 차원: {self.X_train.shape[1]}")
        
        return self
    
    def get_train_data(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.X_train, self.y_train
    
    def get_dev_data(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.X_dev, self.y_dev
    
    def get_test_data(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.X_test, self.y_test
    
    def get_class_weights(self) -> np.ndarray:
        """클래스 불균형 처리를 위한 가중치 계산"""
        class_counts = np.bincount(self.y_train)
        total = len(self.y_train)
        weights = total / (len(class_counts) * class_counts)
        return weights.astype(np.float64)


# ============================================================================
# 배치 생성기
# ============================================================================

def create_batches(
    X: np.ndarray, 
    y: np.ndarray, 
    batch_size: int, 
    shuffle: bool = True
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    데이터를 배치로 분할
    
    Args:
        X: 특성 행렬 (n_samples, n_features)
        y: 레이블 배열 (n_samples,)
        batch_size: 배치 크기
        shuffle: 셔플 여부
    
    Returns:
        배치 리스트 [(X_batch, y_batch), ...]
    """
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    
    if shuffle:
        np.random.shuffle(indices)
    
    batches = []
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_indices = indices[start_idx:end_idx]
        batches.append((X[batch_indices], y[batch_indices]))
    
    return batches


if __name__ == "__main__":
    # 테스트
    print("🧪 데이터 전처리 테스트")
    
    dataset = GillamDataset(vectorizer_type="tfidf", max_features=3000, base_dir=".")
    dataset.load_and_preprocess(split_dir="split")
    
    X_train, y_train = dataset.get_train_data()
    X_dev, y_dev = dataset.get_dev_data()
    X_test, y_test = dataset.get_test_data()
    
    print(f"\n📊 데이터 형태:")
    print(f"   X_train: {X_train.shape}")
    print(f"   X_dev: {X_dev.shape}")
    print(f"   X_test: {X_test.shape}")
    
    print(f"\n⚖️  클래스 가중치: {dataset.get_class_weights()}")
    
    # 배치 테스트
    batches = create_batches(X_train, y_train, batch_size=32, shuffle=True)
    print(f"\n📦 배치 수: {len(batches)}")
    print(f"   첫 번째 배치: X={batches[0][0].shape}, y={batches[0][1].shape}")
    
    print("\n✅ 테스트 완료!")

