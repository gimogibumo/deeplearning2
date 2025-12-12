"""
NumPy 기반 신경망 라이브러리
PyTorch/TensorFlow 없이 순수 NumPy로 구현

Components:
- Layers: Linear, Embedding, Dropout
- Activations: ReLU, Sigmoid, Softmax, Tanh
- Loss: CrossEntropyLoss
- Optimizers: SGD, Adam
"""

import numpy as np
from typing import List, Tuple, Optional, Dict


# ============================================================================
# 기본 레이어 클래스
# ============================================================================

class Layer:
    """모든 레이어의 기본 클래스"""
    
    def __init__(self):
        self.params = {}
        self.grads = {}
        self.cache = {}
        self.training = True
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    def backward(self, grad: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    def train(self):
        self.training = True
    
    def eval(self):
        self.training = False


class Linear(Layer):
    """
    Fully Connected Layer (선형 변환)
    y = xW + b
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        
        # Xavier/Glorot 초기화
        scale = np.sqrt(2.0 / (in_features + out_features))
        self.params['W'] = np.random.randn(in_features, out_features) * scale
        
        if bias:
            self.params['b'] = np.zeros(out_features)
        
        self.use_bias = bias
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Args:
            x: (batch_size, in_features)
        Returns:
            out: (batch_size, out_features)
        """
        self.cache['x'] = x
        out = x @ self.params['W']
        if self.use_bias:
            out = out + self.params['b']
        return out
    
    def backward(self, grad: np.ndarray) -> np.ndarray:
        """
        Args:
            grad: (batch_size, out_features) - 상위 레이어에서 전파된 그래디언트
        Returns:
            dx: (batch_size, in_features) - 하위 레이어로 전파할 그래디언트
        """
        x = self.cache['x']
        batch_size = x.shape[0]
        
        # 파라미터 그래디언트
        self.grads['W'] = x.T @ grad / batch_size
        if self.use_bias:
            self.grads['b'] = np.mean(grad, axis=0)
        
        # 입력에 대한 그래디언트
        dx = grad @ self.params['W'].T
        return dx


class Embedding(Layer):
    """
    임베딩 레이어 (lookup table)
    단어 인덱스 -> 벡터
    """
    
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int = 0):
        super().__init__()
        
        # 임베딩 가중치 초기화
        self.params['W'] = np.random.randn(num_embeddings, embedding_dim) * 0.01
        self.params['W'][padding_idx] = 0  # 패딩은 0 벡터
        
        self.padding_idx = padding_idx
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Args:
            x: (batch_size, seq_len) - 단어 인덱스
        Returns:
            out: (batch_size, seq_len, embedding_dim)
        """
        self.cache['x'] = x
        return self.params['W'][x]
    
    def backward(self, grad: np.ndarray) -> np.ndarray:
        """
        Args:
            grad: (batch_size, seq_len, embedding_dim)
        Returns:
            None (임베딩은 입력이 인덱스이므로)
        """
        x = self.cache['x']
        
        # 임베딩 가중치 그래디언트
        self.grads['W'] = np.zeros_like(self.params['W'])
        np.add.at(self.grads['W'], x, grad)
        self.grads['W'] /= x.shape[0]
        
        # 패딩 인덱스 그래디언트는 0
        self.grads['W'][self.padding_idx] = 0
        
        return None  # 인덱스에 대한 그래디언트는 없음


class Dropout(Layer):
    """
    드롭아웃 레이어 (정규화)
    """
    
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p  # 드롭 확률
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        if self.training and self.p > 0:
            self.cache['mask'] = (np.random.rand(*x.shape) > self.p).astype(np.float64)
            return x * self.cache['mask'] / (1 - self.p)
        return x
    
    def backward(self, grad: np.ndarray) -> np.ndarray:
        if self.training and self.p > 0:
            return grad * self.cache['mask'] / (1 - self.p)
        return grad


# ============================================================================
# 활성화 함수
# ============================================================================

class ReLU(Layer):
    """ReLU 활성화 함수"""
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.cache['x'] = x
        return np.maximum(0, x)
    
    def backward(self, grad: np.ndarray) -> np.ndarray:
        x = self.cache['x']
        return grad * (x > 0).astype(np.float64)


class Sigmoid(Layer):
    """Sigmoid 활성화 함수"""
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        # 수치 안정성
        x = np.clip(x, -500, 500)
        out = 1 / (1 + np.exp(-x))
        self.cache['out'] = out
        return out
    
    def backward(self, grad: np.ndarray) -> np.ndarray:
        out = self.cache['out']
        return grad * out * (1 - out)


class Tanh(Layer):
    """Tanh 활성화 함수"""
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        out = np.tanh(x)
        self.cache['out'] = out
        return out
    
    def backward(self, grad: np.ndarray) -> np.ndarray:
        out = self.cache['out']
        return grad * (1 - out ** 2)


class Softmax(Layer):
    """Softmax 활성화 함수 (주로 출력층에서 사용)"""
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        # 수치 안정성을 위해 최대값 빼기
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        out = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
        self.cache['out'] = out
        return out
    
    def backward(self, grad: np.ndarray) -> np.ndarray:
        # CrossEntropyLoss와 함께 사용 시 간단해짐
        return grad


# ============================================================================
# 손실 함수
# ============================================================================

class CrossEntropyLoss:
    """
    크로스 엔트로피 손실 함수
    Softmax + NLLLoss 결합
    """
    
    def __init__(self, class_weights: np.ndarray = None):
        self.class_weights = class_weights
    
    def forward(self, logits: np.ndarray, targets: np.ndarray) -> float:
        """
        Args:
            logits: (batch_size, num_classes) - 모델 출력 (softmax 전)
            targets: (batch_size,) - 정답 레이블
        Returns:
            loss: 스칼라 손실값
        """
        batch_size = logits.shape[0]
        
        # Softmax
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        self.cache = {'probs': probs, 'targets': targets, 'batch_size': batch_size}
        
        # 수치 안정성
        probs_clipped = np.clip(probs, 1e-15, 1 - 1e-15)
        
        # Negative Log Likelihood
        log_probs = -np.log(probs_clipped[np.arange(batch_size), targets])
        
        # 클래스 가중치 적용
        if self.class_weights is not None:
            weights = self.class_weights[targets]
            loss = np.mean(log_probs * weights)
        else:
            loss = np.mean(log_probs)
        
        return loss
    
    def backward(self) -> np.ndarray:
        """
        Returns:
            grad: (batch_size, num_classes)
        """
        probs = self.cache['probs']
        targets = self.cache['targets']
        batch_size = self.cache['batch_size']
        
        grad = probs.copy()
        grad[np.arange(batch_size), targets] -= 1
        
        # 클래스 가중치 적용
        if self.class_weights is not None:
            weights = self.class_weights[targets].reshape(-1, 1)
            grad = grad * weights
        
        return grad / batch_size


# ============================================================================
# 옵티마이저
# ============================================================================

class SGD:
    """확률적 경사 하강법 (모멘텀 포함)"""
    
    def __init__(self, params: List[Dict], lr: float = 0.01, momentum: float = 0.9):
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.velocity = {}
    
    def step(self, layers: List[Layer]):
        for i, layer in enumerate(layers):
            for name, param in layer.params.items():
                if name not in layer.grads:
                    continue
                
                key = f"{i}_{name}"
                if key not in self.velocity:
                    self.velocity[key] = np.zeros_like(param)
                
                # 모멘텀 업데이트
                self.velocity[key] = self.momentum * self.velocity[key] - self.lr * layer.grads[name]
                layer.params[name] += self.velocity[key]
    
    def zero_grad(self, layers: List[Layer]):
        for layer in layers:
            layer.grads = {}


class Adam:
    """Adam 옵티마이저"""
    
    def __init__(self, lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {}  # 1차 모멘트
        self.v = {}  # 2차 모멘트
        self.t = 0   # 타임스텝
    
    def step(self, layers: List[Layer]):
        self.t += 1
        
        for i, layer in enumerate(layers):
            for name, param in layer.params.items():
                if name not in layer.grads:
                    continue
                
                key = f"{i}_{name}"
                grad = layer.grads[name]
                
                # 모멘트 초기화
                if key not in self.m:
                    self.m[key] = np.zeros_like(param)
                    self.v[key] = np.zeros_like(param)
                
                # 모멘트 업데이트
                self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grad
                self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grad ** 2)
                
                # 편향 보정
                m_hat = self.m[key] / (1 - self.beta1 ** self.t)
                v_hat = self.v[key] / (1 - self.beta2 ** self.t)
                
                # 파라미터 업데이트
                layer.params[name] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
    
    def zero_grad(self, layers: List[Layer]):
        for layer in layers:
            layer.grads = {}


# ============================================================================
# 신경망 모델 클래스
# ============================================================================

class Sequential:
    """레이어들을 순차적으로 연결한 신경망"""
    
    def __init__(self, layers: List[Layer]):
        self.layers = layers
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, grad: np.ndarray):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
    
    def train(self):
        for layer in self.layers:
            layer.train()
    
    def eval(self):
        for layer in self.layers:
            layer.eval()
    
    def get_layers(self) -> List[Layer]:
        return self.layers


class MLP:
    """
    Multi-Layer Perceptron (다층 퍼셉트론)
    텍스트 분류를 위한 기본 신경망
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        dropout: float = 0.3
    ):
        layers = []
        
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(Linear(prev_dim, hidden_dim))
            layers.append(ReLU())
            layers.append(Dropout(dropout))
            prev_dim = hidden_dim
        
        layers.append(Linear(prev_dim, output_dim))
        
        self.model = Sequential(layers)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.model.forward(x)
    
    def backward(self, grad: np.ndarray):
        self.model.backward(grad)
    
    def train(self):
        self.model.train()
    
    def eval(self):
        self.model.eval()
    
    def get_layers(self) -> List[Layer]:
        return self.model.get_layers()


# ============================================================================
# 유틸리티 함수
# ============================================================================

def accuracy(predictions: np.ndarray, targets: np.ndarray) -> float:
    """정확도 계산"""
    return np.mean(predictions == targets)


def save_model(layers: List[Layer], filepath: str):
    """모델 저장"""
    state = {}
    for i, layer in enumerate(layers):
        for name, param in layer.params.items():
            state[f"layer_{i}_{name}"] = param
    np.savez(filepath, **state)


def load_model(layers: List[Layer], filepath: str):
    """모델 로드"""
    state = np.load(filepath)
    for i, layer in enumerate(layers):
        for name in layer.params.keys():
            key = f"layer_{i}_{name}"
            if key in state:
                layer.params[name] = state[key]


if __name__ == "__main__":
    # 테스트
    print("🧪 NumPy 신경망 라이브러리 테스트")
    
    # MLP 테스트
    np.random.seed(42)
    
    # 더미 데이터
    X = np.random.randn(32, 100)  # 32 샘플, 100 특성
    y = np.random.randint(0, 2, 32)  # 이진 분류
    
    # 모델 생성
    model = MLP(input_dim=100, hidden_dims=[64, 32], output_dim=2, dropout=0.3)
    criterion = CrossEntropyLoss()
    optimizer = Adam(lr=0.001)
    
    # 학습 루프
    model.train()
    for epoch in range(10):
        # Forward
        logits = model.forward(X)
        loss = criterion.forward(logits, y)
        
        # Backward
        grad = criterion.backward()
        model.backward(grad)
        
        # Update
        optimizer.step(model.get_layers())
        optimizer.zero_grad(model.get_layers())
        
        # Accuracy
        preds = np.argmax(logits, axis=1)
        acc = accuracy(preds, y)
        
        print(f"Epoch {epoch+1}: Loss={loss:.4f}, Accuracy={acc:.4f}")
    
    print("\n✅ 테스트 완료!")

