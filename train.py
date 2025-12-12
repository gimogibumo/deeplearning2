"""
Gillam SLI vs TD 이진 분류 모델 학습 스크립트
- NumPy 기반 MLP 모델 학습
- Train/Dev 세트로 학습 및 검증
- 최적 모델 저장
"""

import numpy as np
import argparse
import json
from pathlib import Path
from tqdm import tqdm

from numpy_nn import MLP, CrossEntropyLoss, Adam, SGD, accuracy, save_model, load_model
from data_preprocessing import GillamDataset, create_batches


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    분류 성능 지표 계산
    
    Returns:
        accuracy, precision, recall, f1
    """
    # True Positives, False Positives, False Negatives
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    
    acc = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn),
        'tn': int(tn)
    }


def train_epoch(model, X, y, criterion, optimizer, batch_size: int):
    """한 에폭 학습"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    batches = create_batches(X, y, batch_size, shuffle=True)
    
    for X_batch, y_batch in batches:
        # Forward
        logits = model.forward(X_batch)
        loss = criterion.forward(logits, y_batch)
        
        # Backward
        grad = criterion.backward()
        model.backward(grad)
        
        # Update
        optimizer.step(model.get_layers())
        optimizer.zero_grad(model.get_layers())
        
        total_loss += loss * len(y_batch)
        preds = np.argmax(logits, axis=1)
        all_preds.extend(preds)
        all_labels.extend(y_batch)
    
    avg_loss = total_loss / len(y)
    metrics = compute_metrics(np.array(all_labels), np.array(all_preds))
    
    return avg_loss, metrics


def evaluate(model, X, y, criterion, batch_size: int):
    """모델 평가"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    batches = create_batches(X, y, batch_size, shuffle=False)
    
    for X_batch, y_batch in batches:
        logits = model.forward(X_batch)
        loss = criterion.forward(logits, y_batch)
        
        total_loss += loss * len(y_batch)
        preds = np.argmax(logits, axis=1)
        all_preds.extend(preds)
        all_labels.extend(y_batch)
    
    avg_loss = total_loss / len(y)
    metrics = compute_metrics(np.array(all_labels), np.array(all_preds))
    
    return avg_loss, metrics, np.array(all_preds)


def train_model(
    hidden_dims: list = [256, 128],
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    dropout: float = 0.3,
    vectorizer_type: str = "tfidf",
    max_features: int = 3000,
    optimizer_type: str = "adam",
    base_dir: str = ".",
    save_dir: str = "checkpoints",
    seed: int = 42
):
    """모델 학습 메인 함수"""
    
    np.random.seed(seed)
    
    # 데이터 로드
    print("\n" + "="*60)
    print("📂 데이터 준비")
    print("="*60)
    
    dataset = GillamDataset(
        vectorizer_type=vectorizer_type,
        max_features=max_features,
        base_dir=base_dir
    )
    dataset.load_and_preprocess(split_dir=f"{base_dir}/split")
    
    X_train, y_train = dataset.get_train_data()
    X_dev, y_dev = dataset.get_dev_data()
    
    input_dim = X_train.shape[1]
    output_dim = 2
    
    # 모델 생성
    print("\n" + "="*60)
    print("🤖 모델 설정")
    print("="*60)
    print(f"   Input dim: {input_dim}")
    print(f"   Hidden dims: {hidden_dims}")
    print(f"   Output dim: {output_dim}")
    print(f"   Dropout: {dropout}")
    
    model = MLP(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=output_dim,
        dropout=dropout
    )
    
    # 손실 함수 (클래스 가중치 적용)
    class_weights = dataset.get_class_weights()
    criterion = CrossEntropyLoss(class_weights=class_weights)
    print(f"   Class weights: {class_weights}")
    
    # 옵티마이저
    if optimizer_type == "adam":
        optimizer = Adam(lr=learning_rate)
    else:
        optimizer = SGD(params=[], lr=learning_rate, momentum=0.9)
    print(f"   Optimizer: {optimizer_type.upper()}")
    print(f"   Learning rate: {learning_rate}")
    
    # 체크포인트 디렉토리
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    
    # 학습
    print("\n" + "="*60)
    print(f"🚀 학습 시작 (epochs={epochs}, batch_size={batch_size})")
    print("="*60)
    
    best_f1 = 0
    best_epoch = 0
    history = {
        'train_loss': [], 'train_acc': [], 'train_f1': [],
        'dev_loss': [], 'dev_acc': [], 'dev_f1': []
    }
    
    best_model_params = None
    
    for epoch in range(epochs):
        # Train
        train_loss, train_metrics = train_epoch(
            model, X_train, y_train, criterion, optimizer, batch_size
        )
        
        # Evaluate
        dev_loss, dev_metrics, _ = evaluate(
            model, X_dev, y_dev, criterion, batch_size
        )
        
        # History 저장
        history['train_loss'].append(float(train_loss))
        history['train_acc'].append(train_metrics['accuracy'])
        history['train_f1'].append(train_metrics['f1'])
        history['dev_loss'].append(float(dev_loss))
        history['dev_acc'].append(dev_metrics['accuracy'])
        history['dev_f1'].append(dev_metrics['f1'])
        
        # 로그 출력
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"\nEpoch {epoch+1}/{epochs}")
            print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_metrics['accuracy']:.4f}, F1: {train_metrics['f1']:.4f}")
            print(f"  Dev   - Loss: {dev_loss:.4f}, Acc: {dev_metrics['accuracy']:.4f}, F1: {dev_metrics['f1']:.4f}")
        
        # Best 모델 저장
        if dev_metrics['f1'] > best_f1:
            best_f1 = dev_metrics['f1']
            best_epoch = epoch + 1
            
            # 현재 파라미터 복사
            best_model_params = {}
            for i, layer in enumerate(model.get_layers()):
                for name, param in layer.params.items():
                    best_model_params[f"layer_{i}_{name}"] = param.copy()
            
            print(f"  💾 Best 모델 저장 (F1: {best_f1:.4f})")
    
    # Best 모델 저장
    np.savez(save_path / "best_model.npz", **best_model_params)
    
    # 설정 저장
    config = {
        'input_dim': input_dim,
        'hidden_dims': hidden_dims,
        'output_dim': output_dim,
        'dropout': dropout,
        'vectorizer_type': vectorizer_type,
        'max_features': max_features,
        'best_epoch': best_epoch,
        'best_dev_f1': best_f1
    }
    
    with open(save_path / "config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    # 히스토리 저장
    with open(save_path / "history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n" + "="*60)
    print(f"🏆 학습 완료!")
    print("="*60)
    print(f"   Best Epoch: {best_epoch}")
    print(f"   Best Dev F1: {best_f1:.4f}")
    print(f"   모델 저장: {save_path}/best_model.npz")
    
    return {
        'best_epoch': best_epoch,
        'best_dev_f1': best_f1,
        'history': history,
        'config': config
    }


def main():
    parser = argparse.ArgumentParser(description="Gillam SLI/TD 분류 모델 학습")
    
    parser.add_argument("--hidden_dims", type=str, default="256,128",
                        help="은닉층 차원 (쉼표로 구분)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="학습 에폭 수")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="배치 크기")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                        help="학습률")
    parser.add_argument("--dropout", type=float, default=0.3,
                        help="드롭아웃 비율")
    parser.add_argument("--vectorizer", type=str, default="tfidf",
                        choices=["tfidf", "bow"],
                        help="벡터화 방법")
    parser.add_argument("--max_features", type=int, default=3000,
                        help="최대 특성 수")
    parser.add_argument("--optimizer", type=str, default="adam",
                        choices=["adam", "sgd"],
                        help="옵티마이저")
    parser.add_argument("--base_dir", type=str, default=".",
                        help="기본 디렉토리")
    parser.add_argument("--save_dir", type=str, default="checkpoints",
                        help="체크포인트 저장 디렉토리")
    parser.add_argument("--seed", type=int, default=42,
                        help="랜덤 시드")
    
    args = parser.parse_args()
    
    # hidden_dims 파싱
    hidden_dims = [int(x) for x in args.hidden_dims.split(",")]
    
    train_model(
        hidden_dims=hidden_dims,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        dropout=args.dropout,
        vectorizer_type=args.vectorizer,
        max_features=args.max_features,
        optimizer_type=args.optimizer,
        base_dir=args.base_dir,
        save_dir=args.save_dir,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
