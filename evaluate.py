"""
Test 세트 최종 평가 스크립트
- 튜닝된 Best 모델 로드
- Test 세트에서 최종 성능 평가
- 결과 리포트 생성
"""

import numpy as np
import json
import argparse
from pathlib import Path

from numpy_nn import MLP, CrossEntropyLoss
from data_preprocessing import GillamDataset, create_batches
from train import compute_metrics


def load_model(model_path: str, config_path: str):
    """저장된 모델 로드"""
    
    # 설정 로드
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # 모델 생성
    model = MLP(
        input_dim=config.get('input_dim', config.get('max_features', 3000)),
        hidden_dims=config['hidden_dims'],
        output_dim=config.get('output_dim', 2),
        dropout=config['dropout']
    )
    
    # 파라미터 로드
    state = np.load(model_path)
    for i, layer in enumerate(model.get_layers()):
        for name in layer.params.keys():
            key = f"layer_{i}_{name}"
            if key in state:
                layer.params[name] = state[key]
    
    return model, config


def evaluate_test(
    model_path: str = "tuning_results/best_tuned_model.npz",
    config_path: str = "tuning_results/best_config.json",
    base_dir: str = ".",
    save_dir: str = "results"
):
    """Test 세트 최종 평가"""
    
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    
    print("="*60)
    print("🧪 Test 세트 최종 평가")
    print("="*60)
    
    # 설정 로드
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"\n📋 모델 설정:")
    print(f"   Hidden dims: {config['hidden_dims']}")
    print(f"   Dropout: {config['dropout']}")
    print(f"   Learning rate: {config.get('learning_rate', 'N/A')}")
    print(f"   Max features: {config['max_features']}")
    print(f"   Best Dev F1: {config.get('best_dev_f1', 'N/A')}")
    
    # 데이터 로드
    print("\n📂 데이터 로드 중...")
    dataset = GillamDataset(
        vectorizer_type="tfidf",
        max_features=config['max_features'],
        base_dir=base_dir
    )
    dataset.load_and_preprocess(split_dir=f"{base_dir}/split")
    
    X_train, y_train = dataset.get_train_data()
    X_dev, y_dev = dataset.get_dev_data()
    X_test, y_test = dataset.get_test_data()
    
    # 모델 로드
    print("\n🤖 모델 로드 중...")
    model, _ = load_model(model_path, config_path)
    model.eval()
    
    # 평가
    print("\n📊 평가 중...")
    
    results = {}
    
    for split_name, X, y in [('train', X_train, y_train), 
                              ('dev', X_dev, y_dev), 
                              ('test', X_test, y_test)]:
        all_preds = []
        all_labels = []
        
        batches = create_batches(X, y, batch_size=32, shuffle=False)
        
        for X_batch, y_batch in batches:
            logits = model.forward(X_batch)
            preds = np.argmax(logits, axis=1)
            all_preds.extend(preds)
            all_labels.extend(y_batch)
        
        metrics = compute_metrics(np.array(all_labels), np.array(all_preds))
        results[split_name] = metrics
    
    # 결과 출력
    print("\n" + "="*60)
    print("📊 최종 결과")
    print("="*60)
    
    for split_name in ['train', 'dev', 'test']:
        m = results[split_name]
        print(f"\n{split_name.upper()} 세트:")
        print(f"   Accuracy:  {m['accuracy']:.4f}")
        print(f"   Precision: {m['precision']:.4f}")
        print(f"   Recall:    {m['recall']:.4f}")
        print(f"   F1 Score:  {m['f1']:.4f}")
    
    # Confusion Matrix (Test)
    test_m = results['test']
    print(f"\n📊 Test Confusion Matrix:")
    print(f"                 Predicted")
    print(f"                 TD    SLI")
    print(f"   Actual TD   [{test_m['tn']:3d}   {test_m['fp']:3d}]")
    print(f"   Actual SLI  [{test_m['fn']:3d}   {test_m['tp']:3d}]")
    
    # Classification Report
    print(f"\n📋 Classification Report (Test):")
    print(f"{'':15} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}")
    print(f"{'-'*45}")
    
    # TD (class 0)
    td_precision = test_m['tn'] / (test_m['tn'] + test_m['fn']) if (test_m['tn'] + test_m['fn']) > 0 else 0
    td_recall = test_m['tn'] / (test_m['tn'] + test_m['fp']) if (test_m['tn'] + test_m['fp']) > 0 else 0
    td_f1 = 2 * td_precision * td_recall / (td_precision + td_recall) if (td_precision + td_recall) > 0 else 0
    print(f"{'TD':15} {td_precision:>10.4f} {td_recall:>10.4f} {td_f1:>10.4f}")
    
    # SLI (class 1)
    print(f"{'SLI':15} {test_m['precision']:>10.4f} {test_m['recall']:>10.4f} {test_m['f1']:>10.4f}")
    print(f"{'-'*45}")
    print(f"{'Accuracy':15} {test_m['accuracy']:>10.4f}")
    print(f"{'Macro Avg':15} {(td_precision + test_m['precision'])/2:>10.4f} {(td_recall + test_m['recall'])/2:>10.4f} {(td_f1 + test_m['f1'])/2:>10.4f}")
    
    # 결과 저장
    final_results = {
        'model_config': config,
        'results': {
            split: {k: float(v) if isinstance(v, (np.floating, float)) else int(v) 
                    for k, v in metrics.items()}
            for split, metrics in results.items()
        },
        'test_summary': {
            'accuracy': results['test']['accuracy'],
            'precision': results['test']['precision'],
            'recall': results['test']['recall'],
            'f1': results['test']['f1']
        }
    }
    
    with open(save_path / "final_results.json", 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\n💾 결과 저장: {save_path / 'final_results.json'}")
    
    print("\n" + "="*60)
    print("✅ 평가 완료!")
    print("="*60)
    
    return final_results


def main():
    parser = argparse.ArgumentParser(description="Test 세트 최종 평가")
    
    parser.add_argument("--model_path", type=str, 
                        default="tuning_results/best_tuned_model.npz",
                        help="모델 파일 경로")
    parser.add_argument("--config_path", type=str,
                        default="tuning_results/best_config.json",
                        help="설정 파일 경로")
    parser.add_argument("--base_dir", type=str, default=".",
                        help="기본 디렉토리")
    parser.add_argument("--save_dir", type=str, default="results",
                        help="결과 저장 디렉토리")
    
    args = parser.parse_args()
    
    evaluate_test(
        model_path=args.model_path,
        config_path=args.config_path,
        base_dir=args.base_dir,
        save_dir=args.save_dir
    )


if __name__ == "__main__":
    main()

