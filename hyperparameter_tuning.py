"""
하이퍼파라미터 튜닝 스크립트
Grid Search를 통한 최적 하이퍼파라미터 탐색
"""

import numpy as np
import json
import itertools
from datetime import datetime
from pathlib import Path

from numpy_nn import MLP, CrossEntropyLoss, Adam, SGD
from data_preprocessing import GillamDataset, create_batches
from train import train_epoch, evaluate, compute_metrics


def grid_search(
    base_dir: str = ".",
    save_dir: str = "tuning_results",
    seed: int = 42
):
    """
    Grid Search를 통한 하이퍼파라미터 튜닝
    """
    np.random.seed(seed)
    
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    
    # 탐색할 하이퍼파라미터 공간
    param_grid = {
        'hidden_dims': [[256, 128], [512, 256], [256, 128, 64], [128, 64]],
        'learning_rate': [0.0005, 0.001, 0.002],
        'dropout': [0.2, 0.3, 0.5],
        'batch_size': [16, 32],
        'max_features': [2000, 3000, 4000]
    }
    
    # 모든 조합 생성
    keys = param_grid.keys()
    combinations = list(itertools.product(*param_grid.values()))
    
    print("="*60)
    print("🔍 하이퍼파라미터 튜닝 시작")
    print("="*60)
    print(f"   총 조합 수: {len(combinations)}")
    print(f"   탐색 공간:")
    for key, values in param_grid.items():
        print(f"      {key}: {values}")
    
    results = []
    best_f1 = 0
    best_config = None
    best_model_params = None
    
    # 데이터는 한 번만 로드 (max_features는 나중에 조정)
    print("\n📂 기본 데이터 로드 중...")
    
    for i, values in enumerate(combinations):
        config = dict(zip(keys, values))
        
        print(f"\n{'='*60}")
        print(f"🧪 실험 {i+1}/{len(combinations)}")
        print(f"   Config: {config}")
        print(f"{'='*60}")
        
        try:
            # 데이터 로드 (max_features 적용)
            dataset = GillamDataset(
                vectorizer_type="tfidf",
                max_features=config['max_features'],
                base_dir=base_dir
            )
            dataset.load_and_preprocess(split_dir=f"{base_dir}/split")
            
            X_train, y_train = dataset.get_train_data()
            X_dev, y_dev = dataset.get_dev_data()
            
            input_dim = X_train.shape[1]
            
            # 모델 생성
            model = MLP(
                input_dim=input_dim,
                hidden_dims=config['hidden_dims'],
                output_dim=2,
                dropout=config['dropout']
            )
            
            # 손실 함수 및 옵티마이저
            class_weights = dataset.get_class_weights()
            criterion = CrossEntropyLoss(class_weights=class_weights)
            optimizer = Adam(lr=config['learning_rate'])
            
            # 학습 (고정 에폭)
            epochs = 30
            exp_best_f1 = 0
            exp_best_epoch = 0
            exp_best_params = None
            
            for epoch in range(epochs):
                train_loss, train_metrics = train_epoch(
                    model, X_train, y_train, criterion, optimizer, config['batch_size']
                )
                dev_loss, dev_metrics, _ = evaluate(
                    model, X_dev, y_dev, criterion, config['batch_size']
                )
                
                if dev_metrics['f1'] > exp_best_f1:
                    exp_best_f1 = dev_metrics['f1']
                    exp_best_epoch = epoch + 1
                    
                    # 파라미터 복사
                    exp_best_params = {}
                    for j, layer in enumerate(model.get_layers()):
                        for name, param in layer.params.items():
                            exp_best_params[f"layer_{j}_{name}"] = param.copy()
            
            print(f"   Best Epoch: {exp_best_epoch}, Best Dev F1: {exp_best_f1:.4f}")
            
            # 결과 기록
            experiment = {
                'experiment_id': i + 1,
                'config': config,
                'best_epoch': exp_best_epoch,
                'best_dev_f1': exp_best_f1
            }
            results.append(experiment)
            
            # 전체 Best 업데이트
            if exp_best_f1 > best_f1:
                best_f1 = exp_best_f1
                best_config = config.copy()
                best_config['best_epoch'] = exp_best_epoch
                best_model_params = exp_best_params
                print(f"   🏆 새로운 Best! F1: {best_f1:.4f}")
        
        except Exception as e:
            print(f"   ❌ 실험 실패: {e}")
            continue
    
    # Best 모델 저장
    if best_model_params is not None:
        np.savez(save_path / "best_tuned_model.npz", **best_model_params)
        
        # Best 설정 저장
        best_config_full = {
            'input_dim': best_config['max_features'],  # 근사값
            'hidden_dims': best_config['hidden_dims'],
            'output_dim': 2,
            'dropout': best_config['dropout'],
            'learning_rate': best_config['learning_rate'],
            'batch_size': best_config['batch_size'],
            'max_features': best_config['max_features'],
            'best_epoch': best_config['best_epoch'],
            'best_dev_f1': best_f1
        }
        
        with open(save_path / "best_config.json", 'w') as f:
            json.dump(best_config_full, f, indent=2)
    
    # 전체 결과 저장
    tuning_results = {
        'param_grid': {k: [str(v) if isinstance(v, list) else v for v in vals] 
                       for k, vals in param_grid.items()},
        'total_experiments': len(combinations),
        'successful_experiments': len(results),
        'best_config': best_config,
        'best_dev_f1': best_f1,
        'all_results': sorted(results, key=lambda x: x['best_dev_f1'], reverse=True),
        'timestamp': datetime.now().isoformat()
    }
    
    with open(save_path / "tuning_results.json", 'w') as f:
        json.dump(tuning_results, f, indent=2, default=str)
    
    # 결과 요약
    print("\n" + "="*60)
    print("📊 하이퍼파라미터 튜닝 완료!")
    print("="*60)
    print(f"   성공한 실험: {len(results)}/{len(combinations)}")
    print(f"   Best Dev F1: {best_f1:.4f}")
    print(f"   Best Config: {best_config}")
    
    # Top 5 결과
    print(f"\n📋 Top 5 결과:")
    for rank, exp in enumerate(sorted(results, key=lambda x: x['best_dev_f1'], reverse=True)[:5], 1):
        print(f"   {rank}. F1: {exp['best_dev_f1']:.4f}")
        print(f"      {exp['config']}")
    
    print(f"\n   결과 저장: {save_path}")
    
    return tuning_results


def quick_search(
    base_dir: str = ".",
    save_dir: str = "tuning_results",
    seed: int = 42
):
    """
    빠른 하이퍼파라미터 탐색 (축소된 공간)
    """
    np.random.seed(seed)
    
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    
    # 축소된 탐색 공간
    param_grid = {
        'hidden_dims': [[256, 128], [512, 256]],
        'learning_rate': [0.001, 0.002],
        'dropout': [0.3, 0.5],
        'batch_size': [32],
        'max_features': [3000]
    }
    
    keys = param_grid.keys()
    combinations = list(itertools.product(*param_grid.values()))
    
    print("="*60)
    print("⚡ 빠른 하이퍼파라미터 탐색")
    print("="*60)
    print(f"   총 조합 수: {len(combinations)}")
    
    results = []
    best_f1 = 0
    best_config = None
    best_model_params = None
    
    for i, values in enumerate(combinations):
        config = dict(zip(keys, values))
        
        print(f"\n실험 {i+1}/{len(combinations)}: {config}")
        
        try:
            dataset = GillamDataset(
                vectorizer_type="tfidf",
                max_features=config['max_features'],
                base_dir=base_dir
            )
            dataset.load_and_preprocess(split_dir=f"{base_dir}/split")
            
            X_train, y_train = dataset.get_train_data()
            X_dev, y_dev = dataset.get_dev_data()
            
            model = MLP(
                input_dim=X_train.shape[1],
                hidden_dims=config['hidden_dims'],
                output_dim=2,
                dropout=config['dropout']
            )
            
            criterion = CrossEntropyLoss(class_weights=dataset.get_class_weights())
            optimizer = Adam(lr=config['learning_rate'])
            
            exp_best_f1 = 0
            exp_best_params = None
            
            for epoch in range(30):
                train_epoch(model, X_train, y_train, criterion, optimizer, config['batch_size'])
                _, dev_metrics, _ = evaluate(model, X_dev, y_dev, criterion, config['batch_size'])
                
                if dev_metrics['f1'] > exp_best_f1:
                    exp_best_f1 = dev_metrics['f1']
                    exp_best_params = {}
                    for j, layer in enumerate(model.get_layers()):
                        for name, param in layer.params.items():
                            exp_best_params[f"layer_{j}_{name}"] = param.copy()
            
            print(f"   Dev F1: {exp_best_f1:.4f}")
            
            results.append({
                'config': config,
                'best_dev_f1': exp_best_f1
            })
            
            if exp_best_f1 > best_f1:
                best_f1 = exp_best_f1
                best_config = config
                best_model_params = exp_best_params
                print(f"   🏆 New Best!")
        
        except Exception as e:
            print(f"   ❌ 실패: {e}")
    
    # 저장
    if best_model_params:
        np.savez(save_path / "best_tuned_model.npz", **best_model_params)
        
        with open(save_path / "best_config.json", 'w') as f:
            json.dump({
                'hidden_dims': best_config['hidden_dims'],
                'dropout': best_config['dropout'],
                'learning_rate': best_config['learning_rate'],
                'batch_size': best_config['batch_size'],
                'max_features': best_config['max_features'],
                'best_dev_f1': best_f1
            }, f, indent=2)
    
    print(f"\n🏆 Best F1: {best_f1:.4f}")
    print(f"   Config: {best_config}")
    
    return best_config, best_f1


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="하이퍼파라미터 튜닝")
    parser.add_argument("--mode", type=str, default="quick",
                        choices=["quick", "full"],
                        help="탐색 모드 (quick: 빠른 탐색, full: 전체 탐색)")
    parser.add_argument("--base_dir", type=str, default=".")
    parser.add_argument("--save_dir", type=str, default="tuning_results")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    if args.mode == "quick":
        quick_search(args.base_dir, args.save_dir, args.seed)
    else:
        grid_search(args.base_dir, args.save_dir, args.seed)
