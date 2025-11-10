"""
🎯 SCRIPT D'ENTRAÎNEMENT OPTIMISÉ POUR GNN
==========================================

Ce script implémente les meilleures pratiques pour entraîner un GNN (Graph Neural Network)
sur des données de landmarks faciaux :

✅ Graph Convolution avec connexions anatomiques MediaPipe
✅ Adam optimizer avec weight decay
✅ ReduceLROnPlateau scheduler
✅ Early stopping avec patience
✅ Exploitation de la topologie faciale naturelle

COMMANDE D'UTILISATION :
-----------------------
python scripts/train_gnn_script.py \
    --dataset_path ./datasets/fer2013_eyes_mouth_features \
    --model_name emotion_gnn \
    --model_size standard \
    --batch_size 128 \
    --epochs 150 \
    --learning_rate 1e-3 \
    --patience 30
"""

import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import json
import argparse
from pathlib import Path
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from classes.EmotionGnnModel import EmotionGNN, EmotionGNNMedium, EmotionGNNDeep, ModelManager


def train_gnn_emotion_classifier(
    dataset_path: str,
    model_name: str,
    model_size: str = 'standard',
    batch_size: int = 128,
    epochs: int = 150,
    learning_rate: float = 1e-3,
    patience: int = 30,
    pooling: str = 'mean',
    save_dir: str = "./models"
):
    """
    Fonction principale pour entraîner le GNN
    """
    print("\n" + "="*80)
    print("🎯 ENTRAÎNEMENT GNN POUR CLASSIFICATION D'ÉMOTIONS")
    print("="*80)
    
    # Charger les métadonnées
    metadata_path = Path(dataset_path) / "metadata.json"
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    print(f"\n📊 Dataset :")
    print(f"   Features   : {metadata['feature_dimension']} dimensions")
    print(f"   Classes    : {metadata['n_classes']}")
    print(f"   Landmarks  : {metadata['feature_dimension'] // 3}")
    
    # Charger les données
    X_train = np.load(Path(dataset_path) / "X_train.npy")
    y_train = np.load(Path(dataset_path) / "y_train.npy")
    X_val = np.load(Path(dataset_path) / "X_val.npy")
    y_val = np.load(Path(dataset_path) / "y_val.npy")
    X_test = np.load(Path(dataset_path) / "X_test.npy")
    y_test = np.load(Path(dataset_path) / "y_test.npy")
    
    print(f"\n📁 Données chargées :")
    print(f"   Train : {X_train.shape}")
    print(f"   Val   : {X_val.shape}")
    print(f"   Test  : {X_test.shape}")
    
    # Créer les DataLoaders
    train_dataset = TensorDataset(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(y_train).long()
    )
    val_dataset = TensorDataset(
        torch.from_numpy(X_val).float(),
        torch.from_numpy(y_val).long()
    )
    test_dataset = TensorDataset(
        torch.from_numpy(X_test).float(),
        torch.from_numpy(y_test).long()
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Créer le modèle
    n_landmarks = metadata['feature_dimension'] // 3
    
    if model_size == 'standard':
        model = EmotionGNN(
            input_dim=metadata['feature_dimension'],
            n_classes=metadata['n_classes'],
            n_landmarks=n_landmarks,
            hidden_dims=[128, 256, 256, 128],
            dropout=0.15,
            pooling=pooling
        )
        print(f"\n🧠 Modèle GNN STANDARD créé")
    elif model_size == 'medium':
        model = EmotionGNNMedium(
            input_dim=metadata['feature_dimension'],
            n_classes=metadata['n_classes'],
            n_landmarks=n_landmarks,
            hidden_dims=[128, 256, 384, 256, 128],
            dropout=0.2,
            pooling=pooling
        )
        print(f"\n🧠 Modèle GNN MEDIUM créé (BatchNorm pour stabilité)")
    elif model_size == 'deep':
        model = EmotionGNNDeep(
            input_dim=metadata['feature_dimension'],
            n_classes=metadata['n_classes'],
            n_landmarks=n_landmarks,
            hidden_dims=[128, 256, 512, 512, 256, 128],
            dropout=0.15,
            pooling=pooling
        )
        print(f"\n🧠 Modèle GNN DEEP créé (⚠️ peut ne pas converger)")
    else:
        raise ValueError(f"model_size doit être 'standard', 'medium' ou 'deep', pas '{model_size}'")
    
    print(f"   Paramètres : {model.get_num_parameters():,}")
    print(f"   Pooling    : {pooling}")
    print(f"   Arêtes     : {model.edge_index.shape[1]}")
    
    # Créer le dossier de sauvegarde
    save_path = Path(save_dir) / model_name
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Entraîner avec ModelManager
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    history = ModelManager.train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        n_epochs=epochs,
        device=device,
        learning_rate=learning_rate,
        weight_decay=1e-4,
        patience=patience,
        save_dir=str(save_path),
        model_name=model_name
    )
    
    # Sauvegarder l'historique
    history_path = save_path / f"{model_name}_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    # Tracer les courbes
    plot_path = save_path / f"{model_name}_curves.png"
    ModelManager.plot_history(history, str(plot_path))
    
    print(f"\n✅ Fichiers sauvegardés dans : {save_path}")
    
    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraîner un GNN pour la classification d'émotions")
    parser.add_argument("--dataset_path", type=str, default="./datasets/fer2013_features",
                       help="Chemin vers le dataset")
    parser.add_argument("--model_name", type=str, default="emotion_gnn_deep",
                       help="Nom du modèle à sauvegarder")
    parser.add_argument("--model_size", type=str, default="medium", choices=['standard', 'medium', 'deep'],
                       help="Taille du modèle (standard=400-500K, medium=600K [RECOMMANDÉ], deep=800K-1M)")
    parser.add_argument("--batch_size", type=int, default=128,
                       help="Taille des batchs")
    parser.add_argument("--epochs", type=int, default=150,
                       help="Nombre maximum d'epochs")
    parser.add_argument("--learning_rate", type=float, default=0.003,#0.001
                       help="Learning rate initial (plus élevé que Transformer)")
    parser.add_argument("--patience", type=int, default=20,
                       help="Patience pour early stopping")
    parser.add_argument("--pooling", type=str, default="mean", choices=['mean', 'max', 'attention'],
                       help="Type de pooling global (mean=simple, attention=meilleur mais plus lent)")
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("⚙️  CONFIGURATION")
    print("="*80)
    for arg, value in vars(args).items():
        print(f"{arg:20s} : {value}")
    print("="*80)
    
    print("\n💡 POURQUOI GNN POUR LES LANDMARKS :")
    print("   ✅ Structure naturelle : Graphe facial anatomique (MediaPipe)")
    print("   ✅ Moins de paramètres : ~400-800K vs 1M+ (Transformer)")
    print("   ✅ Connexions réelles  : Yeux, sourcils, bouche, iris")
    print("   ✅ Performances attendues : 62-68% (vs 57.6% Transformer)")
    print("="*80)
    
    model, history = train_gnn_emotion_classifier(
        dataset_path=args.dataset_path,
        model_name=args.model_name,
        model_size=args.model_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        patience=args.patience,
        pooling=args.pooling,
        save_dir=f"./models/"
    )
