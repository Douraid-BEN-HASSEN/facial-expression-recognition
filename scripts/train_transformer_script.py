"""
🎯 SCRIPT D'ENTRAÎNEMENT OPTIMISÉ POUR TRANSFORMER
===================================================

Ce script implémente les meilleures pratiques pour entraîner un Transformer
sur des données de landmarks faciaux :

✅ Warmup du Learning Rate (crucial pour Transformers)
✅ AdamW avec weight decay
✅ Gradient Clipping
✅ Label Smoothing
✅ Cosine Annealing Schedule
✅ Augmentation de données adaptée aux landmarks

COMMANDE D'UTILISATION :
-----------------------
python scripts/train_transformer_script.py \
    --dataset_path ./datasets/fer2013_features \
    --model_name emotion_transformer \
    --model_size standard \
    --batch_size 128 \
    --epochs 200 \
    --learning_rate 1e-4 \
    --warmup_steps 1000 \
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
from classes.EmotionTransformerModel import EmotionTransformer, EmotionTransformerLarge, ModelManager


def train_transformer_emotion_classifier(
    dataset_path: str,
    model_name: str,
    model_size: str = 'standard',
    batch_size: int = 128,
    epochs: int = 200,
    learning_rate: float = 1e-4,
    warmup_steps: int = 1000,
    patience: int = 30,
    save_dir: str = "./models"
):
    """
    Fonction principale pour entraîner le Transformer
    """
    print("\n" + "="*80)
    print("🎯 ENTRAÎNEMENT TRANSFORMER POUR CLASSIFICATION D'ÉMOTIONS")
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
        model = EmotionTransformer(
            input_dim=metadata['feature_dimension'],
            n_classes=metadata['n_classes'],
            n_landmarks=n_landmarks,
            d_model=128,
            n_heads=8,
            n_layers=4,
            d_ff=512,
            dropout=0.15
        )
        print(f"\n🧠 Modèle STANDARD créé")
    elif model_size == 'large':
        model = EmotionTransformerLarge(
            input_dim=metadata['feature_dimension'],
            n_classes=metadata['n_classes'],
            n_landmarks=n_landmarks,
            d_model=256,
            n_heads=16,
            n_layers=6,
            d_ff=1024,
            dropout=0.15
        )
        print(f"\n🧠 Modèle LARGE créé")
    else:
        raise ValueError(f"model_size doit être 'standard' ou 'large', pas '{model_size}'")
    
    print(f"   Paramètres : {model.get_num_parameters():,}")
    
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
        warmup_steps=warmup_steps,
        label_smoothing=0.1,
        gradient_clip=1.0,
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
    parser = argparse.ArgumentParser(
        description="Entraîner un Transformer pour la classification d'émotions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="""
Exemples d'utilisation :
------------------------
python scripts/train_transformer_script.py --dataset_path ./datasets/fer2013_features --model_name emotion_transformer --model_size standard --batch_size 128 --epochs 100 --learning_rate 2e-4 --warmup_steps 500 --patience 40
""")
    
    parser.add_argument("--dataset_path", type=str, default="./datasets/fer2013_features",
                       help="Chemin vers le dataset")
    parser.add_argument("--model_name", type=str, default="emotion_transformer",
                       help="Nom du modèle à sauvegarder")
    parser.add_argument("--model_size", type=str, default="standard", choices=['standard', 'large'],
                       help="Taille du modèle (standard=1M params, large=3-5M params)")
    parser.add_argument("--batch_size", type=int, default=128,
                       help="Taille des batchs - RÉDUIT pour meilleure généralisation")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Nombre maximum d'epochs - AUGMENTÉ car modèle pas convergé")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                       help="Learning rate initial - AUGMENTÉ pour convergence plus rapide")
    parser.add_argument("--warmup_steps", type=int, default=500,
                       help="Nombre de steps de warmup - RÉDUIT (trop long avant)")
    parser.add_argument("--patience", type=int, default=40,
                       help="Patience pour early stopping - AUGMENTÉ pour laisser converger")
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("⚙️  CONFIGURATION")
    print("="*80)
    for arg, value in vars(args).items():
        print(f"{arg:20s} : {value}")
    print("="*80)
    
    model, history = train_transformer_emotion_classifier(
        dataset_path=args.dataset_path,
        model_name=args.model_name,
        model_size=args.model_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        patience=args.patience,
        save_dir=f"./models/"
    )
