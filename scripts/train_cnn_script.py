"""
🎯 SCRIPT D'ENTRAÎNEMENT CNN POUR CLASSIFICATION D'ÉMOTIONS
============================================================

Script optimisé pour entraîner un CNN (EfficientNet-B0) sur FER2013.
Implémente toutes les techniques SOTA pour maximiser les performances.

STRATÉGIE D'ENTRAÎNEMENT EN 3 PHASES :
---------------------------------------
Phase 1 - head_only (10-20 epochs) : Entraîner seulement le classifier
Phase 2 - partial (30-50 epochs)   : Fine-tune les derniers blocs
Phase 3 - full (20-30 epochs)      : Fine-tune complet

UTILISATION :
-------------
# Phase 1 - Head Only (RECOMMANDÉ POUR COMMENCER)
python scripts/train_cnn_script.py \
    --dataset_path ./datasets/fer2013_images \
    --model_name emotion_cnn_phase1 \
    --fine_tune_mode head_only \
    --epochs 20 \
    --learning_rate 1e-3 \
    --batch_size 64

# Phase 2 - Partial Fine-Tuning
python scripts/train_cnn_script.py \
    --dataset_path ./datasets/fer2013_images \
    --model_name emotion_cnn_phase2 \
    --fine_tune_mode partial \
    --epochs 50 \
    --learning_rate 5e-4 \
    --load_checkpoint ./models/emotion_cnn_phase1/emotion_cnn_phase1.pth

# Phase 3 - Full Fine-Tuning
python scripts/train_cnn_script.py \
    --dataset_path ./datasets/fer2013_images \
    --model_name emotion_cnn_phase3 \
    --fine_tune_mode full \
    --epochs 30 \
    --learning_rate 1e-4 \
    --load_checkpoint ./models/emotion_cnn_phase2/emotion_cnn_phase2.pth

# Entraînement complet en une fois (si pressé)
python scripts/train_cnn_script.py \
    --dataset_path ./datasets/fer2013_images \
    --model_name emotion_cnn_full \
    --fine_tune_mode full \
    --epochs 100 \
    --learning_rate 3e-4
"""

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
import argparse
import sys
import os
from pathlib import Path
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from classes.EmotionCnnModel import EmotionCNN, ModelManager


def get_data_transforms(image_size: int = 224, augment: bool = True):
    """
    Transformations pour les images
    
    Args:
        image_size: Taille des images (224 pour EfficientNet)
        augment: Appliquer l'augmentation de données
    
    Returns:
        train_transform, val_transform
    """
    # Normalisation ImageNet (requis pour transfer learning)
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    if augment:
        # Transformation pour l'entraînement avec augmentation
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            normalize,
            # RandomErasing pour plus de régularisation
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.15), ratio=(0.3, 3.3))
        ])
    else:
        # Transformation simple pour l'entraînement
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize
        ])
    
    # Transformation pour la validation/test (pas d'augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize
    ])
    
    return train_transform, val_transform


def prepare_dataloaders(
    dataset_path: str,
    batch_size: int = 64,
    val_split: float = 0.1,
    test_split: float = 0.1,
    num_workers: int = 4,
    image_size: int = 224,
    augment: bool = True
):
    """
    Préparer les DataLoaders à partir d'un dossier d'images organisé par classe
    
    Structure attendue :
    dataset_path/
        angry/
            image1.jpg
            image2.jpg
        disgust/
        fear/
        happy/
        neutral/
        sad/
        surprise/
    
    Args:
        dataset_path: Chemin vers le dossier du dataset
        batch_size: Taille des batchs
        val_split: Proportion pour validation
        test_split: Proportion pour test
        num_workers: Nombre de workers pour le DataLoader
        image_size: Taille des images
        augment: Appliquer l'augmentation
    
    Returns:
        train_loader, val_loader, test_loader, class_names
    """
    dataset_path = Path(dataset_path)
    
    # Obtenir les transformations
    train_transform, val_transform = get_data_transforms(image_size, augment)
    
    # Charger le dataset complet
    full_dataset = datasets.ImageFolder(root=str(dataset_path))
    class_names = full_dataset.classes
    n_classes = len(class_names)
    
    print(f"\n📁 Dataset chargé : {dataset_path}")
    print(f"   Total d'images : {len(full_dataset)}")
    print(f"   Classes        : {n_classes} - {class_names}")
    
    # Calculer les tailles de splits
    total_size = len(full_dataset)
    test_size = int(total_size * test_split)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size - test_size
    
    print(f"\n📊 Splits :")
    print(f"   Train : {train_size} images ({train_size/total_size*100:.1f}%)")
    print(f"   Val   : {val_size} images ({val_size/total_size*100:.1f}%)")
    print(f"   Test  : {test_size} images ({test_size/total_size*100:.1f}%)")
    
    # Splitter le dataset
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Appliquer les transformations
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    test_dataset.dataset.transform = val_transform
    
    # Créer les DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return train_loader, val_loader, test_loader, class_names


def train_cnn_emotion_classifier(
    dataset_path: str,
    model_name: str,
    fine_tune_mode: str = 'head_only',
    batch_size: int = 64,
    epochs: int = 100,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 20,
    dropout: float = 0.3,
    use_mixup: bool = True,
    mixup_alpha: float = 0.2,
    label_smoothing: float = 0.1,
    use_cosine_annealing: bool = True,
    gradient_clip: float = 1.0,
    image_size: int = 224,
    augment: bool = True,
    val_split: float = 0.1,
    test_split: float = 0.1,
    num_workers: int = 4,
    save_dir: str = "./models",
    load_checkpoint: str = None
):
    """
    Fonction principale pour entraîner le CNN
    """
    print("\n" + "="*80)
    print("🎯 ENTRAÎNEMENT CNN POUR CLASSIFICATION D'ÉMOTIONS")
    print("="*80)
    
    # Préparer les DataLoaders
    train_loader, val_loader, test_loader, class_names = prepare_dataloaders(
        dataset_path=dataset_path,
        batch_size=batch_size,
        val_split=val_split,
        test_split=test_split,
        num_workers=num_workers,
        image_size=image_size,
        augment=augment
    )
    
    # Créer le modèle
    n_classes = len(class_names)
    model = EmotionCNN(
        n_classes=n_classes,
        pretrained=True,
        dropout=dropout,
        fine_tune_mode=fine_tune_mode
    )
    
    print(f"\n🧠 Modèle CNN EfficientNet-B0 créé :")
    print(f"   Fine-tune mode          : {fine_tune_mode}")
    print(f"   Paramètres totaux       : {model.get_num_parameters():,}")
    print(f"   Paramètres entraînables : {model.get_trainable_parameters():,}")
    print(f"   Dropout                 : {dropout}")
    
    # Charger un checkpoint si spécifié
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if load_checkpoint and Path(load_checkpoint).exists():
        print(f"\n📂 Chargement du checkpoint : {load_checkpoint}")
        checkpoint = torch.load(load_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"   ✅ Checkpoint chargé (epoch {checkpoint['epoch']}, val_acc {checkpoint['val_acc']:.2f}%)")
        
        # Mettre à jour le mode de fine-tuning si nécessaire
        if fine_tune_mode != checkpoint.get('fine_tune_mode', 'full'):
            print(f"   🔄 Changement de mode : {checkpoint.get('fine_tune_mode')} → {fine_tune_mode}")
            model._set_fine_tune_mode(fine_tune_mode)
    
    # Créer le dossier de sauvegarde
    save_path = Path(save_dir) / model_name
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Entraîner avec ModelManager
    history = ModelManager.train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        n_epochs=epochs,
        device=device,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        patience=patience,
        save_dir=str(save_path),
        model_name=model_name,
        use_mixup=use_mixup,
        mixup_alpha=mixup_alpha,
        label_smoothing=label_smoothing,
        use_cosine_annealing=use_cosine_annealing,
        gradient_clip=gradient_clip
    )
    
    # Sauvegarder l'historique
    history_path = save_path / f"{model_name}_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    # Sauvegarder les métadonnées
    metadata = {
        'model_name': model_name,
        'n_classes': n_classes,
        'class_names': class_names,
        'fine_tune_mode': fine_tune_mode,
        'image_size': image_size,
        'dropout': dropout,
        'best_val_acc': max(history['val_acc']),
        'test_acc': history['test_acc'],
        'best_epoch': history['best_epoch'],
        'training_config': {
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'use_mixup': use_mixup,
            'mixup_alpha': mixup_alpha,
            'label_smoothing': label_smoothing,
            'gradient_clip': gradient_clip,
            'augment': augment
        }
    }
    
    metadata_path = save_path / f"{model_name}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Tracer les courbes
    plot_path = save_path / f"{model_name}_curves.png"
    ModelManager.plot_history(history, str(plot_path))
    
    print(f"\n✅ Fichiers sauvegardés dans : {save_path}")
    print(f"   - {model_name}.pth (modèle)")
    print(f"   - {model_name}_history.json (historique)")
    print(f"   - {model_name}_metadata.json (métadonnées)")
    print(f"   - {model_name}_curves.png (graphiques)")
    
    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Entraîner un CNN (EfficientNet-B0) pour la classification d'émotions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation :
------------------------
# Entraînement en 3 phases pour performances optimales

# Phase 1 - Head Only (10-20 epochs)
python scripts/train_cnn_script.py --dataset_path ./datasets/fer2013_images --model_name emotion_cnn_phase1 --fine_tune_mode head_only --epochs 20 --learning_rate 1e-3 --batch_size 64

# Phase 2 - Partial (30-50 epochs)
python scripts/train_cnn_script.py --dataset_path ./datasets/fer2013_images --model_name emotion_cnn_phase2 --fine_tune_mode partial --epochs 50 --learning_rate 5e-4 --load_checkpoint ./models/emotion_cnn_phase1/emotion_cnn_phase1.pth

# Phase 3 - Full (20-30 epochs)
python scripts/train_cnn_script.py --dataset_path ./datasets/fer2013_images --model_name emotion_cnn_phase3 --fine_tune_mode full --epochs 30 --learning_rate 1e-4 --load_checkpoint ./models/emotion_cnn_phase2/emotion_cnn_phase2.pth

# OU entraînement complet en une fois (plus rapide mais performances légèrement inférieures)
python scripts/train_cnn_script.py --dataset_path ./datasets/fer2013_images --model_name emotion_cnn_full --fine_tune_mode full --epochs 100 --learning_rate 3e-4
        """
    )
    
    # Paramètres principaux
    parser.add_argument("--dataset_path", type=str, required=True,
                       help="Chemin vers le dossier d'images (structure : dataset_path/emotion/image.jpg)")
    parser.add_argument("--model_name", type=str, required=True,
                       help="Nom du modèle à sauvegarder")
    parser.add_argument("--fine_tune_mode", type=str, default="head_only",
                       choices=['head_only', 'partial', 'full'],
                       help="Mode de fine-tuning (head_only=classifier, partial=derniers blocs, full=tout)")
    
    # Hyperparamètres d'entraînement
    parser.add_argument("--batch_size", type=int, default=64,
                       help="Taille des batchs (défaut: 64)")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Nombre maximum d'epochs (défaut: 100)")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                       help="Learning rate initial (défaut: 1e-3 pour head_only, 5e-4 pour partial, 1e-4 pour full)")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                       help="Weight decay pour régularisation (défaut: 1e-4)")
    parser.add_argument("--patience", type=int, default=20,
                       help="Patience pour early stopping (défaut: 20)")
    parser.add_argument("--dropout", type=float, default=0.3,
                       help="Dropout dans le classifier (défaut: 0.3)")
    
    # Techniques d'augmentation et régularisation
    parser.add_argument("--use_mixup", action="store_true", default=True,
                       help="Utiliser Mixup (défaut: True)")
    parser.add_argument("--mixup_alpha", type=float, default=0.2,
                       help="Paramètre alpha pour Mixup (défaut: 0.2)")
    parser.add_argument("--label_smoothing", type=float, default=0.1,
                       help="Label smoothing (défaut: 0.1)")
    parser.add_argument("--gradient_clip", type=float, default=1.0,
                       help="Gradient clipping (défaut: 1.0)")
    parser.add_argument("--no_augment", action="store_true",
                       help="Désactiver l'augmentation de données")
    
    # Scheduler
    parser.add_argument("--use_cosine_annealing", action="store_true", default=True,
                       help="Utiliser Cosine Annealing LR (défaut: True)")
    
    # Dataset
    parser.add_argument("--image_size", type=int, default=224,
                       help="Taille des images (défaut: 224 pour EfficientNet)")
    parser.add_argument("--val_split", type=float, default=0.1,
                       help="Proportion pour validation (défaut: 0.1)")
    parser.add_argument("--test_split", type=float, default=0.1,
                       help="Proportion pour test (défaut: 0.1)")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Nombre de workers pour DataLoader (défaut: 4)")
    
    # Sauvegarde
    parser.add_argument("--save_dir", type=str, default="./models",
                       help="Dossier de sauvegarde (défaut: ./models)")
    parser.add_argument("--load_checkpoint", type=str, default=None,
                       help="Chemin vers un checkpoint à charger pour continuer l'entraînement")
    
    args = parser.parse_args()
    
    # Vérifier que le dataset existe
    if not Path(args.dataset_path).exists():
        print(f"\n❌ ERREUR : Le dossier du dataset n'existe pas : {args.dataset_path}")
        sys.exit(1)
    
    # Afficher la configuration
    print("\n" + "="*80)
    print("⚙️  CONFIGURATION D'ENTRAÎNEMENT")
    print("="*80)
    print(f"\n📁 Dataset :")
    print(f"   Path        : {args.dataset_path}")
    print(f"   Image size  : {args.image_size}x{args.image_size}")
    print(f"   Val split   : {args.val_split*100:.0f}%")
    print(f"   Test split  : {args.test_split*100:.0f}%")
    print(f"   Augment     : {not args.no_augment}")
    
    print(f"\n🧠 Modèle :")
    print(f"   Architecture    : EfficientNet-B0")
    print(f"   Name            : {args.model_name}")
    print(f"   Fine-tune mode  : {args.fine_tune_mode}")
    print(f"   Dropout         : {args.dropout}")
    print(f"   Load checkpoint : {args.load_checkpoint if args.load_checkpoint else 'None'}")
    
    print(f"\n🎓 Entraînement :")
    print(f"   Batch size        : {args.batch_size}")
    print(f"   Epochs            : {args.epochs}")
    print(f"   Learning rate     : {args.learning_rate}")
    print(f"   Weight decay      : {args.weight_decay}")
    print(f"   Patience          : {args.patience}")
    print(f"   Gradient clip     : {args.gradient_clip}")
    
    print(f"\n🔧 Régularisation :")
    print(f"   Mixup             : {args.use_mixup} (alpha={args.mixup_alpha})")
    print(f"   Label smoothing   : {args.label_smoothing}")
    print(f"   Cosine annealing  : {args.use_cosine_annealing}")
    
    print(f"\n💾 Sauvegarde :")
    print(f"   Directory : {args.save_dir}/{args.model_name}/")
    
    print("="*80)
    
    # Recommandations basées sur le mode
    if args.fine_tune_mode == 'head_only':
        print("\n💡 MODE HEAD_ONLY - Phase 1 :")
        print("   ✅ Idéal pour démarrer l'entraînement")
        print("   ✅ Rapide (10-20 epochs suffisent)")
        print("   ✅ Permet au classifier de converger")
        print("   ➡️  Prochaine étape : mode 'partial' avec lr=5e-4")
    elif args.fine_tune_mode == 'partial':
        print("\n💡 MODE PARTIAL - Phase 2 :")
        print("   ✅ Fine-tune les derniers blocs")
        print("   ✅ Amélioration progressive (30-50 epochs)")
        print("   ➡️  Prochaine étape : mode 'full' avec lr=1e-4")
    elif args.fine_tune_mode == 'full':
        print("\n💡 MODE FULL - Phase 3 :")
        print("   ✅ Fine-tune complet pour performances maximales")
        print("   ✅ Learning rate plus bas recommandé (1e-4)")
        print("   🎯 Objectif : 62-68% accuracy sur FER2013")
    
    print("="*80 + "\n")
    
    # Lancer l'entraînement
    model, history = train_cnn_emotion_classifier(
        dataset_path=args.dataset_path,
        model_name=args.model_name,
        fine_tune_mode=args.fine_tune_mode,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        dropout=args.dropout,
        use_mixup=args.use_mixup,
        mixup_alpha=args.mixup_alpha,
        label_smoothing=args.label_smoothing,
        use_cosine_annealing=args.use_cosine_annealing,
        gradient_clip=args.gradient_clip,
        image_size=args.image_size,
        augment=not args.no_augment,
        val_split=args.val_split,
        test_split=args.test_split,
        num_workers=args.num_workers,
        save_dir=args.save_dir,
        load_checkpoint=args.load_checkpoint
    )
    
    print("\n" + "="*80)
    print("🎉 ENTRAÎNEMENT TERMINÉ !")
    print("="*80)
    print(f"✅ Meilleure Val Accuracy : {max(history['val_acc']):.2f}%")
    print(f"✅ Test Accuracy          : {history['test_acc']:.2f}%")
    print(f"📁 Modèle sauvegardé      : {args.save_dir}/{args.model_name}/{args.model_name}.pth")
    print("="*80)
