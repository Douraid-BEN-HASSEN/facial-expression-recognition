"""
🎯 SCRIPT D'ENTRAÎNEMENT CLASSIFIER POUR CLASSIFICATION D'ÉMOTIONS
==================================================================================

Script pour entraîner un classifieur d'émotions à partir de features extraites.

Modèle simple ou ResNet

UTILISATION :
-------------
python scripts/train_script.py \
    --data_dir ./datasets/fer2013_features \
    --model_type standard \
    --input_dim 478 \
    --batch_size 128 \
    --n_epochs 100 \
    --learning_rate 0.001 \
    --early_stopping_patience 15 \
    --dropout_rate 0.3 \
    --save_dir ./models \
    --model_name emotion_classifier

"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Dict, Tuple
import argparse

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from classes.EmotionClassifierModel import (
    EmotionClassifier, 
    EmotionClassifierResNet,
    ModelManager
)

class Trainer:
    """
    Classe pour gérer l'entraînement du modèle
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5
    ):
        """
        Initialise le trainer
        
        Args:
            model: Modèle PyTorch à entraîner
            device: Device d'entraînement ('cpu' ou 'cuda')
            learning_rate: Taux d'apprentissage
            weight_decay: Régularisation L2
        """
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        
        # Optimizer avec weight decay pour régularisation
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Loss avec Label Smoothing léger
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
        
        # Historique
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'test_loss': [],
            'test_acc': [],
            'learning_rates': []
        }
        
        # Best model tracking
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.best_model_state = None
        
        # Mixup optimal de l'essai précédent
        self.use_mixup = True
        self.mixup_alpha = 0.2  
        self.mixup_prob = 0.4  # Appliqué 40% du temps
    
    def mixup_data(self, x, y, alpha=0.2):
        """
        Mixup: mélange aléatoire de paires d'exemples
        Améliore la généralisation en créant des exemples virtuels
        """
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        
        batch_size = x.size()[0]
        index = torch.randperm(batch_size).to(x.device)
        
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam
    
    def mixup_criterion(self, pred, y_a, y_b, lam):
        """Loss function pour Mixup"""
        return lam * self.criterion(pred, y_a) + (1 - lam) * self.criterion(pred, y_b)
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """
        Entraîner sur une epoch avec Mixup
        
        Returns:
            train_loss, train_accuracy
        """
        self.model.train()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc="Training", leave=False, ncols=100)
        
        for batch_x, batch_y in pbar:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Appliquer Mixup avec probabilité réduite (30%)
            if self.use_mixup and np.random.rand() > (1 - self.mixup_prob):
                mixed_x, y_a, y_b, lam = self.mixup_data(batch_x, batch_y, self.mixup_alpha)
                outputs = self.model(mixed_x)
                loss = self.mixup_criterion(outputs, y_a, y_b, lam)
            else:
                # Forward pass normal
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
            
            # Backward pass
            loss.backward()
            
            # Clip gradients pour stabilité
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update weights
            self.optimizer.step()
            
            # Statistiques
            running_loss += loss.item() * batch_x.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * correct / total:.2f}%'
            })
        
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        Valider le modèle
        
        Returns:
            val_loss, val_accuracy
        """
        self.model.eval()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # Forward pass
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                
                # Statistiques
                running_loss += loss.item() * batch_x.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        val_loss = running_loss / total
        val_acc = correct / total
        
        return val_loss, val_acc
    
    def test(self, test_loader: DataLoader) -> Tuple[float, float]:
        """
        Tester le modèle
        
        Returns:
            test_loss, test_accuracy
        """
        self.model.eval()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # Forward pass
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                
                # Statistiques
                running_loss += loss.item() * batch_x.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        test_loss = running_loss / total
        test_acc = correct / total
        
        return test_loss, test_acc
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        n_epochs: int = 100,
        early_stopping_patience: int = 15,
        lr_scheduler_patience: int = 5,
        save_dir: str = "./models",
        model_name: str = "emotion_classifier"
    ) -> Dict:
        """
        Entraîner le modèle avec early stopping
        
        Args:
            train_loader: DataLoader d'entraînement
            val_loader: DataLoader de validation
            test_loader: DataLoader de test
            n_epochs: Nombre maximum d'epochs
            early_stopping_patience: Patience pour early stopping
            lr_scheduler_patience: Patience pour réduire le learning rate
            save_dir: Dossier de sauvegarde
            model_name: Nom du modèle
        
        Returns:
            Dictionnaire avec l'historique d'entraînement
        """
        print("="*80)
        print("🚀 DÉBUT DE L'ENTRAÎNEMENT")
        print("="*80)
        print(f"Device            : {self.device}")
        print(f"Learning Rate     : {self.learning_rate}")
        print(f"Epochs max        : {n_epochs}")
        print(f"Early Stopping    : {early_stopping_patience} epochs")
        print(f"Train samples     : {len(train_loader.dataset)}")
        print(f"Val samples       : {len(val_loader.dataset)}")
        print(f"Test samples      : {len(test_loader.dataset)}")
        print(f"Batch size        : {train_loader.batch_size}")
        print(f"Paramètres model  : {self.model.get_num_parameters():,}")
        print("="*80 + "\n")
        
        # Learning rate scheduler plus conservateur
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,  # Réduction standard
            patience=10,  # Patience réduite pour ajuster plus tôt
            min_lr=1e-5,
        )
        
        # Early stopping
        patience_counter = 0
        
        for epoch in range(n_epochs):
            print(f"Epoch {epoch+1}/{n_epochs}")
            
            # Entraîner
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Valider
            val_loss, val_acc = self.validate(val_loader)
            
            # Tester
            test_loss, test_acc = self.test(test_loader)

            # Sauvegarder l'historique
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['test_loss'].append(test_loss)
            self.history['test_acc'].append(test_acc)
            self.history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
            
            # Afficher les résultats
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
            print(f"  Val Loss  : {val_loss:.4f} | Val Acc  : {val_acc*100:.2f}%")
            print(f"  Test Loss : {test_loss:.4f} | Test Acc : {test_acc*100:.2f}%")
            
            # Sauvegarder le meilleur modèle
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch + 1
                self.best_model_state = self.model.state_dict().copy()
                patience_counter = 0
                print(f"  ✅ Nouveau meilleur modèle ! Val Acc: {val_acc*100:.2f}%")
            else:
                patience_counter += 1
            
            # Learning rate scheduling
            scheduler.step(val_acc)
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\n⏹️  Early stopping après {epoch+1} epochs")
                print(f"   Meilleur modèle à l'epoch {self.best_epoch} avec Val Acc: {self.best_val_acc*100:.2f}%")
                break
            
            print()
        
        # Restaurer le meilleur modèle
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print(f"\n✅ Meilleur modèle restauré (Epoch {self.best_epoch}, Val Acc: {self.best_val_acc*100:.2f}%)")
        
        # Résumé final
        print("\n" + "="*80)
        print("✅ ENTRAÎNEMENT TERMINÉ")
        print("="*80)
        print(f"Meilleur Val Acc   : {self.best_val_acc*100:.2f}%")
        print(f"Meilleure epoch    : {self.best_epoch}")
        print(f"Epochs effectuées  : {len(self.history['train_loss'])}")
        print("="*80)
        
        return self.history
    
    def plot_history(self, save_path: str = None):
        """
        Tracer les courbes d'entraînement
        
        Args:
            save_path: Chemin pour sauvegarder le graphique
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Loss
        axes[0, 0].plot(epochs, self.history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        axes[0, 0].plot(epochs, self.history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        axes[0, 0].plot(epochs, self.history['test_loss'], 'orange', label='Test Loss', linewidth=2)
        axes[0, 0].axvline(x=self.best_epoch, color='g', linestyle='--', label=f'Best Epoch ({self.best_epoch})')
        axes[0, 0].set_xlabel('Epoch', fontsize=12)
        axes[0, 0].set_ylabel('Loss', fontsize=12)
        axes[0, 0].set_title('Loss During Training', fontsize=14, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[0, 1].plot(epochs, [acc*100 for acc in self.history['train_acc']], 'b-', label='Train Acc', linewidth=2)
        axes[0, 1].plot(epochs, [acc*100 for acc in self.history['val_acc']], 'r-', label='Val Acc', linewidth=2)
        axes[0, 1].plot(epochs, [acc*100 for acc in self.history['test_acc']], 'orange', label='Test Acc', linewidth=2)
        axes[0, 1].axvline(x=self.best_epoch, color='g', linestyle='--', label=f'Best Epoch ({self.best_epoch})')
        axes[0, 1].set_xlabel('Epoch', fontsize=12)
        axes[0, 1].set_ylabel('Accuracy (%)', fontsize=12)
        axes[0, 1].set_title('Accuracy During Training', fontsize=14, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Learning Rate
        axes[1, 0].plot(epochs, self.history['learning_rates'], 'purple', linewidth=2)
        axes[1, 0].set_xlabel('Epoch', fontsize=12)
        axes[1, 0].set_ylabel('Learning Rate', fontsize=12)
        axes[1, 0].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Overfitting Gap
        gap = [train - val for train, val in zip(self.history['train_acc'], self.history['val_acc'])]
        axes[1, 1].plot(epochs, [g*100 for g in gap], 'orange', linewidth=2)
        axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1, 1].set_xlabel('Epoch', fontsize=12)
        axes[1, 1].set_ylabel('Gap (%)', fontsize=12)
        axes[1, 1].set_title('Train-Val Accuracy Gap (Overfitting)', fontsize=14, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"📊 Graphique sauvegardé : {save_path}")
        
        plt.show()

def train_emotion_classifier(
    data_dir: str = "./supervised_data",
    model_type: str = "standard",  # 'standard', 'resnet'
    input_dim: int = 478,
    batch_size: int = 128,
    n_epochs: int = 100,
    learning_rate: float = 0.001,
    early_stopping_patience: int = 15,
    dropout_rate: float = 0.3,
    save_dir: str = "./models",
    model_name: str = "emotion_classifier"
):
    """
    Fonction principale pour entraîner le modèle
    
    Args:
        data_dir: Dossier contenant les données (X_train.npy, y_train.npy, etc.)
        model_type: Type de modèle ('standard', 'resnet')
        input_dim: Dimension des features d'entrée
        batch_size: Taille des batchs
        n_epochs: Nombre maximum d'epochs
        learning_rate: Taux d'apprentissage
        early_stopping_patience: Patience pour early stopping
        dropout_rate: Taux de dropout
        save_dir: Dossier de sauvegarde
        model_name: Nom du modèle
    """
    
    print("="*80)
    print("🎭 ENTRAÎNEMENT DU CLASSIFIEUR D'ÉMOTIONS")
    print("="*80)
    
    # Charger les métadonnées
    metadata_path = Path(data_dir) / "metadata.json"
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    print(f"\n📊 Métadonnées du dataset :")
    print(f"   Features      : {metadata['feature_dimension']} dimensions")
    print(f"   Classes       : {metadata['n_classes']}")
    
    # Charger les données
    print(f"\n📁 Chargement des données depuis : {data_dir}")
    
    X_train = np.load(Path(data_dir) / "X_train.npy")
    y_train = np.load(Path(data_dir) / "y_train.npy")
    X_val = np.load(Path(data_dir) / "X_val.npy")
    y_val = np.load(Path(data_dir) / "y_val.npy")
    X_test = np.load(Path(data_dir) / "X_test.npy")
    y_test = np.load(Path(data_dir) / "y_test.npy")
    
    print(f"   Train : {X_train.shape}")
    print(f"   Val   : {X_val.shape}")
    print(f"   Test  : {X_test.shape}")
    
    # Convertir en tensors
    X_train_tensor = torch.from_numpy(X_train)
    y_train_tensor = torch.from_numpy(y_train)
    X_val_tensor = torch.from_numpy(X_val)
    y_val_tensor = torch.from_numpy(y_val)
    X_test_tensor = torch.from_numpy(X_test)
    y_test_tensor = torch.from_numpy(y_test)
    
    # Créer les DataLoaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0  # Windows compatible
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Créer le modèle
    print(f"\n🧠 Création du modèle : {model_type.upper()}")
    
    if model_type == "standard":
        model = EmotionClassifier(
            input_dim=input_dim,
            n_classes=7,
            hidden_dims=[512, 256, 128, 64],
            dropout_rate=dropout_rate,
            use_batch_norm=True
        )
    elif model_type == "resnet":
        model = EmotionClassifierResNet(
            input_dim=input_dim,
            n_classes=7,
            hidden_dims=[512, 512, 256, 256, 128],
            dropout_rate=dropout_rate
        )
        print(f"   🎭 Architecture optimisée pour masques faciaux")
        print(f"   ✓ Attention mechanism activé")
        print(f"   ✓ Multi-scale feature extraction (3 branches)")
        print(f"   ✓ Residual connections")
    else:
        raise ValueError(f"Type de modèle inconnu : {model_type}. Utilisez 'standard', 'resnet' ou 'facial_mask'")
    
    print(f"   Paramètres : {model.get_num_parameters():,}")
    
    # Créer le trainer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = Trainer(
        model=model,
        device=device,
        learning_rate=learning_rate,
        weight_decay=4e-4  # Weight decay optimal
    )
    
    # Entraîner
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        n_epochs=n_epochs,
        early_stopping_patience=early_stopping_patience,
        save_dir=save_dir,
        model_name=model_name
    )
    
    # Sauvegarder le modèle
    print(f"\n💾 Sauvegarde du modèle...")
    
    files = ModelManager.save_model(
        model=model,
        save_dir=save_dir,
        model_name=model_name
    )
    
    # Sauvegarder l'historique
    history_path = Path(save_dir) / f"{model_name}_history.json"
    with open(history_path, 'w') as f:
        # Convertir en types Python natifs pour JSON
        history_json = {
            k: [float(v) for v in vals] for k, vals in history.items()
        }
        json.dump(history_json, f, indent=2)
    
    print(f"✅ Historique sauvegardé : {history_path}")
    
    # Tracer les courbes
    plot_path = Path(save_dir) / f"{model_name}_training_curves.png"
    trainer.plot_history(save_path=str(plot_path))
    
    print("\n" + "="*80)
    print("🎉 ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS")
    print("="*80)
    print(f"\n📦 Fichiers sauvegardés :")
    for key, path in files.items():
        print(f"   {key:12} : {path}")
    print(f"   history      : {history_path}")
    print(f"   curves       : {plot_path}")
    print("="*80)
    
    return model, history

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Entraîner un modèle pour la classification d'émotions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="""
Exemples d'utilisation :
------------------------
python ./scripts/train_classifier_script.py --dataset_path ./datasets/fer2013_features --model_name emotion_classifier --model_type standard --batch_size 256 --epochs 100 --learning_rate 0.002 --dropout_rate 0.35 --patience 20 --input_dim 1434
        """)
    
    parser.add_argument("--dataset_path", type=str, default="./datasets/fer2013_features",
                       help="Chemin vers le dataset")
    parser.add_argument("--model_name", type=str, default="emotion_classifier",
                       help="Nom du modèle à sauvegarder")
    parser.add_argument("--model_type", type=str, default="standard",
                       choices=["standard", "resnet"],
                       help="Type d'architecture (standard, resnet)")
    parser.add_argument("--batch_size", type=int, default=256,
                       help="Taille des batchs")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Nombre maximum d'epochs")
    parser.add_argument("--learning_rate", type=float, default=0.002,
                          help="Taux d'apprentissage")
    parser.add_argument("--dropout_rate", type=float, default=0.3,
                       help="Taux de dropout")
    parser.add_argument("--patience", type=int, default=15,
                          help="Patience pour early stopping")
    parser.add_argument("--input_dim", type=int, default=1434,
                       help="Dimension des features d'entrée")
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("🚀 CONFIGURATION")
    print("="*80)
    for arg, value in vars(args).items():
        print(f"   {arg:20} : {value}")
    print("="*80 + "\n")

    model, history = train_emotion_classifier(
        data_dir=args.dataset_path,
        model_type=args.model_type,
        input_dim=args.input_dim,
        batch_size=args.batch_size,
        n_epochs=args.epochs,
        learning_rate=args.learning_rate,
        dropout_rate=args.dropout_rate,
        early_stopping_patience=args.patience,
        save_dir=f"./models/{args.model_name}",
        model_name=args.model_name
    )
