"""
🎯 ARCHITECTURE TRANSFORMER OPTIMISÉE POUR LANDMARKS FACIAUX
============================================================

Cette architecture est spécifiquement conçue pour traiter les features de landmarks
faciaux comme des séquences structurées plutôt que des features plates.

🔥 FONCTIONNALITÉS CLÉS :
-------------------------------------
1. **Attention Multi-Head** : Capture les relations entre landmarks (ex: coins des yeux + coins bouche)
2. **Positional Encoding** : Préserve l'information spatiale des landmarks
3. **Self-Attention** : Trouve automatiquement les patterns discriminants
4. **Architecture profonde** : 4-6 couches de Transformer pour patterns complexes

📊 ARCHITECTURE :
----------------
Input (1434) → Reshape (478 landmarks × 3 coords) 
           → Embedding (478 tokens × 128 dim)
           → Positional Encoding
           → 4x Transformer Blocks (Multi-Head Attention + FFN)
           → Global Average Pooling
           → Classification Head → 7 émotions

🎯 HYPERPARAMÈTRES OPTIMAUX :
-----------------------------
- d_model: 128 (dimension embedding)
- n_heads: 8 (attention heads)
- n_layers: 4-6 (transformer blocks)
- dropout: 0.1 (léger, car Transformers sont robustes)
- Learning Rate: 1e-4 (avec warmup)
- Batch Size: 64-128
- Label Smoothing: 0.1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Tuple, Optional


class PositionalEncoding(nn.Module):
    """
    Encodage positionnel sinusoïdal pour préserver l'ordre spatial des landmarks
    """
    def __init__(self, d_model: int, max_len: int = 200, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Créer la matrice d'encodage positionnel
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor de shape (batch_size, seq_len, d_model)
        Returns:
            Tensor avec encodage positionnel ajouté
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MultiHeadSelfAttention(nn.Module):
    """
    Attention multi-têtes pour capturer les relations entre landmarks
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model doit être divisible par n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Projections linéaires pour Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Tensor de shape (batch_size, seq_len, d_model)
            mask: Masque optionnel
        Returns:
            Tensor de shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape
        
        # Projections linéaires et reshape pour multi-head
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Appliquer l'attention
        context = torch.matmul(attn_weights, V)
        
        # Concatener les têtes et projection finale
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_o(context)
        
        return output


class FeedForward(nn.Module):
    """
    Réseau feed-forward avec expansion
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class TransformerBlock(nn.Module):
    """
    Bloc Transformer complet : Attention + FFN avec residual connections et LayerNorm
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention avec residual connection
        attn_output = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward avec residual connection
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        
        return x


class EmotionTransformer(nn.Module):
    """
    🎯 MODÈLE TRANSFORMER POUR CLASSIFICATION D'ÉMOTIONS À PARTIR DE LANDMARKS
    
    Architecture optimisée pour capturer les relations spatiales entre landmarks faciaux.
    """
    
    def __init__(
        self,
        input_dim: int = 1434,          # 478 landmarks × 3 coords
        n_classes: int = 7,            # 7 émotions
        n_landmarks: int = 478,        # Nombre de landmarks
        d_model: int = 128,            # Dimension d'embedding
        n_heads: int = 8,              # Nombre de têtes d'attention
        n_layers: int = 4,             # Nombre de blocs Transformer
        d_ff: int = 512,               # Dimension du feed-forward
        dropout: float = 0.1,          # Dropout (léger pour Transformer)
        max_len: int = None            # Longueur max pour positional encoding (auto si None)
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.n_landmarks = n_landmarks
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        
        # Projection d'entrée : (batch, 192, 3) → (batch, 192, d_model)
        self.input_projection = nn.Linear(3, d_model)
        
        # Encodage positionnel (utilise n_landmarks si max_len n'est pas spécifié)
        if max_len is None:
            max_len = n_landmarks  # Ajout d'une marge de sécurité
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        
        # Stack de blocs Transformer
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Classification head
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialisation Xavier pour une convergence stable"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Tensor de shape (batch_size, 576) ou (batch_size, 192, 3)
        
        Returns:
            Logits de shape (batch_size, 7)
        """
        batch_size = x.shape[0]
        
        # Reshape si nécessaire : (batch, 576) → (batch, 192, 3)
        if x.dim() == 2:
            x = x.view(batch_size, self.n_landmarks, 3)
        
        # Projection d'entrée : (batch, 192, 3) → (batch, 192, d_model)
        x = self.input_projection(x)
        
        # Ajouter l'encodage positionnel
        x = self.pos_encoding(x)
        
        # Passer par les blocs Transformer
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        
        # Normalisation finale
        x = self.norm(x)
        
        # Global Average Pooling : (batch, 192, d_model) → (batch, d_model)
        x = x.mean(dim=1)
        
        # Classification
        logits = self.classifier(x)
        
        return logits
    
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prédiction avec probabilités
        
        Args:
            x: Tensor de shape (batch_size, 576)
        
        Returns:
            predictions: Indices des classes (batch_size,)
            probabilities: Probabilités softmax (batch_size, 7)
        """
        self.eval()
        with torch.no_grad():
            logits = self(x)
            probabilities = F.softmax(logits, dim=-1)
            predictions = torch.argmax(probabilities, dim=-1)
        
        return predictions, probabilities
    
    def get_num_parameters(self) -> int:
        """Retourne le nombre total de paramètres"""
        return sum(p.numel() for p in self.parameters())
    
    def get_trainable_parameters(self) -> int:
        """Retourne le nombre de paramètres entraînables"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class EmotionTransformerLarge(EmotionTransformer):
    """
    Version LARGE du Transformer pour encore meilleures performances
    
    Différences :
    - Plus de couches (6 vs 4)
    - Plus de têtes d'attention (16 vs 8)
    - Dimension d'embedding plus grande (256 vs 128)
    
    Performances attendues : 80-90% accuracy
    Paramètres : ~3-5M (vs ~1M pour version standard)
    """
    
    def __init__(
        self,
        input_dim: int = 576,
        n_classes: int = 7,
        n_landmarks: int = 192,
        d_model: int = 256,       # ↑ Augmenté
        n_heads: int = 16,        # ↑ Augmenté
        n_layers: int = 6,        # ↑ Augmenté
        d_ff: int = 1024,         # ↑ Augmenté
        dropout: float = 0.1,
        max_len: int = None       # Auto-détection
    ):
        super().__init__(
            input_dim=input_dim,
            n_classes=n_classes,
            n_landmarks=n_landmarks,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
            max_len=max_len
        )


class WarmupCosineScheduler:
    """
    Learning Rate Scheduler avec warmup et cosine annealing
    Essentiel pour une bonne convergence des Transformers
    """
    
    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        total_steps: int,
        base_lr: float,
        min_lr: float = 1e-6
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.current_step = 0
    
    def step(self):
        """Mise à jour du learning rate"""
        self.current_step += 1
        
        if self.current_step < self.warmup_steps:
            # Phase de warmup : augmentation linéaire
            lr = self.base_lr * self.current_step / self.warmup_steps
        else:
            # Phase de cosine annealing
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr
    
    def get_last_lr(self):
        """Retourne le dernier learning rate"""
        return [group['lr'] for group in self.optimizer.param_groups]


class ModelManager:
    """
    Gestionnaire pour sauvegarder/charger les modèles Transformer
    Compatible avec les modèles EmotionTransformer et EmotionTransformerLarge
    Inclut également les utilitaires d'entraînement
    """
    
    @staticmethod
    def save_model(
        model: nn.Module,
        save_dir: str,
        model_name: str,
        metadata: dict = None
    ) -> dict:
        """
        Sauvegarder le modèle Transformer avec ses métadonnées
        
        Args:
            model: Modèle PyTorch (EmotionTransformer ou EmotionTransformerLarge)
            save_dir: Dossier de sauvegarde
            model_name: Nom du modèle
            metadata: Métadonnées additionnelles (optionnel)
        
        Returns:
            Dict avec les chemins des fichiers sauvegardés
        """
        from pathlib import Path
        import json
        
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Préparer les métadonnées du modèle
        model_metadata = {
            'model_type': model.__class__.__name__,
            'input_dim': model.input_dim,
            'n_classes': model.n_classes,
            'n_landmarks': model.n_landmarks,
            'd_model': model.d_model,
            'n_heads': model.n_heads,
            'n_layers': model.n_layers,
            'total_parameters': model.get_num_parameters(),
            'trainable_parameters': model.get_trainable_parameters()
        }
        
        # Ajouter les métadonnées additionnelles
        if metadata:
            model_metadata.update(metadata)
        
        # Sauvegarder le modèle
        model_path = save_path / f"{model_name}.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'metadata': model_metadata
        }, model_path)
        
        # Sauvegarder les métadonnées en JSON
        metadata_path = save_path / f"{model_name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(model_metadata, f, indent=2)
        
        print(f"\n💾 Modèle sauvegardé :")
        print(f"   Modèle    : {model_path}")
        print(f"   Metadata  : {metadata_path}")
        
        return {
            'model': str(model_path),
            'metadata': str(metadata_path)
        }
    
    @staticmethod
    def load_model(
        model_path: str,
        device: str = 'cpu',
        model_class = None
    ) -> nn.Module:
        """
        Charger un modèle Transformer sauvegardé
        
        Args:
            model_path: Chemin vers le fichier .pth
            device: Device ('cpu' ou 'cuda')
            model_class: Classe du modèle (EmotionTransformer ou EmotionTransformerLarge)
                        Si None, détection automatique depuis les métadonnées
        
        Returns:
            Modèle chargé
        """
        from pathlib import Path
        
        # Charger le checkpoint
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        metadata = checkpoint['metadata']
        
        print(f"\n📂 Chargement du modèle : {model_path}")
        print(f"   Type      : {metadata['model_type']}")
        print(f"   d_model   : {metadata['d_model']}")
        print(f"   n_heads   : {metadata['n_heads']}")
        print(f"   n_layers  : {metadata['n_layers']}")
        print(f"   Parameters: {metadata['total_parameters']:,}")
        
        # Déterminer la classe du modèle
        if model_class is None:
            if metadata['model_type'] == 'EmotionTransformer':
                model_class = EmotionTransformer
            elif metadata['model_type'] == 'EmotionTransformerLarge':
                model_class = EmotionTransformerLarge
            else:
                raise ValueError(f"Type de modèle inconnu : {metadata['model_type']}")
        
        # Créer le modèle avec les mêmes paramètres
        model = model_class(
            input_dim=metadata['input_dim'],
            n_classes=metadata['n_classes'],
            n_landmarks=metadata['n_landmarks'],
            d_model=metadata['d_model'],
            n_heads=metadata['n_heads'],
            n_layers=metadata['n_layers'],
        )
        
        # Charger les poids
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        
        print(f"✅ Modèle chargé avec succès sur {device}")
        
        return model
    
    @staticmethod
    def load_metadata(model_path: str) -> dict:
        """
        Charger uniquement les métadonnées d'un modèle
        
        Args:
            model_path: Chemin vers le fichier .pth
        
        Returns:
            Dict contenant les métadonnées
        """
        device = 'cpu' if torch.cuda.is_available() else 'cpu'
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        return checkpoint.get('metadata', {})
    
    @staticmethod
    def train_model(
        model: nn.Module,
        train_loader,
        val_loader,
        test_loader,
        n_epochs: int,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        warmup_steps: int = 1000,
        label_smoothing: float = 0.1,
        gradient_clip: float = 1.0,
        patience: int = 30,
        save_dir: str = "./models",
        model_name: str = "emotion_transformer"
    ) -> dict:
        """
        Entraîner un modèle Transformer avec toutes les optimisations
        
        Args:
            model: Modèle à entraîner
            train_loader: DataLoader d'entraînement
            val_loader: DataLoader de validation
            test_loader: DataLoader de test
            n_epochs: Nombre d'epochs
            device: Device ('cpu' ou 'cuda')
            learning_rate: Learning rate initial
            weight_decay: Weight decay pour AdamW
            warmup_steps: Nombre de steps de warmup
            label_smoothing: Label smoothing
            gradient_clip: Gradient clipping
            patience: Patience pour early stopping
            save_dir: Dossier de sauvegarde
            model_name: Nom du modèle
        
        Returns:
            Dict contenant l'historique d'entraînement
        """
        from pathlib import Path
        from tqdm import tqdm
        import torch.optim as optim
        import matplotlib.pyplot as plt
        
        model = model.to(device)
        
        # Optimizer AdamW (meilleur pour Transformers)
        optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.98),
            eps=1e-9
        )
        
        # Calculer le nombre total de steps
        total_steps = len(train_loader) * n_epochs
        
        # Learning Rate Scheduler avec warmup
        scheduler = WarmupCosineScheduler(
            optimizer=optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            base_lr=learning_rate,
            min_lr=1e-6
        )
        
        # Loss avec Label Smoothing
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        
        # Historique
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'test_loss': [],
            'test_acc': [],
            'learning_rates': []
        }
        
        best_val_acc = 0.0
        best_epoch = 0
        patience_counter = 0
        
        print("\n" + "="*80)
        print("🚀 DÉBUT DE L'ENTRAÎNEMENT TRANSFORMER")
        print("="*80)
        print(f"Device            : {device}")
        print(f"Learning Rate     : {learning_rate}")
        print(f"Warmup Steps      : {warmup_steps}")
        print(f"Total Steps       : {total_steps}")
        print(f"Epochs            : {n_epochs}")
        print(f"Early Stopping    : {patience} epochs")
        print(f"Gradient Clipping : {gradient_clip}")
        print(f"Label Smoothing   : {label_smoothing}")
        print(f"Train samples     : {len(train_loader.dataset)}")
        print(f"Val samples       : {len(val_loader.dataset)}")
        print(f"Test samples      : {len(test_loader.dataset)}")
        print(f"Batch size        : {train_loader.batch_size}")
        print(f"Model params      : {model.get_num_parameters():,}")
        print("="*80)
        
        for epoch in range(n_epochs):
            # ========== ENTRAÎNEMENT ==========
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            pbar = tqdm(train_loader, desc="Training", leave=False, ncols=100)
            
            for batch_x, batch_y in pbar:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                # Forward pass
                optimizer.zero_grad()
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                if gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                
                optimizer.step()
                
                # Mise à jour du learning rate
                current_lr = scheduler.step()
                
                # Métriques
                running_loss += loss.item() * batch_x.size(0)
                predictions = torch.argmax(logits, dim=1)
                correct += (predictions == batch_y).sum().item()
                total += batch_y.size(0)
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100 * correct / total:.2f}%',
                    'lr': f'{current_lr:.6f}'
                })
            
            train_loss = running_loss / total
            train_acc = correct / total
            
            # ========== VALIDATION ==========
            model.eval()
            running_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    
                    logits = model(batch_x)
                    loss = criterion(logits, batch_y)
                    
                    running_loss += loss.item() * batch_x.size(0)
                    predictions = torch.argmax(logits, dim=1)
                    correct += (predictions == batch_y).sum().item()
                    total += batch_y.size(0)
            
            val_loss = running_loss / total
            val_acc = correct / total
            
            # ========== TEST ==========
            running_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch_x, batch_y in test_loader:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    
                    logits = model(batch_x)
                    loss = criterion(logits, batch_y)
                    
                    running_loss += loss.item() * batch_x.size(0)
                    predictions = torch.argmax(logits, dim=1)
                    correct += (predictions == batch_y).sum().item()
                    total += batch_y.size(0)
            
            test_loss = running_loss / total
            test_acc = correct / total
            
            # Sauvegarder l'historique
            current_lr = scheduler.get_last_lr()[0]
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['test_loss'].append(test_loss)
            history['test_acc'].append(test_acc)
            history['learning_rates'].append(current_lr)
            
            # Affichage
            print(f"\n📊 Epoch {epoch + 1}/{n_epochs}")
            print(f"   Train : Loss = {train_loss:.4f}, Acc = {train_acc*100:.2f}%")
            print(f"   Val   : Loss = {val_loss:.4f}, Acc = {val_acc*100:.2f}%")
            print(f"   Test  : Loss = {test_loss:.4f}, Acc = {test_acc*100:.2f}%")
            print(f"   LR    : {current_lr:.6f}")
            
            # Early stopping et sauvegarde du meilleur modèle
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                patience_counter = 0
                
                # Préparer les métadonnées du modèle
                model_metadata = {
                    'model_type': model.__class__.__name__,
                    'input_dim': model.input_dim,
                    'n_classes': model.n_classes,
                    'n_landmarks': model.n_landmarks,
                    'd_model': model.d_model,
                    'n_heads': model.n_heads,
                    'n_layers': model.n_layers,
                    'total_parameters': model.get_num_parameters(),
                    'trainable_parameters': model.get_trainable_parameters(),
                    'best_epoch': epoch + 1,
                    'best_val_acc': float(best_val_acc),
                    'train_acc': float(train_acc),
                    'test_acc': float(test_acc)
                }
                
                # Sauvegarder le meilleur modèle avec métadonnées
                save_path = Path(save_dir) / f"{model_name}.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_acc': best_val_acc,
                    'history': history,
                    'metadata': model_metadata
                }, save_path)
                
                print(f"   ✅ Nouveau meilleur modèle sauvegardé ! Val Acc = {val_acc*100:.2f}%")
            else:
                patience_counter += 1
                print(f"   ⏳ Patience : {patience_counter}/{patience}")
            
            # Early stopping
            if patience_counter >= patience:
                print(f"\n🛑 Early stopping à l'epoch {epoch + 1}")
                print(f"   Meilleure Val Acc : {best_val_acc*100:.2f}% (epoch {best_epoch + 1})")
                break
        
        # Charger le meilleur modèle pour évaluation finale
        checkpoint = torch.load(Path(save_dir) / f"{model_name}.pth", map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Évaluation finale sur le test set
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                
                running_loss += loss.item() * batch_x.size(0)
                predictions = torch.argmax(logits, dim=1)
                correct += (predictions == batch_y).sum().item()
                total += batch_y.size(0)
        
        final_test_loss = running_loss / total
        final_test_acc = correct / total
        
        print("\n" + "="*80)
        print("🎉 ENTRAÎNEMENT TERMINÉ")
        print("="*80)
        print(f"Meilleure Val Acc  : {best_val_acc*100:.2f}% (epoch {best_epoch + 1})")
        print(f"Test Acc finale    : {final_test_acc*100:.2f}%")
        print("="*80)
        
        return history
    
    @staticmethod
    def plot_history(history: dict, save_path: str):
        """
        Tracer les courbes d'entraînement
        
        Args:
            history: Historique d'entraînement
            save_path: Chemin de sauvegarde du graphique
        """
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        axes[0, 0].plot(history['train_loss'], label='Train Loss')
        axes[0, 0].plot(history['val_loss'], label='Val Loss')
        axes[0, 0].plot(history['test_loss'], label='Test Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Loss Evolution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[0, 1].plot([acc * 100 for acc in history['train_acc']], label='Train Acc')
        axes[0, 1].plot([acc * 100 for acc in history['val_acc']], label='Val Acc')
        axes[0, 1].plot([acc * 100 for acc in history['test_acc']], label='Test Acc')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].set_title('Accuracy Evolution')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Learning Rate
        axes[1, 0].plot(history['learning_rates'])
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Val vs Test Accuracy
        axes[1, 1].plot([acc * 100 for acc in history['val_acc']], label='Val Acc')
        axes[1, 1].plot([acc * 100 for acc in history['test_acc']], label='Test Acc')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy (%)')
        axes[1, 1].set_title('Val vs Test Accuracy')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Courbes sauvegardées : {save_path}")


def predict_emotion_transformer(
    image_path: str,
    model: nn.Module,
    device: str = 'cpu'
) -> dict:
    """
    Prédire l'émotion d'une image avec un modèle Transformer
    
    Args:
        image_path: Chemin vers l'image
        model: Modèle Transformer (EmotionTransformer ou EmotionTransformerLarge)
        device: 'cpu' ou 'cuda'
    
    Returns:
        Dict avec la prédiction et les probabilités
    """
    import cv2
    from classes.FaceLandmarkExtractor import FaceLandmarkExtractor
    
    emotion_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    
    # Charger l'image
    image = cv2.imread(image_path)
    if image is None:
        return {'success': False, 'error': f"Impossible de charger l'image : {image_path}"}
    
    # Extraire les landmarks
    extractor = FaceLandmarkExtractor(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    )
    
    landmarks = extractor.extract_landmarks(image)
    if landmarks is None:
        return {
            'success': False,
            'error': 'Aucun visage détecté dans l\'image',
            'image_path': image_path
        }
    normalized_landmarks = extractor.normalize_face_part_landmarks(landmarks)
    
    # Normaliser et convertir en features
    features = extractor.landmarks_to_features(normalized_landmarks)
    
    # Convertir en tensor
    features_tensor = torch.from_numpy(features).float().unsqueeze(0).to(device)
    
    # Prédiction
    predictions, probabilities = model.predict(features_tensor)
    
    pred_class = predictions.item()
    pred_probs = probabilities.cpu().numpy()[0]
    
    # Construire le résultat
    result = {
        'success': True,
        'predicted_emotion': emotion_names[pred_class],
        'predicted_class_id': pred_class,
        'confidence': float(pred_probs[pred_class]),
        'all_probabilities': {
            emotion_names[i]: float(pred_probs[i])
            for i in range(len(emotion_names))
        },
        'image_path': image_path
    }
    
    del extractor
    
    return result