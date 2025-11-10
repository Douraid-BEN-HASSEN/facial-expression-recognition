"""
Modèle de classification supervisée pour les expressions faciales
Architecture optimisée pour les features de landmarks (1434 dimensions)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, List
import numpy as np
from pathlib import Path


class EmotionClassifier(nn.Module):
    """
    Architecture de réseau de neurones profond pour la classification d'émotions
    à partir de features de landmarks faciaux
    
    Architecture:
        - Input: 1434 features (478 landmarks × 3 coordonnées)
        - 5 couches fully connected avec BatchNorm et Dropout
        - Output: 7 classes d'émotions
    
    Features:
        ✅ Batch Normalization pour stabiliser l'entraînement
        ✅ Dropout pour régularisation
        ✅ Residual connections pour faciliter l'apprentissage
        ✅ LeakyReLU pour éviter les neurones morts
    """
    
    def __init__(
        self,
        input_dim: int = 1434,
        n_classes: int = 7,
        hidden_dims: List[int] = [512, 256, 128, 64],
        dropout_rate: float = 0.3,
        use_batch_norm: bool = True
    ):
        """
        Initialise le modèle
        
        Args:
            input_dim: Dimension des features d'entrée (1434 pour landmarks)
            n_classes: Nombre de classes de sortie (7 émotions)
            hidden_dims: Liste des dimensions des couches cachées
            dropout_rate: Taux de dropout pour régularisation
            use_batch_norm: Utiliser BatchNorm ou non
        """
        super(EmotionClassifier, self).__init__()
        
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        
        # Construire les couches
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            # Couche linéaire
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Batch Normalization
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            # Activation
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            
            # Dropout (sauf dernière couche cachée)
            if i < len(hidden_dims) - 1:
                layers.append(nn.Dropout(dropout_rate))
            
            prev_dim = hidden_dim
        
        # Encoder les couches cachées
        self.hidden_layers = nn.Sequential(*layers)
        
        # Couche de sortie
        self.output_layer = nn.Linear(hidden_dims[-1], n_classes)
        
        # Initialisation des poids
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialisation He pour LeakyReLU"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Tensor de shape (batch_size, 1434)
        
        Returns:
            Logits de shape (batch_size, 7)
        """
        # Passer par les couches cachées
        features = self.hidden_layers(x)
        
        # Couche de sortie
        logits = self.output_layer(features)
        
        return logits
    
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prédiction avec probabilities
        
        Args:
            x: Tensor de shape (batch_size, 1434)
        
        Returns:
            predictions: Indices des classes prédites (batch_size,)
            probabilities: Probabilités softmax (batch_size, 7)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = F.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
        
        return predictions, probabilities
    
    def get_num_parameters(self) -> int:
        """Retourne le nombre total de paramètres"""
        return sum(p.numel() for p in self.parameters())
    
    def get_trainable_parameters(self) -> int:
        """Retourne le nombre de paramètres entraînables"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class EmotionClassifierResNet(nn.Module):
    """
    Version avec Residual Connections pour améliorer l'apprentissage profond
    Meilleure convergence pour des architectures plus profondes
    """
    
    def __init__(
        self,
        input_dim: int = 1434,
        n_classes: int = 7,
        hidden_dims: List[int] = [512, 512, 256, 256, 128],
        dropout_rate: float = 0.3
    ):
        super(EmotionClassifierResNet, self).__init__()
        
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        
        # Première couche de projection
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout_rate)
        )
        
        # Blocs résiduels
        self.residual_blocks = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.residual_blocks.append(
                ResidualBlock(
                    hidden_dims[i],
                    hidden_dims[i + 1],
                    dropout_rate=dropout_rate
                )
            )
        
        # Couche de sortie
        self.output_layer = nn.Linear(hidden_dims[-1], n_classes)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Projection initiale
        x = self.input_projection(x)
        
        # Passer par les blocs résiduels
        for block in self.residual_blocks:
            x = block(x)
        
        # Couche de sortie
        logits = self.output_layer(x)
        
        return logits
    
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prédiction avec probabilités"""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = F.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
        return predictions, probabilities
    
    def get_num_parameters(self) -> int:
        """Retourne le nombre total de paramètres"""
        return sum(p.numel() for p in self.parameters())
    
    def get_trainable_parameters(self) -> int:
        """Retourne le nombre de paramètres entraînables"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ResidualBlock(nn.Module):
    """Bloc résiduel avec skip connection - régularisation équilibrée"""
    
    def __init__(self, in_dim: int, out_dim: int, dropout_rate: float = 0.3):
        super(ResidualBlock, self).__init__()
        
        self.block = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout_rate),  # Un seul dropout
            nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim)
        )
        
        # Skip connection avec projection si dimensions différentes
        if in_dim != out_dim:
            self.skip_connection = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim)
            )
        else:
            self.skip_connection = nn.Identity()
        
        self.activation = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip_connection(x)
        out = self.block(x)
        out = out + residual  # Skip connection
        out = self.activation(out)
        return out


class ModelManager:
    """
    Gestionnaire pour sauvegarder/charger le modèle
    """
    
    @staticmethod
    def save_model(
        model: nn.Module,
        save_dir: str = "./models",
        model_name: str = "emotion_classifier"
    ) -> Dict[str, str]:
        """
        Sauvegarde le modèle
        
        Args:
            model: Modèle PyTorch à sauvegarder
            save_dir: Dossier de sauvegarde
            model_name: Nom du modèle
        
        Returns:
            Dict avec les chemins des fichiers sauvegardés
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Sauvegarder les poids du modèle
        model_file = save_path / f"{model_name}.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': {
                'class_name': model.__class__.__name__,
                'input_dim': model.input_dim,
                'n_classes': model.n_classes,
                'hidden_dims': model.hidden_dims if hasattr(model, 'hidden_dims') else None,
                'dropout_rate': model.dropout_rate if hasattr(model, 'dropout_rate') else None,
                'use_batch_norm': model.use_batch_norm if hasattr(model, 'use_batch_norm') else None,
            }
        }, model_file)
        
        
        print(f"✅ Modèle sauvegardé : {model_file}")
        
        return {
            'model': str(model_file)
        }
    
    @staticmethod
    def load_model(
        model_path: str,
        device: str = 'cpu'
    ) -> Tuple[nn.Module, object]:
        """
        Charge le modèle
        
        Args:
            model_path: Chemin vers le fichier .pth
            device: 'cpu' ou 'cuda'
        
        Returns:
            model: Modèle PyTorch chargé
        """
        # Charger le checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        config = checkpoint['model_config']
        
        # Recréer le modèle selon la classe
        if config['class_name'] == 'EmotionClassifier':
            model = EmotionClassifier(
                input_dim=config['input_dim'],
                n_classes=config['n_classes'],
                hidden_dims=config['hidden_dims'],
                dropout_rate=config['dropout_rate'],
                use_batch_norm=config['use_batch_norm']
            )
        elif config['class_name'] == 'EmotionClassifierResNet':
            model = EmotionClassifierResNet(
                input_dim=config['input_dim'],
                n_classes=config['n_classes'],
                hidden_dims=config['hidden_dims'],
                dropout_rate=config['dropout_rate']
            )
        else:
            raise ValueError(f"Classe de modèle inconnue : {config['class_name']}")
        
        # Charger les poids
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        print(f"✅ Modèle chargé : {model_path}")
        print(f"   Architecture : {config['class_name']}")
        print(f"   Paramètres   : {model.get_num_parameters():,}")
        
        return model


# ==================== FONCTION DE PRÉDICTION ====================

def predict_emotion(
    image_path: str,
    model: nn.Module,
    device: str = 'cpu'
) -> Dict:
    """
    Prédire l'émotion d'une image
    
    Args:
        image_path: Chemin vers l'image
        model: Modèle PyTorch
        device: 'cpu' ou 'cuda'
    
    Returns:
        Dict avec la prédiction et les probabilités
    """
    from classes.FaceLandmarkExtractor import FaceLandmarkExtractor
    import cv2
    
    emotion_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    
    # Charger l'image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Impossible de charger l'image : {image_path}")
    
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
            'error': 'Aucun visage détecté',
            'image_path': image_path
        }
    
    # Normaliser les landmarks et convertir en features
    normalized_landmarks = extractor.normalize_landmarks(landmarks)
    features = extractor.landmarks_to_features(normalized_landmarks)
    features = features.reshape(1, -1)

    # Convertir en tensor
    features_tensor = torch.from_numpy(features.astype(np.float32)).to(device)
    
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