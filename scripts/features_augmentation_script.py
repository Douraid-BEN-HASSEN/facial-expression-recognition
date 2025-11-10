"""
Script d'augmentation de features pour les landmarks faciaux normalisés
Applique des transformations géométriques qui préservent la structure des landmarks
"""

import numpy as np
import cv2
from pathlib import Path
import json
from typing import Tuple, Dict
from tqdm import tqdm

# Configuration
INPUT_FEATURES_DIR = "./datasets/fer2013_features"
OUTPUT_FEATURES_DIR = "./datasets/fer2013_features_augmented_v2"
AUGMENT_BY = 7.5  # Facteur d'augmentation global

class LandmarkAugmenter:
    @staticmethod
    def visualize_landmarks(features: np.ndarray, image_size: int = 256, color: tuple = (0, 255, 0), show: bool = True, ax=None, title=None):
        """
        Affiche les landmarks à partir d'un vecteur de features (1434,)
        Args:
            features: Vecteur de features (1434,)
            image_size: Taille de l'image carrée de visualisation
            color: Couleur des points (BGR)
            show: Affiche la figure si True
            ax: Matplotlib axis (optionnel)
            title: Titre de la figure (optionnel)
        """
        import matplotlib.pyplot as plt
        landmarks = features.reshape(-1, 3)
        img = np.zeros((image_size, image_size, 3), dtype=np.uint8)
        xs = (landmarks[:, 0] * image_size).astype(int)
        ys = (landmarks[:, 1] * image_size).astype(int)
        for x, y in zip(xs, ys):
            if 0 <= x < image_size and 0 <= y < image_size:
                img = cv2.circle(img, (x, y), 1, color, -1)
        if ax is not None:
            ax.imshow(img[..., ::-1])
            ax.axis('off')
            if title:
                ax.set_title(title)
        elif show:
            plt.figure(figsize=(3, 3))
            plt.imshow(img[..., ::-1])
            plt.axis('off')
            if title:
                plt.title(title)
            plt.show()
    """
    Classe pour augmenter les features de landmarks faciaux
    Les transformations préservent la cohérence géométrique des points
    """
    
    def __init__(self, feature_dim: int = 1434):
        """
        Args:
            feature_dim: Dimension des features (1434 = 478 landmarks × 3 coordonnées)
        """
        self.feature_dim = feature_dim
        self.n_landmarks = feature_dim // 3
    
    def reshape_to_landmarks(self, features: np.ndarray) -> np.ndarray:
        """
        Convertit un vecteur de features (1434,) en array de landmarks (478, 3)
        
        Args:
            features: Vecteur de features (1434,)
        
        Returns:
            Array de landmarks (478, 3) avec colonnes [x, y, z]
        """
        return features.reshape(self.n_landmarks, 3)
    
    def flatten_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Convertit un array de landmarks (478, 3) en vecteur de features (1434,)
        
        Args:
            landmarks: Array de landmarks (478, 3)
        
        Returns:
            Vecteur de features (1434,)
        """
        return landmarks.flatten()
    
    def horizontal_flip(self, features: np.ndarray) -> np.ndarray:
        """
        Applique un flip horizontal en inversant les coordonnées X
        et en échangeant les landmarks gauche/droite
        
        Args:
            features: Vecteur de features (1434,)
        
        Returns:
            Features flippées (1434,)
        """
        landmarks = self.reshape_to_landmarks(features.copy())
        
        # Inverser les coordonnées X (landmarks normalisés entre 0 et 1)
        landmarks[:, 0] = 1.0 - landmarks[:, 0]
        
        # Échanger les landmarks gauche/droite pour préserver la cohérence anatomique
        # Liste des paires de landmarks à échanger (gauche <-> droite)
        # Source: MediaPipe Face Mesh topology
        swap_pairs = [
            # Yeux
            (33, 263), (7, 163), (163, 144), (144, 145), (145, 153), (153, 154), (154, 155), (155, 133),
            (33, 246), (160, 385), (158, 387), (133, 362), (173, 398), (157, 384), (158, 387), (159, 388),
            (160, 466), (161, 388), (246, 33),
            # Sourcils
            (46, 53), (53, 52), (52, 65), (65, 55), (70, 63), (63, 105), (105, 66), (66, 107),
            (336, 296), (296, 334), (334, 293), (293, 300), (276, 283), (283, 282), (282, 295), (295, 285),
            # Nez (quelques points latéraux)
            (98, 327), (97, 326), (2, 2),  # Le point 2 reste central
            # Bouche
            (61, 291), (185, 40), (40, 39), (39, 37), (0, 267), (267, 269), (269, 270), (270, 409),
            (409, 291), (375, 321), (321, 405), (314, 17), (17, 84), (181, 91), (146, 61),
            (91, 181), (146, 375), (61, 291), (185, 40), (40, 39), (39, 37),
            # Mâchoire
            (127, 356), (34, 264), (139, 368), (162, 389), (21, 251), (54, 284), (103, 332), (67, 297), (109, 338),
        ]
        
        # Créer un array temporaire pour stocker les landmarks échangés
        landmarks_flipped = landmarks.copy()
        
        # Échanger les paires
        for left_idx, right_idx in swap_pairs:
            if left_idx < self.n_landmarks and right_idx < self.n_landmarks:
                landmarks_flipped[left_idx], landmarks_flipped[right_idx] = \
                    landmarks[right_idx].copy(), landmarks[left_idx].copy()
        
        return self.flatten_landmarks(landmarks_flipped)
    
    def add_gaussian_noise(self, features: np.ndarray, noise_scale: float = 0.01) -> np.ndarray:
        """
        Ajoute un bruit gaussien léger aux coordonnées des landmarks
        
        Args:
            features: Vecteur de features (1434,)
            noise_scale: Écart-type du bruit (recommandé: 0.005-0.02)
        
        Returns:
            Features bruitées (1434,)
        """
        landmarks = self.reshape_to_landmarks(features.copy())
        
        # Ajouter du bruit gaussien seulement sur x et y (pas z)
        noise_xy = np.random.normal(0, noise_scale, size=(self.n_landmarks, 2))
        landmarks[:, :2] += noise_xy
        
        # Clipper les valeurs pour rester dans [0, 1] (landmarks normalisés)
        landmarks[:, :2] = np.clip(landmarks[:, :2], 0.0, 1.0)
        
        return self.flatten_landmarks(landmarks)
    
    def scale_transform(self, features: np.ndarray, scale_range: Tuple[float, float] = (0.95, 1.05)) -> np.ndarray:
        """
        Applique une transformation de mise à l'échelle autour du centre
        
        Args:
            features: Vecteur de features (1434,)
            scale_range: Plage de facteurs d'échelle (min, max)
        
        Returns:
            Features mises à l'échelle (1434,)
        """
        landmarks = self.reshape_to_landmarks(features.copy())
        
        # Calculer le centre des landmarks
        center = np.mean(landmarks[:, :2], axis=0)
        
        # Générer un facteur d'échelle aléatoire
        scale_factor = np.random.uniform(scale_range[0], scale_range[1])
        
        # Centrer, mettre à l'échelle, puis recentrer
        landmarks[:, :2] = (landmarks[:, :2] - center) * scale_factor + center
        
        # Clipper pour rester dans [0, 1]
        landmarks[:, :2] = np.clip(landmarks[:, :2], 0.0, 1.0)
        
        return self.flatten_landmarks(landmarks)
    
    def rotate_transform(self, features: np.ndarray, angle_range: Tuple[float, float] = (-5, 5)) -> np.ndarray:
        """
        Applique une rotation légère autour du centre des landmarks
        
        Args:
            features: Vecteur de features (1434,)
            angle_range: Plage d'angles en degrés (min, max)
        
        Returns:
            Features pivotées (1434,)
        """
        landmarks = self.reshape_to_landmarks(features.copy())
        
        # Calculer le centre des landmarks
        center = np.mean(landmarks[:, :2], axis=0)
        
        # Générer un angle aléatoire et le convertir en radians
        angle_deg = np.random.uniform(angle_range[0], angle_range[1])
        angle_rad = np.radians(angle_deg)
        
        # Matrice de rotation
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        rotation_matrix = np.array([
            [cos_a, -sin_a],
            [sin_a, cos_a]
        ])
        
        # Centrer, pivoter, puis recentrer
        centered = landmarks[:, :2] - center
        rotated = centered @ rotation_matrix.T
        landmarks[:, :2] = rotated + center
        
        # Clipper pour rester dans [0, 1]
        landmarks[:, :2] = np.clip(landmarks[:, :2], 0.0, 1.0)
        
        return self.flatten_landmarks(landmarks)
    
    def translate_transform(self, features: np.ndarray, translate_range: float = 0.02) -> np.ndarray:
        """
        Applique une légère translation aléatoire
        
        Args:
            features: Vecteur de features (1434,)
            translate_range: Amplitude maximale de la translation
        
        Returns:
            Features translatées (1434,)
        """
        landmarks = self.reshape_to_landmarks(features.copy())
        
        # Générer une translation aléatoire
        tx = np.random.uniform(-translate_range, translate_range)
        ty = np.random.uniform(-translate_range, translate_range)
        
        # Appliquer la translation
        landmarks[:, 0] += tx
        landmarks[:, 1] += ty
        
        # Clipper pour rester dans [0, 1]
        landmarks[:, :2] = np.clip(landmarks[:, :2], 0.0, 1.0)
        
        return self.flatten_landmarks(landmarks)
    
    def augment_sample(self, features: np.ndarray, augmentation_type: str = 'random') -> np.ndarray:
        """
        Applique une augmentation aléatoire ou spécifique à un échantillon
        
        Args:
            features: Vecteur de features (1434,)
            augmentation_type: Type d'augmentation
                - 'random': Choisit aléatoirement parmi toutes les transformations
                - 'flip': Flip horizontal
                - 'noise': Bruit gaussien
                - 'scale': Mise à l'échelle
                - 'rotate': Rotation
                - 'translate': Translation
                - 'combined': Combine plusieurs transformations
        
        Returns:
            Features augmentées (1434,)
        """
        if augmentation_type == 'random':
            # augmentation_type = np.random.choice([
            #     'flip', 'noise', 'scale', 'rotate', 'translate', 'combined'
            # ])
            augmentation_type = np.random.choice([
                'noise', 'scale', 'rotate', 'translate', 'combined'
            ])
        # if augmentation_type == 'flip':
        #     return self.horizontal_flip(features)
        
        noise_scale = np.random.uniform(0.005, 0.008)
        translate_range = np.random.uniform(0.01, 0.08)
        angle_range = (-6, 6)
        scale_range = (0.92, 0.98)
        if augmentation_type == 'noise':
            return self.add_gaussian_noise(features, noise_scale=noise_scale)

        elif augmentation_type == 'scale':
            return self.scale_transform(features, scale_range=scale_range)
        
        elif augmentation_type == 'rotate':
            return self.rotate_transform(features, angle_range=angle_range)
        
        elif augmentation_type == 'translate':
            return self.translate_transform(features, translate_range=translate_range)
        
        elif augmentation_type == 'combined':
            # Appliquer plusieurs transformations en séquence
            augmented = features.copy()
            
            augmentation_added = False
            while not augmentation_added:
                # Probabilité d'appliquer chaque transformation
                if np.random.rand() > 0.5:
                    augmented = self.scale_transform(augmented, scale_range=scale_range)
                    augmentation_added = True

                if np.random.rand() > 0.5:
                    augmented = self.rotate_transform(augmented, angle_range=angle_range)
                    augmentation_added = True

                if np.random.rand() > 0.5:
                    augmented = self.translate_transform(augmented, translate_range=translate_range)
                    augmentation_added = True

                if np.random.rand() > 0.5:
                    augmented = self.add_gaussian_noise(augmented, noise_scale=noise_scale)
                    augmentation_added = True

            return augmented
        
        else:
            raise ValueError(f"Type d'augmentation inconnu : {augmentation_type}")

def augment_features_dataset(
    input_dir: str,
    output_dir: str,
    augment_by: float = 1.5,
    random_state: int = 42
) -> Dict[str, str]:
    """
    Augmente un dataset de features pour équilibrer les classes
    
    Args:
        input_dir: Dossier contenant les features originales
        output_dir: Dossier de sortie pour les features augmentées
        target_samples_per_class: Nombre cible d'échantillons par classe
        random_state: Seed pour reproductibilité
    
    Returns:
        Dict avec les chemins des fichiers créés
    """
    
    np.random.seed(random_state)
    
    print("="*80)
    print("🎭 AUGMENTATION DE FEATURES - LANDMARKS FACIAUX")
    print("="*80)
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # ========== CHARGEMENT DES MÉTADONNÉES ==========
    metadata_file = input_path / "metadata.json"
    if not metadata_file.exists():
        raise FileNotFoundError(f"❌ Fichier metadata.json introuvable dans {input_dir}")
    
    with open(metadata_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    print(f"\n📁 Dataset source : {input_dir}")
    print(f"📁 Dataset cible  : {output_dir}")
    print(f"🎯 Augmentation par classe : {augment_by}x\n")
    
    emotion_names = metadata['emotion_names']
    n_classes = metadata['n_classes']
    feature_dim = metadata['feature_dimension']
    
    # ========== CHARGEMENT DES DONNÉES ==========
    print("📥 Chargement des données originales...")
    
    X_train = np.load(input_path / "X_train.npy")
    y_train = np.load(input_path / "y_train.npy")
    X_val = np.load(input_path / "X_val.npy")
    y_val = np.load(input_path / "y_val.npy")
    X_test = np.load(input_path / "X_test.npy")
    y_test = np.load(input_path / "y_test.npy")
    
    print(f"   ✅ Train : {X_train.shape}")
    print(f"   ✅ Val   : {X_val.shape}")
    print(f"   ✅ Test  : {X_test.shape}")
    
    # ========== ANALYSE DE LA DISTRIBUTION ==========
    print(f"\n📊 Distribution actuelle des classes (Train) :")
    class_counts = {}
    for class_id in range(n_classes):
        count = np.sum(y_train == class_id)
        class_counts[class_id] = count
        emotion_name = emotion_names[class_id]
        print(f"   {emotion_name:12} : {count:5} échantillons")

    print(f"\n📊 Distribution actuelle des classes (Validation) :")
    for class_id in range(n_classes):
        count = np.sum(y_val == class_id)
        emotion_name = emotion_names[class_id]
        print(f"   {emotion_name:12} : {count:5} échantillons")
    
    print(f"\n📊 Distribution actuelle des classes (Test) :")
    for class_id in range(n_classes):
        count = np.sum(y_test == class_id)
        emotion_name = emotion_names[class_id]
        print(f"   {emotion_name:12} : {count:5} échantillons")
    
    # ========== AUGMENTATION ==========
    print(f"\n🔄 Augmentation des classes sous-représentées...")
    
    augmenter = LandmarkAugmenter(feature_dim=feature_dim)
    
    total_augmentations = 0

    # Augmentation Train
    X_train_augmented = []
    y_train_augmented = []
    
    for class_id in range(n_classes):
        emotion_name = emotion_names[class_id]
        
        # Récupérer les échantillons de cette classe
        class_mask = (y_train == class_id)
        X_class = X_train[class_mask]
        
        current_count = len(X_class)
        needed = int(augment_by * current_count) - current_count
        
        # Ajouter les échantillons originaux
        X_train_augmented.append(X_class)
        y_train_augmented.append(np.full(current_count, class_id))
        
        if needed > 0:
            print(f"\n   🔄 {emotion_name} : génération de {needed} échantillons supplémentaires")
            
            # Générer des échantillons augmentés
            augmented_samples = []
            
            for i in tqdm(range(needed), desc=f"      {emotion_name}", ncols=80):
                # Choisir un échantillon source aléatoire
                source_idx = np.random.randint(0, current_count)
                source_features = X_class[source_idx]
                
                # Appliquer une augmentation aléatoire
                augmented = augmenter.augment_sample(source_features, augmentation_type='random')
                augmented_samples.append(augmented)
            
            augmented_samples = np.array(augmented_samples)
            X_train_augmented.append(augmented_samples)
            y_train_augmented.append(np.full(needed, class_id))
            
            total_augmentations += needed
            print(f"      ✅ {needed} échantillons créés")
            
            # Visualiser un échantillon augmenté
            if len(augmented_samples) > 0:
                import matplotlib.pyplot as plt
                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                
                # Original
                LandmarkAugmenter.visualize_landmarks(
                    X_class[source_idx], 
                    show=False, 
                    ax=axes[0], 
                    title=f"{emotion_name} - Original"
                )
                
                # Augmenté (dernier échantillon créé)
                LandmarkAugmenter.visualize_landmarks(
                    augmented_samples[-1], 
                    show=False, 
                    ax=axes[1], 
                    title=f"{emotion_name} - Augmenté"
                )
                
                # Un autre augmenté aléatoire
                random_idx = np.random.randint(0, len(augmented_samples))
                LandmarkAugmenter.visualize_landmarks(
                    augmented_samples[random_idx], 
                    show=False, 
                    ax=axes[2], 
                    title=f"{emotion_name} - Augmenté (aléatoire)"
                )
                
                plt.tight_layout()
                plt.show()
        else:
            print(f"   ✅ {emotion_name} : déjà au-dessus de la cible ({current_count} échantillons)")
    
    # Augmentation Validation
    X_val_augmented = []
    y_val_augmented = []

    for class_id in range(n_classes):
        emotion_name = emotion_names[class_id]
        
        # Récupérer les échantillons de cette classe
        class_mask = (y_val == class_id)
        X_class = X_val[class_mask]
        
        current_count = len(X_class)
        needed = int(augment_by * current_count) - current_count
        
        # Ajouter les échantillons originaux
        X_val_augmented.append(X_class)
        y_val_augmented.append(np.full(current_count, class_id))
        
        if needed > 0:
            print(f"\n   🔄 {emotion_name} : génération de {needed} échantillons supplémentaires")
            
            # Générer des échantillons augmentés
            augmented_samples = []
            
            for i in tqdm(range(needed), desc=f"      {emotion_name}", ncols=80):
                # Choisir un échantillon source aléatoire
                source_idx = np.random.randint(0, current_count)
                source_features = X_class[source_idx]
                
                # Appliquer une augmentation aléatoire
                augmented = augmenter.augment_sample(source_features, augmentation_type='random')
                augmented_samples.append(augmented)
            
            augmented_samples = np.array(augmented_samples)
            X_val_augmented.append(augmented_samples)
            y_val_augmented.append(np.full(needed, class_id))
            
            total_augmentations += needed
            print(f"      ✅ {needed} échantillons créés")
            
            # Visualiser un échantillon augmenté
            if len(augmented_samples) > 0:
                import matplotlib.pyplot as plt
                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                
                # Original
                LandmarkAugmenter.visualize_landmarks(
                    X_class[source_idx], 
                    show=False, 
                    ax=axes[0], 
                    title=f"{emotion_name} - Original"
                )
                
                # Augmenté (dernier échantillon créé)
                LandmarkAugmenter.visualize_landmarks(
                    augmented_samples[-1], 
                    show=False, 
                    ax=axes[1], 
                    title=f"{emotion_name} - Augmenté"
                )
                
                # Un autre augmenté aléatoire
                random_idx = np.random.randint(0, len(augmented_samples))
                LandmarkAugmenter.visualize_landmarks(
                    augmented_samples[random_idx], 
                    show=False, 
                    ax=axes[2], 
                    title=f"{emotion_name} - Augmenté (aléatoire)"
                )
                
                plt.tight_layout()
                plt.show()
        else:
            print(f"   ✅ {emotion_name} : déjà au-dessus de la cible ({current_count} échantillons)")

    # Augmentation Test
    X_test_augmented = []
    y_test_augmented = []

    for class_id in range(n_classes):
        emotion_name = emotion_names[class_id]
        
        # Récupérer les échantillons de cette classe
        class_mask = (y_test == class_id)
        X_class = X_test[class_mask]
        
        current_count = len(X_class)
        needed = int(augment_by * current_count) - current_count
        
        # Ajouter les échantillons originaux
        X_test_augmented.append(X_class)
        y_test_augmented.append(np.full(current_count, class_id))
        
        if needed > 0:
            print(f"\n   🔄 {emotion_name} : génération de {needed} échantillons supplémentaires")
            
            # Générer des échantillons augmentés
            augmented_samples = []
            
            for i in tqdm(range(needed), desc=f"      {emotion_name}", ncols=80):
                # Choisir un échantillon source aléatoire
                source_idx = np.random.randint(0, current_count)
                source_features = X_class[source_idx]
                
                # Appliquer une augmentation aléatoire
                augmented = augmenter.augment_sample(source_features, augmentation_type='random')
                augmented_samples.append(augmented)
            
            augmented_samples = np.array(augmented_samples)
            X_test_augmented.append(augmented_samples)
            y_test_augmented.append(np.full(needed, class_id))
            
            total_augmentations += needed
            print(f"      ✅ {needed} échantillons créés")
            
            # Visualiser un échantillon augmenté
            if len(augmented_samples) > 0:
                import matplotlib.pyplot as plt
                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                
                # Original
                LandmarkAugmenter.visualize_landmarks(
                    X_class[source_idx], 
                    show=False, 
                    ax=axes[0], 
                    title=f"{emotion_name} - Original"
                )
                
                # Augmenté (dernier échantillon créé)
                LandmarkAugmenter.visualize_landmarks(
                    augmented_samples[-1], 
                    show=False, 
                    ax=axes[1], 
                    title=f"{emotion_name} - Augmenté"
                )
                
                # Un autre augmenté aléatoire
                random_idx = np.random.randint(0, len(augmented_samples))
                LandmarkAugmenter.visualize_landmarks(
                    augmented_samples[random_idx], 
                    show=False, 
                    ax=axes[2], 
                    title=f"{emotion_name} - Augmenté (aléatoire)"
                )
                
                plt.tight_layout()
                plt.show()
        else:
            print(f"   ✅ {emotion_name} : déjà au-dessus de la cible ({current_count} échantillons)")

    # Concaténer tous les échantillons
    X_train_final = np.concatenate(X_train_augmented, axis=0)
    y_train_final = np.concatenate(y_train_augmented, axis=0)
    X_val_final = np.concatenate(X_val_augmented, axis=0)
    y_val_final = np.concatenate(y_val_augmented, axis=0)
    X_test_final = np.concatenate(X_test_augmented, axis=0)
    y_test_final = np.concatenate(y_test_augmented, axis=0)
    
    # Mélanger les échantillons
    print(f"\n🔀 Mélange du dataset augmenté...")
    shuffle_idx = np.random.permutation(len(X_train_final))
    X_train_final = X_train_final[shuffle_idx]
    y_train_final = y_train_final[shuffle_idx]

    shuffle_idx = np.random.permutation(len(X_val_final))
    X_val_final = X_val_final[shuffle_idx]
    y_val_final = y_val_final[shuffle_idx]

    shuffle_idx = np.random.permutation(len(X_test_final))
    X_test_final = X_test_final[shuffle_idx]
    y_test_final = y_test_final[shuffle_idx]
    
    print(f"   ✅ Dataset mélangé")
    
    # ========== SAUVEGARDE ==========
    print(f"\n💾 Sauvegarde des données augmentées...")
    
    files_created = {}
    
    # Train augmenté
    np.save(output_path / "X_train.npy", X_train_final.astype(np.float32))
    np.save(output_path / "y_train.npy", y_train_final.astype(np.int64))
    files_created['X_train'] = str(output_path / "X_train.npy")
    files_created['y_train'] = str(output_path / "y_train.npy")
    
    # Validation augmentée
    np.save(output_path / "X_val.npy", X_val_final.astype(np.float32))
    np.save(output_path / "y_val.npy", y_val_final.astype(np.int64))
    files_created['X_val'] = str(output_path / "X_val.npy")
    files_created['y_val'] = str(output_path / "y_val.npy")

    # Test augmenté
    np.save(output_path / "X_test.npy", X_test_final.astype(np.float32))
    np.save(output_path / "y_test.npy", y_test_final.astype(np.int64))
    files_created['X_test'] = str(output_path / "X_test.npy")
    files_created['y_test'] = str(output_path / "y_test.npy")
    
    # ========== MÉTADONNÉES ==========
    print(f"\n📊 Distribution finale des classes :")
    features_stats = {}
    for class_id in range(n_classes):
        count = np.sum(y_train_final == class_id)
        count += np.sum(y_val_final == class_id)
        count += np.sum(y_test_final == class_id)
        emotion_name = emotion_names[class_id]
        percentage = (count / (len(y_train_final) + len(y_val_final) + len(y_test_final))) * 100
        
        features_stats[emotion_name] = {
            'id': class_id,
            'count': int(count),
            'percentage': float(percentage)
        }
        
        print(f"   {emotion_name:12} : {count:5} échantillons ({percentage:5.2f}%)")
    
    # Mettre à jour les métadonnées
    metadata_augmented = metadata.copy()
    metadata_augmented['augmented'] = True
    metadata_augmented['total_augmentations'] = total_augmentations
    metadata_augmented['shapes'] = {
        'X_train': list(X_train_final.shape),
        'y_train': list(y_train_final.shape),
        'X_val': list(X_val_final.shape),
        'y_val': list(y_val_final.shape),
        'X_test': list(X_test_final.shape),
        'y_test': list(y_test_final.shape),
    }
    metadata_augmented['features_stats'] = features_stats
    metadata_augmented['files'] = files_created
    
    metadata_path = output_path / "metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata_augmented, f, indent=2, ensure_ascii=False)
    
    files_created['metadata'] = str(metadata_path)
    
    print(f"   ✅ {len(files_created)} fichiers créés")
    
    # ========== RÉSUMÉ FINAL ==========
    print("\n" + "="*80)
    print("✅ AUGMENTATION TERMINÉE")
    print("="*80)
    print(f"\n📊 Résumé :")
    print(f"   Échantillons originaux : {len(X_train)}")
    print(f"   Échantillons augmentés : {total_augmentations}")
    print(f"   Total final            : {len(X_train_final)}")
    print(f"   Validation (inchangé)  : {len(X_val)}")
    print(f"\n💡 Pour charger les données augmentées :")
    print(f"   X_train = np.load('{files_created['X_train']}')")
    print(f"   y_train = np.load('{files_created['y_train']}')")
    
    return files_created

if __name__ == "__main__":
    # Augmenter le dataset de features
    files = augment_features_dataset(
        input_dir=INPUT_FEATURES_DIR,
        output_dir=OUTPUT_FEATURES_DIR,
        augment_by=AUGMENT_BY,
        random_state=42
    )
    
    print("\n" + "="*80)
    print("📦 Fichiers créés :")
    for key, path in files.items():
        print(f"   {key:12} : {path}")
    print("="*80)
