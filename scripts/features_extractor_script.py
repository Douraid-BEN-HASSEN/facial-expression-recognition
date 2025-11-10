"""
Script d'extraction de features pour l'apprentissage supervisé :
Prend un dossier d'images organisé par classes d'émotions,
extrait les landmarks faciaux et génère un dataset au format optimisé pour PyTorch.
Structure attendue du dataset :
image_folder_path/
├── angry/
├── image1.jpg
├── image2.jpg
└── ...
├── disgust/
└── ...
├── fear/
├── happy/
├── neutral/
├── sad/
└── surprise/
Chaque image est traitée pour extraire les landmarks faciaux à l'aide de MediaPipe.
Les features extraites sont sauvegardées dans des fichiers .npy pour l'entraînement.
Le script génère également un fichier metadata.json contenant des informations sur le dataset.
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, Dict
from tqdm import tqdm
import json
import matplotlib.pyplot as plt

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from classes.FaceLandmarkExtractor import FaceLandmarkExtractor

from sklearn.model_selection import train_test_split

INPUT_IMAGE_DATASET_PATH = "./datasets/fer2013_images"
OUTPUT_FEATURES_DATASET_PATH = "./datasets/fer2013_features"

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
        plt.figure(figsize=(4, 4))
        plt.imshow(img[..., ::-1])
        plt.axis('off')
        if title:
            plt.title(title)
        plt.show()
    
    return img

def extract_features_from_image_folder(
    image_folder_path: str,
    emotion_mapping: Dict[str, int],
    emotion_names: list,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Fonction interne pour extraire les features depuis un dossier
    
    Returns:
        X: Array de features (N, 1434)
        y: Array de labels (N,)
        stats: Dictionnaire de statistiques
    """
    
    root_path = Path(image_folder_path)
    if not root_path.exists():
        raise FileNotFoundError(f"❌ Le dossier {image_folder_path} n'existe pas")

    # Extensions d'images
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    # Collecter les chemins d'images
    all_image_paths = []
    all_labels = []
    emotion_counts = {emotion: 0 for emotion in emotion_names}
    
    for folder_name, emotion_id in emotion_mapping.items():
        emotion_folder = root_path / folder_name
        
        if not emotion_folder.exists():
            print(f"⚠️  Dossier manquant : {folder_name}/")
            continue
        
        images = [
            img for img in emotion_folder.iterdir()
            if img.is_file() and img.suffix.lower() in image_extensions
        ]
        
        emotion_counts[emotion_names[emotion_id]] = len(images)
        
        for img_path in images:
            all_image_paths.append(img_path)
            all_labels.append(emotion_id)
        
        print(f"   {emotion_names[emotion_id]:12} : {len(images):5} images")
    
    total_images = len(all_image_paths)
    print(f"\n   📊 Total : {total_images} images")
    
    if total_images == 0:
        raise ValueError("❌ Aucune image trouvée")
    
    # Initialiser l'extracteur
    print(f"\n   🔄 Extraction des features avec MediaPipe...")
    extractor = FaceLandmarkExtractor(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    )
    
    # Extraire les features
    X_list = []
    y_list = []
    valid_count = 0
    invalid_count = 0
    
    for img_path, label in tqdm(
        zip(all_image_paths, all_labels),
        total=total_images,
        ncols=80
    ):
        try:
            # Charger l'image
            image = cv2.imread(str(img_path))
            
            if image is None:
                invalid_count += 1
                continue
            
            # Extraire landmarks
            landmarks = extractor.extract_landmarks(image)

            if landmarks is not None:
                # Normaliser les landmarks
                normalized_landmarks = extractor.normalize_face_part_landmarks(landmarks)
                
                # Convertir en features (1434 dimensions)
                features = extractor.landmarks_to_features(normalized_landmarks)
                X_list.append(features)
                y_list.append(label)
                valid_count += 1
            else:
                invalid_count += 1
        
        except Exception as e:
            invalid_count += 1
    
    # Fermer MediaPipe
    del extractor
    
    # Convertir en numpy arrays
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    
    valid_rate = (valid_count / total_images) * 100
    
    print(f"\n   ✅ Visages détectés     : {valid_count} ({valid_rate:.2f}%)")
    print(f"   ❌ Visages non détectés : {invalid_count} ({100-valid_rate:.2f}%)")
    
    # Statistiques par émotion
    print(f"\n   📊 Distribution des classes :")
    class_distribution = {}
    for emotion_id in range(7):
        count = np.sum(y == emotion_id)
        percentage = (count / len(y)) * 100 if len(y) > 0 else 0
        emotion_name = emotion_names[emotion_id]
        
        class_distribution[emotion_name] = {
            'id': emotion_id,
            'count': int(count),
            'total_in_folder': emotion_counts[emotion_name],
            'percentage': float(percentage),
            'detection_rate': float(count / emotion_counts[emotion_name] * 100) if emotion_counts[emotion_name] > 0 else 0
        }
        
        print(f"      {emotion_name:12} : {count:5} ({percentage:5.2f}%)")
    
    stats = {
        'n_total_images': total_images,
        'n_valid_samples': valid_count,
        'n_invalid_samples': invalid_count,
        'valid_rate_percent': float(valid_rate),
        'class_distribution': class_distribution
    }
    
    return X, y, stats

def extract_features(
    image_folder_path: str,
    output_dir: str = "./datasets/fer2013_features",
    random_state: int = 42,
    train_split: float = 0.7,
    val_split: float = 0.2,
    test_split: float = 0.1,
) -> Dict[str, str]:
    """
    Extrait les features de landmarks pour l'apprentissage supervisé
    
    Args:
        image_folder_path: Chemin vers le dossier contenant les images
        output_dir: Dossier de sortie pour les fichiers générés
        random_state: Seed pour reproductibilité
    
    Returns:
        Dict avec les chemins des fichiers créés
    
    Structure de sortie:
        output_dir/
        ├── X_train.npy      # Features d'entraînement (N, 1434)
        ├── y_train.npy      # Labels d'entraînement (N,)
        ├── X_val.npy        # Features de validation (N, 1434)
        ├── y_val.npy        # Labels de validation (N,)
        ├── X_test.npy       # Features de test (N, 1434)
        ├── y_test.npy       # Labels de test (N,)
        └── metadata.json    # Métadonnées du dataset
    """
    
    # Mapping des émotions
    emotion_mapping = {
        'angry': 0,
        'disgust': 1,
        'fear': 2,
        'happy': 3,
        'neutral': 4,
        'sad': 5,
        'surprise': 6,
    }
    
    emotion_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    
    print("="*80)
    print("🎭 EXTRACTION DES FEATURES")
    print("="*80)
    
    # Créer le dossier de sortie
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # ========== EXTRACTION DU DATASET D'ENTRAÎNEMENT ==========
    print(f"\n📁 Extraction depuis : {image_folder_path}\n")
    
    X_features, y_features, features_stats = extract_features_from_image_folder(
        image_folder_path, emotion_mapping=emotion_mapping, emotion_names=emotion_names
    )

    print(f"   ✅ FEATURES : {len(X_features)} samples")
    
    # ========== VÉRIFICATION VISUELLE AVANT SAVE ==========
    print(f"\n🔍 Vérification visuelle avant d'enregistrer...")
    sample_idx = np.random.randint(0, len(X_features))
    sample_before = X_features[sample_idx].copy()
    sample_label = y_features[sample_idx]
    
    print(f"   Échantillon #{sample_idx} - Label: {emotion_names[sample_label]}")
    print(f"   Features min: {sample_before.min():.4f}, max: {sample_before.max():.4f}")
    print(f"   Features mean: {sample_before.mean():.4f}, std: {sample_before.std():.4f}")
    
    visualize_landmarks(
        sample_before, 
        image_size=256, 
        title=f"BEFORE Save - {emotion_names[sample_label]}"
    )

    # ========== SPLIT TRAIN/VAL/TEST ==========
    print(f"\n✂️  Split Train/Val/Test ({train_split*100:.1f}%/{val_split*100:.1f}%/{test_split*100:.1f}%)")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_features, y_features,
        test_size=(1 - train_split),
        random_state=random_state,
        stratify=y_features
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=test_split / (val_split + test_split),
        random_state=random_state,
        stratify=y_temp
    )
    print(f"   ✅ Train : {X_train.shape}, Val : {X_val.shape}, Test : {X_test.shape}")

    # ========== SAUVEGARDE DES FICHIERS ==========
    print(f"\n💾 Sauvegarde des fichiers dans : {output_dir}")
    
    files_created = {}
    
    # Train
    np.save(output_path / "X_train.npy", X_train.astype(np.float32))
    np.save(output_path / "y_train.npy", y_train.astype(np.int64))
    files_created['X_train'] = str(output_path / "X_train.npy")
    files_created['y_train'] = str(output_path / "y_train.npy")
    
    # Validation
    np.save(output_path / "X_val.npy", X_val.astype(np.float32))
    np.save(output_path / "y_val.npy", y_val.astype(np.int64))
    files_created['X_val'] = str(output_path / "X_val.npy")
    files_created['y_val'] = str(output_path / "y_val.npy")
    
    # Test
    np.save(output_path / "X_test.npy", X_test.astype(np.float32))
    np.save(output_path / "y_test.npy", y_test.astype(np.int64))
    files_created['X_test'] = str(output_path / "X_test.npy")
    files_created['y_test'] = str(output_path / "y_test.npy")
    
    # Redistribution des dossiers
    n_total = len(X_train) + len(X_val) + len(X_test)
    train_split = len(X_train) / n_total
    val_split = len(X_val) / n_total
    test_split = len(X_test) / n_total

    # Métadonnées
    metadata = {
        'feature_dimension': X_train.shape[1],
        'n_classes': 7,
        'emotion_names': emotion_names,
        'emotion_mapping': emotion_mapping,
        'train_split': train_split,
        'val_split': val_split,
        'test_split': test_split,
        'random_state': random_state,
        'features_stats': features_stats,
        'shapes': {
            'X_train': X_train.shape,
            'y_train': y_train.shape,
            'X_val': X_val.shape,
            'y_val': y_val.shape,
            'X_test': X_test.shape,
            'y_test': y_test.shape,
        },
        'files': files_created
    }
    
    metadata_path = output_path / "metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        # Convertir les shapes en listes pour JSON
        metadata['shapes'] = {k: list(v) if v is not None else None for k, v in metadata['shapes'].items()}
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    files_created['metadata'] = str(metadata_path)
    
    print(f"   ✅ {len(files_created)} fichiers créés")
    
    # ========== VÉRIFICATION APRÈS SAUVEGARDE ==========
    print(f"\n🔍 Vérification des fichiers sauvegardés...")
    
    # Recharger les données
    X_train_loaded = np.load(output_path / "X_train.npy")
    y_train_loaded = np.load(output_path / "y_train.npy")
    
    # Vérifier du même échantillon aléatoire
    check_sample = X_train_loaded[sample_idx]
    check_label = y_train_loaded[sample_idx]
    
    print(f"   Échantillon rechargé #{sample_idx} - Label: {emotion_names[check_label]}")
    print(f"   Features shape: {check_sample.shape}")
    print(f"   Features min: {check_sample.min():.4f}, max: {check_sample.max():.4f}")
    print(f"   Features mean: {check_sample.mean():.4f}, std: {check_sample.std():.4f}")
    
    visualize_landmarks(
        check_sample, 
        image_size=256,
        title=f"SAVED - {emotion_names[check_label]}"
    )
    
    # ========== RÉSUMÉ FINAL ==========
    print("\n" + "="*80)
    print("✅ EXTRACTION TERMINÉE")
    print("="*80)
    print(f"\n📊 Résumé du dataset :")
    print(f"   Train      : {X_train.shape}")
    print(f"   Validation : {X_val.shape}")
    if X_test is not None:
        print(f"   Test       : {X_test.shape}")
    print(f"   Features   : {X_train.shape[1]} dimensions")
    print(f"   Classes    : {len(emotion_names)}")
    
    print(f"\n💡 Pour charger les données :")
    print(f"   X_train = np.load('{files_created['X_train']}')")
    print(f"   y_train = np.load('{files_created['y_train']}')")
    
    return files_created

if __name__ == "__main__":
    # Extraire les features du dossier d'images
    files = extract_features(
        image_folder_path=INPUT_IMAGE_DATASET_PATH,
        output_dir=OUTPUT_FEATURES_DATASET_PATH,
        random_state=42
    )
    
    print("\n" + "="*80)
    print("📦 Fichiers créés :")
    for key, path in files.items():
        print(f"   {key:12} : {path}")
    print("="*80)
