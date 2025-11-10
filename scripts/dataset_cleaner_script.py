# Script for Dataset Cleaning
import cv2
from pathlib import Path
from typing import Tuple
from tqdm import tqdm
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from classes.FaceLandmarkExtractor import FaceLandmarkExtractor

DATASET_PATH = "./datasets/fer2013_images"
THRESHOLD = 0.5 # Confidence threshold for face detection

def get_invalid_images_from_dataset(
    dataset_root: str,
    threshold: float = 0.5
) -> Tuple[str]:
    """
    Extrait les features de landmarks depuis un dataset organisé en sous-dossiers par émotion
    et retourne les images où aucun visage n'a été détecté.
    
    Structure attendue du dataset :
    dataset_root/
    ├── angry/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    ├── disgust/
    │   └── ...
    ├── fear/
    ├── happy/
    ├── neutral/
    ├── sad/
    └── surprise/
    
    Args:
        dataset_root: Chemin vers le dossier racine contenant les sous-dossiers d'émotions
    
    Returns:
        Tuple[str]: Chemins vers les images invalides
    """

    # Vérifier que le dossier racine existe
    dataset_path = Path(dataset_root)
    if not dataset_path.exists():
        raise FileNotFoundError(f"❌ Le dossier {dataset_root} n'existe pas")
    
    print("="*70)
    print("🎭 EXTRACTION DES FEATURES - FER2013")
    print("="*70)
    print(f"📁 Dataset racine : {dataset_root}\n")
    
    # Collecter tous les fichiers images par émotion
    all_image_paths = []
    
    # Extensions d'images supportées
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    # Parcourir chaque dossier d'émotion
    for folder_name in os.listdir(dataset_path):
        emotion_folder = dataset_path / folder_name
        
        # Trouver toutes les images dans ce dossier
        images = [
            img for img in emotion_folder.iterdir() 
            if img.is_file() and img.suffix.lower() in image_extensions
        ]
        
        for img_path in images:
            all_image_paths.append(img_path)
        
        print(f"✅ {folder_name:12} : {len(images):5} images")
    
    total_images = len(all_image_paths)
    
    if total_images == 0:
        raise ValueError("❌ Aucune image trouvée dans les sous-dossiers")
    
    print(f"\n📊 Total images      : {total_images}")
    
    # Initialiser l'extracteur de landmarks
    print("\n🔄 Initialisation de MediaPipe Face Mesh...")
    extractor = FaceLandmarkExtractor(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=threshold
    )
    
    # Extraire les features pour toutes les images
    print(f"\n🔄 Extraction des features sur {total_images} images...\n")
    
    features_list = []
    valid_count = 0
    invalid_count = 0
    invalid_paths = []
    
    for img_path in tqdm(all_image_paths, desc="Extraction", ncols=100):
        try:
            # Charger l'image
            image = cv2.imread(str(img_path))
            
            if image is None:
                print(f"⚠️  Impossible de charger : {img_path}")
                features_list.append(None)
                invalid_count += 1
                invalid_paths.append(str(img_path))
                continue
            
            # Extraire les landmarks
            landmarks = extractor.extract_landmarks(image)
            
            if landmarks is not None:
                # Extraire les features
                features = extractor.landmarks_to_features(landmarks)
                features_list.append(features)
                valid_count += 1
            else:
                features_list.append(None)
                invalid_count += 1
                invalid_paths.append(str(img_path))
        
        except Exception as e:
            print(f"❌ Erreur sur {img_path}: {e}")
            features_list.append(None)
            invalid_count += 1
            invalid_paths.append(str(img_path))
    
    # Fermer MediaPipe proprement
    del extractor
    
    # Statistiques
    valid_rate = (valid_count / total_images) * 100
    
    print("\n" + "="*70)
    print("📊 RÉSULTATS DE L'EXTRACTION")
    print("="*70)
    print(f"✅ Visages détectés      : {valid_count} ({valid_rate:.2f}%)")
    print(f"❌ Visages non détectés  : {invalid_count} ({100-valid_rate:.2f}%)")
    
    return invalid_paths

def delete_invalid_images(invalid_image_paths: Tuple[str]):
    """
    Supprime les images invalides du dataset.
    
    Args:
        invalid_image_paths: Chemins vers les images invalides
    """
    print("\n🗑️  Suppression des images invalides...")
    for img_path in invalid_image_paths:
        try:
            os.remove(img_path)
            print(f"✅ Supprimé : {img_path}")
        except Exception as e:
            print(f"❌ Erreur lors de la suppression de {img_path}: {e}")

# check if DATASET_PATH exists
if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"The specified dataset path '{DATASET_PATH}' does not exist.")

if __name__ == "__main__":
    invalid_paths = get_invalid_images_from_dataset(
        dataset_root=DATASET_PATH,
        threshold=THRESHOLD
    )
    
    # Delete invalid images
    delete_invalid_images(invalid_image_paths=invalid_paths)

    print(f"✅ Cleaning done. {len(invalid_paths)} images removed.")

