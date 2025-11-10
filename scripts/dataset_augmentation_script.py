# Script for Dataset Augmentation
from pathlib import Path
import os
import albumentations as A
import numpy as np
from PIL import Image

DATASET_PATH = "./datasets/fer2013_cleaned_augmented_v2/train"
MINIMUM_IMAGES_PER_CLASS = 7500

def augment_dataset(
    dataset_root: str
):
    """
    Augmente le dataset pour s'assurer que chaque classe d'émotion
    contient au moins MINIMUM_IMAGES_PER_CLASS images.
    
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
    """

    # Vérifier que le dossier racine existe
    dataset_path = Path(dataset_root)
    if not dataset_path.exists():
        raise FileNotFoundError(f"❌ Le dossier {dataset_root} n'existe pas")
    
    print("="*70)
    print("🎭 EXTRACTION DES FEATURES - FER2013")
    print("="*70)
    print(f"📁 Dataset racine : {dataset_root}\n")

    # Extensions d'images supportées
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    # Albumentations pour les augmentations
    augment_normal = A.Compose([
        A.Rotate(limit=[-10.0, 10.0], interpolation=1, border_mode=0, fill=0, fill_mask=0, p=1.0),
        A.HorizontalFlip(p=1.0),
        A.RandomScale(scale_limit=[-0.1, 0.1], interpolation=1, p=1.0),
    ])
    
    # Parcourir chaque dossier d'émotion
    for folder_name in os.listdir(dataset_path):
        emotion_folder = dataset_path / folder_name
        
        # Get list of image files in the emotion folder
        images = [
            img for img in emotion_folder.iterdir() 
            if img.is_file() and img.suffix.lower() in image_extensions
        ]
        
        if len(images) < MINIMUM_IMAGES_PER_CLASS:
            print(f"\n🔄 Augmentation pour la classe '{folder_name}' (images actuelles : {len(images)})")
            created_images = 0
            while len(images) + created_images < MINIMUM_IMAGES_PER_CLASS:
                image_path = np.random.choice(images) # get random image
                image = np.array(Image.open(image_path))
                augmented_image = augment_normal(image=image)["image"]
                output_image = Image.fromarray(augmented_image)
                output_image.save(os.path.join(emotion_folder, f'augmented_{created_images}.png'))
                created_images += 1
            
            print(f"✅ Augmentation terminée pour la classe '{folder_name}'. Images créées : {created_images}")
        
if __name__ == "__main__":
    augment_dataset(
        dataset_root=DATASET_PATH
    )
    print(f"\n🎉 Dataset augmentation done !")


