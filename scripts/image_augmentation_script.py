# Script for image augmentation

from pathlib import Path
import os
import albumentations as A
import numpy as np
from PIL import Image

def augment_image_folder(
    image_folder_path: str,
    minimum_images: int
):
    """
    Augmente le dossier d'images pour s'assurer qu'il
    contient au moins MINIMUM_IMAGES images.
    
    Structure attendue du dataset :
    image_folder_path/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    
    Args:
        image_folder_path: Chemin vers le dossier contenant les images à augmenter
    """

    # Vérifier que le dossier d'images existe
    image_folder_path = Path(image_folder_path)
    if not image_folder_path.exists():
        raise FileNotFoundError(f"❌ Le dossier {image_folder_path} n'existe pas")
    
    print("="*70)
    print("🎭 IMAGE AUGMENTATION")
    print("="*70)
    print(f"📁 Dossier : {image_folder_path}\n")

    # Extensions d'images supportées
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    # Albumentations pour les augmentations
    augment_normal = A.Compose([
        A.Rotate(limit=[-10.0, 10.0], interpolation=1, border_mode=0, fill=0, fill_mask=0, p=1.0),
        A.HorizontalFlip(p=1.0),
        A.RandomScale(scale_limit=[-0.1, 0.1], interpolation=1, p=1.0),
    ])
        
    # Get list of image files in the emotion folder
    images = [
        img for img in image_folder_path.iterdir()
        if img.is_file() and img.suffix.lower() in image_extensions
    ]

    if len(images) < minimum_images:
        print(f"\n🔄 Augmentation pour le dossier '{image_folder_path.name}' (images actuelles : {len(images)})")
        created_images = 0
        while len(images) + created_images < minimum_images:
            image_path = np.random.choice(images) # get random image
            image = np.array(Image.open(image_path))
            augmented_image = augment_normal(image=image)["image"]
            output_image = Image.fromarray(augmented_image)
            output_image.save(os.path.join(image_folder_path, f'augmented_{created_images}.png'))
            created_images += 1

        print(f"✅ Augmentation terminée pour le dossier '{image_folder_path.name}'. Images créées : {created_images}")

if __name__ == "__main__":
    emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    for emotion in emotions:
        augment_image_folder(
            image_folder_path=f'./datasets/fer2013_images/{emotion}',
            minimum_images=10000
        )
    print(f"\n🎉 Dataset augmentation done !")


