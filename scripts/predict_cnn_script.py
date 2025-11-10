"""
🎯 SCRIPT DE PRÉDICTION CNN POUR CLASSIFICATION D'ÉMOTIONS
===========================================================

Script pour faire des prédictions d'émotions avec un modèle CNN entraîné.
Support de Test-Time Augmentation (TTA) pour améliorer les performances.

MODES DISPONIBLES :
-------------------
1. Single Image  : Prédire une seule image
2. Batch         : Prédire un dossier d'images
3. Evaluation    : Évaluer sur un test set organisé par classes

UTILISATION :
-------------
# Prédiction sur une image unique
python scripts/predict_cnn_script.py \
    --model_path ./models/emotion_cnn/emotion_cnn.pth \
    --image_path ./test_image.jpg \
    --show_image

# Prédiction sur un dossier d'images
python scripts/predict_cnn_script.py \
    --model_path ./models/emotion_cnn/emotion_cnn.pth \
    --image_dir ./test_images/ \
    --save_results

# Évaluation sur un test set
python scripts/predict_cnn_script.py \
    --model_path ./models/emotion_cnn/emotion_cnn.pth \
    --test_dir ./datasets/fer2013_images/ \
    --use_tta

# Test-Time Augmentation (TTA) pour boost final (+1-2%)
python scripts/predict_cnn_script.py \
    --model_path ./models/emotion_cnn/emotion_cnn.pth \
    --image_path ./test_image.jpg \
    --use_tta \
    --tta_transforms 5
"""

import torch
from torchvision import transforms
from PIL import Image
import argparse
import sys
import os
from pathlib import Path
import json
import cv2
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from classes.EmotionCnnModel import EmotionCNN, ModelManager


def get_inference_transform(image_size: int = 224):
    """
    Transformation pour l'inférence (sans augmentation)
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize
    ])


def get_tta_transforms(image_size: int = 224, n_transforms: int = 5):
    """
    Test-Time Augmentation (TTA) : plusieurs transformations pour une image
    
    Retourne une liste de transformations légèrement différentes.
    Les prédictions sont ensuite moyennées pour plus de robustesse.
    
    Args:
        image_size: Taille des images
        n_transforms: Nombre de transformations (5-10 recommandé)
    
    Returns:
        Liste de transformations
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    tta_list = []
    
    # Transform 1 : Original
    tta_list.append(transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize
    ]))
    
    # Transform 2 : Horizontal flip
    tta_list.append(transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        normalize
    ]))
    
    # Transform 3 : Légère rotation gauche
    tta_list.append(transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomRotation(degrees=(-5, -5)),
        transforms.ToTensor(),
        normalize
    ]))
    
    # Transform 4 : Légère rotation droite
    tta_list.append(transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomRotation(degrees=(5, 5)),
        transforms.ToTensor(),
        normalize
    ]))
    
    # Transform 5 : Brightness ajustement
    if n_transforms >= 5:
        tta_list.append(transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ColorJitter(brightness=0.1),
            transforms.ToTensor(),
            normalize
        ]))
    
    # Transform 6-10 : Variations supplémentaires
    if n_transforms >= 6:
        tta_list.append(transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ColorJitter(contrast=0.1),
            transforms.ToTensor(),
            normalize
        ]))
    
    if n_transforms >= 7:
        tta_list.append(transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.ToTensor(),
            normalize
        ]))
    
    if n_transforms >= 8:
        tta_list.append(transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.RandomRotation(degrees=(-5, 5)),
            transforms.ToTensor(),
            normalize
        ]))
    
    if n_transforms >= 9:
        tta_list.append(transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            normalize
        ]))
    
    if n_transforms >= 10:
        tta_list.append(transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ColorJitter(brightness=0.1),
            transforms.ToTensor(),
            normalize
        ]))
    
    return tta_list[:n_transforms]


def predict_with_tta(
    model: torch.nn.Module,
    image: Image.Image,
    device: str,
    n_transforms: int = 5,
    image_size: int = 224
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prédiction avec Test-Time Augmentation
    
    Args:
        model: Modèle CNN
        image: Image PIL
        device: 'cuda' ou 'cuda'
        n_transforms: Nombre de transformations TTA
        image_size: Taille des images
    
    Returns:
        prediction: Classe prédite
        probabilities: Probabilités moyennes
    """
    tta_transforms = get_tta_transforms(image_size, n_transforms)
    
    all_probs = []
    
    model.eval()
    with torch.no_grad():
        for transform in tta_transforms:
            img_tensor = transform(image).unsqueeze(0).to(device)
            _, probs = model.predict(img_tensor)
            all_probs.append(probs.cpu())
    
    # Moyenner les probabilités
    avg_probs = torch.stack(all_probs).mean(dim=0)
    prediction = torch.argmax(avg_probs, dim=1)
    
    return prediction, avg_probs


def load_model_with_metadata(model_path: str, device: str = 'cuda') -> Tuple[torch.nn.Module, Dict]:
    """
    Charger le modèle et ses métadonnées
    
    Returns:
        model, metadata
    """
    model_path = Path(model_path)
    
    # Charger le modèle
    model = ModelManager.load_model(str(model_path), device=device)
    
    # Charger les métadonnées si disponibles
    metadata_path = model_path.parent / f"{model_path.stem}_metadata.json"
    
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    else:
        # Métadonnées par défaut
        metadata = {
            'n_classes': 7,
            'class_names': ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise'],
            'image_size': 224
        }
        print(f"⚠️  Métadonnées non trouvées, utilisation des valeurs par défaut")
    
    return model, metadata


def predict_single_image(
    model_path: str,
    image_path: str,
    device: str = 'cuda',
    use_tta: bool = False,
    tta_transforms: int = 5,
    show_image: bool = False
):
    """
    Prédire l'émotion sur une seule image
    """
    print("\n" + "="*80)
    print("🎯 PRÉDICTION D'ÉMOTION AVEC CNN")
    print("="*80)
    
    # Charger le modèle et les métadonnées
    model, metadata = load_model_with_metadata(model_path, device)
    
    class_names = metadata['class_names']
    image_size = metadata.get('image_size', 224)
    
    # Charger l'image
    print(f"\n🔍 Analyse de l'image : {image_path}")
    image = Image.open(image_path).convert('RGB')
    
    # Prédire
    if use_tta:
        print(f"   Mode : Test-Time Augmentation ({tta_transforms} transformations)")
        prediction, probabilities = predict_with_tta(
            model, image, device, tta_transforms, image_size
        )
    else:
        print(f"   Mode : Inférence standard")
        transform = get_inference_transform(image_size)
        img_tensor = transform(image).unsqueeze(0).to(device)
        prediction, probabilities = model.predict(img_tensor)
    
    # Résultats
    pred_class = prediction.item()
    pred_probs = probabilities.cpu().numpy()[0]
    
    print("\n" + "="*80)
    print("✅ RÉSULTAT DE LA PRÉDICTION")
    print("="*80)
    print(f"Émotion détectée : {class_names[pred_class]}")
    print(f"Confiance        : {pred_probs[pred_class]*100:.2f}%")
    print(f"\n📊 Probabilités pour chaque émotion :")
    
    # Trier par probabilité décroissante
    sorted_indices = np.argsort(pred_probs)[::-1]
    
    for idx in sorted_indices:
        emotion = class_names[idx]
        prob = pred_probs[idx]
        bar = "█" * int(prob * 50)
        print(f"   {emotion:10s} : {bar} {prob*100:5.2f}%")
    
    # Afficher l'image si demandé
    if show_image:
        img_cv = cv2.imread(image_path)
        if img_cv is not None:
            # Ajouter le texte de prédiction
            text = f"{class_names[pred_class]} ({pred_probs[pred_class]*100:.1f}%)"
            cv2.putText(
                img_cv, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
            )
            
            cv2.imshow('Prediction CNN', img_cv)
            print("\n💡 Appuyez sur une touche pour fermer la fenêtre...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    print("="*80)
    
    return {
        'success': True,
        'predicted_emotion': class_names[pred_class],
        'predicted_class_id': pred_class,
        'confidence': float(pred_probs[pred_class]),
        'all_probabilities': {
            class_names[i]: float(pred_probs[i])
            for i in range(len(class_names))
        },
        'image_path': image_path,
        'use_tta': use_tta
    }


def predict_batch(
    model_path: str,
    image_dir: str,
    device: str = 'cuda',
    use_tta: bool = False,
    tta_transforms: int = 5,
    save_results: bool = False,
    output_path: str = None
):
    """
    Prédire les émotions sur un lot d'images
    """
    print("\n" + "="*80)
    print("🎯 PRÉDICTION PAR LOT AVEC CNN")
    print("="*80)
    
    # Charger le modèle
    model, metadata = load_model_with_metadata(model_path, device)
    
    class_names = metadata['class_names']
    image_size = metadata.get('image_size', 224)
    
    # Trouver toutes les images
    image_dir = Path(image_dir)
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(list(image_dir.glob(f"*{ext}")))
        image_files.extend(list(image_dir.glob(f"*{ext.upper()}")))
    
    print(f"\n📁 {len(image_files)} images trouvées dans {image_dir}")
    
    if len(image_files) == 0:
        print("❌ Aucune image trouvée !")
        return []
    
    # Prédire sur toutes les images
    results = []
    success_count = 0
    
    transform = get_inference_transform(image_size)
    
    print(f"\n🔍 Traitement des images...")
    if use_tta:
        print(f"   Mode : Test-Time Augmentation ({tta_transforms} transformations)")
    
    for image_path in tqdm(image_files, desc="Prédiction", ncols=100):
        try:
            image = Image.open(image_path).convert('RGB')
            
            if use_tta:
                prediction, probabilities = predict_with_tta(
                    model, image, device, tta_transforms, image_size
                )
            else:
                img_tensor = transform(image).unsqueeze(0).to(device)
                prediction, probabilities = model.predict(img_tensor)
            
            pred_class = prediction.item()
            pred_probs = probabilities.cpu().numpy()[0]
            
            results.append({
                'success': True,
                'predicted_emotion': class_names[pred_class],
                'predicted_class_id': pred_class,
                'confidence': float(pred_probs[pred_class]),
                'all_probabilities': {
                    class_names[i]: float(pred_probs[i])
                    for i in range(len(class_names))
                },
                'image_path': str(image_path),
                'use_tta': use_tta
            })
            success_count += 1
            
        except Exception as e:
            results.append({
                'success': False,
                'error': str(e),
                'image_path': str(image_path)
            })
    
    # Afficher les statistiques
    print("\n" + "="*80)
    print("📊 STATISTIQUES")
    print("="*80)
    print(f"Total d'images    : {len(image_files)}")
    print(f"Succès            : {success_count}")
    print(f"Échecs            : {len(image_files) - success_count}")
    print(f"Taux de réussite  : {success_count/len(image_files)*100:.1f}%")
    
    # Compter les émotions détectées
    if success_count > 0:
        emotion_counts = {}
        for result in results:
            if result['success']:
                emotion = result['predicted_emotion']
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        print(f"\n📊 Distribution des émotions détectées :")
        for emotion, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = count / success_count * 100
            bar = "█" * int(percentage / 2)
            print(f"   {emotion:10s} : {bar} {count:4d} ({percentage:5.1f}%)")
    
    # Sauvegarder les résultats si demandé
    if save_results:
        if output_path is None:
            output_path = image_dir / "predictions_cnn.json"
        
        output_path = Path(output_path)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Résultats sauvegardés : {output_path}")
    
    print("="*80)
    
    return results


def evaluate_on_test_set(
    model_path: str,
    test_dir: str,
    device: str = 'cuda',
    use_tta: bool = False,
    tta_transforms: int = 5
):
    """
    Évaluer le modèle CNN sur un ensemble de test organisé par émotions
    
    Structure attendue :
    test_dir/
        angry/
            image1.jpg
            image2.jpg
        happy/
            image1.jpg
        ...
    """
    print("\n" + "="*80)
    print("📊 ÉVALUATION CNN SUR ENSEMBLE DE TEST")
    print("="*80)
    
    # Charger le modèle
    model, metadata = load_model_with_metadata(model_path, device)
    
    class_names = metadata['class_names']
    image_size = metadata.get('image_size', 224)
    
    test_dir = Path(test_dir)
    
    # Trouver tous les dossiers d'émotions
    emotion_dirs = [d for d in test_dir.iterdir() if d.is_dir()]
    
    if len(emotion_dirs) == 0:
        print(f"❌ Aucun dossier trouvé dans {test_dir}")
        return
    
    print(f"\n📁 {len(emotion_dirs)} classes d'émotions trouvées")
    if use_tta:
        print(f"   Mode : Test-Time Augmentation ({tta_transforms} transformations)")
    
    # Évaluer sur chaque émotion
    total_correct = 0
    total_images = 0
    confusion = {}
    
    transform = get_inference_transform(image_size)
    
    for emotion_dir in emotion_dirs:
        true_emotion = emotion_dir.name
        
        # Trouver toutes les images
        image_files = list(emotion_dir.glob("*.jpg")) + \
                     list(emotion_dir.glob("*.png")) + \
                     list(emotion_dir.glob("*.jpeg"))
        
        if len(image_files) == 0:
            continue
        
        print(f"\n🔍 Évaluation de '{true_emotion}' ({len(image_files)} images)...")
        
        correct = 0
        emotion_confusion = {}
        
        for image_path in tqdm(image_files, desc=f"{true_emotion}", leave=False, ncols=80):
            try:
                image = Image.open(image_path).convert('RGB')
                
                if use_tta:
                    prediction, probabilities = predict_with_tta(
                        model, image, device, tta_transforms, image_size
                    )
                else:
                    img_tensor = transform(image).unsqueeze(0).to(device)
                    prediction, probabilities = model.predict(img_tensor)
                
                pred_emotion = class_names[prediction.item()]
                
                if pred_emotion == true_emotion:
                    correct += 1
                
                emotion_confusion[pred_emotion] = emotion_confusion.get(pred_emotion, 0) + 1
                
            except Exception as e:
                print(f"   ⚠️  Erreur sur {image_path.name}: {e}")
        
        accuracy = correct / len(image_files) * 100 if len(image_files) > 0 else 0
        print(f"   Accuracy : {accuracy:.1f}% ({correct}/{len(image_files)})")
        
        total_correct += correct
        total_images += len(image_files)
        confusion[true_emotion] = emotion_confusion
    
    # Afficher les résultats globaux
    print("\n" + "="*80)
    print("📊 RÉSULTATS GLOBAUX")
    print("="*80)
    print(f"Total d'images : {total_images}")
    print(f"Correctes      : {total_correct}")
    print(f"Accuracy       : {total_correct/total_images*100:.2f}%")
    
    # Afficher la matrice de confusion
    print("\n📊 MATRICE DE CONFUSION :")
    print("="*80)
    
    emotions = sorted(confusion.keys())
    
    # En-tête
    print(f"{'Vraie→ Prédite↓':<15s}", end="")
    for emotion in emotions:
        print(f"{emotion[:4]:>6s}", end="")
    print()
    print("-" * (15 + 6 * len(emotions)))
    
    # Lignes
    for true_emotion in emotions:
        print(f"{true_emotion:<15s}", end="")
        for pred_emotion in emotions:
            count = confusion[true_emotion].get(pred_emotion, 0)
            print(f"{count:>6d}", end="")
        print()
    
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prédiction d'émotions avec modèle CNN",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation :
------------------------
# Prédiction sur une image unique
python scripts/predict_cnn_script.py --model_path ./models/emotion_cnn/emotion_cnn.pth --image_path ./test.jpg --show_image

# Prédiction avec TTA (Test-Time Augmentation) pour boost +1-2%
python scripts/predict_cnn_script.py --model_path ./models/emotion_cnn/emotion_cnn.pth --image_path ./test.jpg --use_tta --tta_transforms 5

# Prédiction sur un dossier d'images
python scripts/predict_cnn_script.py --model_path ./models/emotion_cnn/emotion_cnn.pth --image_dir ./test_images/ --save_results

# Évaluation sur un test set organisé par émotions
python scripts/predict_cnn_script.py --model_path ./models/emotion_cnn/emotion_cnn.pth --test_dir ./datasets/fer2013_images/ --use_tta
        """
    )
    
    # Paramètres principaux
    parser.add_argument("--model_path", type=str,
                       help="Chemin vers le modèle .pth (REQUIS)")
    parser.add_argument("--image_path", type=str, default=None,
                       help="Chemin vers une image (mode image unique)")
    parser.add_argument("--image_dir", type=str, default=None,
                       help="Chemin vers un dossier d'images (mode batch)")
    parser.add_argument("--test_dir", type=str, default=None,
                       help="Chemin vers un dossier de test organisé par émotions (mode évaluation)")
    
    # Test-Time Augmentation
    parser.add_argument("--use_tta", action="store_true",
                       help="Utiliser Test-Time Augmentation pour boost +1-2%")
    parser.add_argument("--tta_transforms", type=int, default=5,
                       help="Nombre de transformations TTA (5-10 recommandé)")
    
    # Device
    parser.add_argument("--device", type=str, default="cuda",
                       choices=['cuda', 'cuda'],
                       help="Device à utiliser (défaut: cuda)")
    
    # Options d'affichage
    parser.add_argument("--show_image", action="store_true",
                       help="Afficher l'image avec la prédiction (mode image unique)")
    parser.add_argument("--save_results", action="store_true",
                       help="Sauvegarder les résultats en JSON (mode batch)")
    parser.add_argument("--output_path", type=str, default=None,
                       help="Chemin de sauvegarde des résultats")
    
    args = parser.parse_args()

    # Vérifier que model_path est fourni et existe
    if not Path(args.model_path).exists():
        print("\n" + "="*80)
        print(f"❌ ERREUR : Le fichier modèle n'existe pas : {args.model_path}")
        print("="*80)
        sys.exit(1)
    
    # Vérifier le device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA non disponible, utilisation du CPU")
        args.device = 'cuda'
    
    # Vérifier qu'au moins un mode est spécifié
    if not any([args.image_path, args.image_dir, args.test_dir]):
        print("\n" + "="*80)
        print("❌ ERREUR : Vous devez spécifier au moins une option :")
        print("   --image_path  (pour une image unique)")
        print("   --image_dir   (pour un dossier d'images)")
        print("   --test_dir    (pour évaluer sur un test set)")
        print("="*80)
        parser.print_help()
        sys.exit(1)
    
    # Mode image unique
    if args.image_path:
        if not Path(args.image_path).exists():
            print(f"\n❌ ERREUR : L'image n'existe pas : {args.image_path}")
            sys.exit(1)
        
        predict_single_image(
            model_path=args.model_path,
            image_path=args.image_path,
            device=args.device,
            use_tta=args.use_tta,
            tta_transforms=args.tta_transforms,
            show_image=args.show_image
        )
    
    # Mode batch
    elif args.image_dir:
        if not Path(args.image_dir).exists():
            print(f"\n❌ ERREUR : Le dossier n'existe pas : {args.image_dir}")
            sys.exit(1)
        
        predict_batch(
            model_path=args.model_path,
            image_dir=args.image_dir,
            device=args.device,
            use_tta=args.use_tta,
            tta_transforms=args.tta_transforms,
            save_results=args.save_results,
            output_path=args.output_path
        )
    
    # Mode évaluation
    elif args.test_dir:
        if not Path(args.test_dir).exists():
            print(f"\n❌ ERREUR : Le dossier de test n'existe pas : {args.test_dir}")
            sys.exit(1)
        
        evaluate_on_test_set(
            model_path=args.model_path,
            test_dir=args.test_dir,
            device=args.device,
            use_tta=args.use_tta,
            tta_transforms=args.tta_transforms
        )
