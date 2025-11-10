"""
🎯 SCRIPT DE PRÉDICTION AVEC MODÈLE GNN
========================================

Script pour faire des prédictions d'émotions avec un modèle GNN (Graph Neural Network) entraîné.

UTILISATION :
------------
# Prédiction sur une seule image
python scripts/predict_gnn_script.py \
    --model_path ./models/emotion_gnn/emotion_gnn.pth \
    --image_path ./test_image.jpg \
    --face_part eyes_mouth

# Prédiction sur un dossier d'images
python scripts/predict_gnn_script.py \
    --model_path ./models/emotion_gnn/emotion_gnn.pth \
    --image_dir ./test_images/ \
    --face_part eyes_mouth \
    --save_results

# Évaluation sur un test set
python scripts/predict_gnn_script.py \
    --model_path ./models/emotion_gnn/emotion_gnn.pth \
    --test_dir ./datasets/fer2013/test/
"""

import torch
import argparse
import sys
import os
from pathlib import Path
import json
import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from classes.EmotionGnnModel import ModelManager, predict_emotion_gnn


def predict_single_image(
    model_path: str,
    image_path: str,
    face_part: str = 'eyes_mouth',
    device: str = 'cpu',
    show_image: bool = False
):
    """
    Prédire l'émotion sur une seule image
    """
    print("\n" + "="*80)
    print("🎯 PRÉDICTION D'ÉMOTION AVEC GNN")
    print("="*80)
    
    # Charger le modèle
    print(f"\n📂 Chargement du modèle GNN...")
    model = ModelManager.load_model(model_path, device=device)
    
    # Prédire
    print(f"\n🔍 Analyse de l'image : {image_path}")
    result = predict_emotion_gnn(
        image_path=image_path,
        model=model,
        face_part=face_part,
        device=device
    )
    
    # Afficher les résultats
    print("\n" + "="*80)
    if result['success']:
        print("✅ RÉSULTAT DE LA PRÉDICTION")
        print("="*80)
        print(f"Émotion détectée : {result['predicted_emotion']}")
        print(f"Confiance        : {result['confidence']*100:.2f}%")
        print(f"\n📊 Probabilités pour chaque émotion :")
        
        # Trier par probabilité décroissante
        sorted_probs = sorted(
            result['all_probabilities'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        for emotion, prob in sorted_probs:
            bar = "█" * int(prob * 50)
            print(f"   {emotion:10s} : {bar} {prob*100:5.2f}%")
        
        # Afficher l'image si demandé
        if show_image:
            image = cv2.imread(image_path)
            if image is not None:
                # Ajouter le texte de prédiction
                text = f"{result['predicted_emotion']} ({result['confidence']*100:.1f}%)"
                cv2.putText(
                    image, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
                )
                
                cv2.imshow('Prediction GNN', image)
                print("\n💡 Appuyez sur une touche pour fermer la fenêtre...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
    else:
        print("❌ ÉCHEC DE LA PRÉDICTION")
        print("="*80)
        print(f"Erreur : {result.get('error', 'Erreur inconnue')}")
    
    print("="*80)
    
    return result


def predict_batch(
    model_path: str,
    image_dir: str,
    face_part: str = 'eyes_mouth',
    device: str = 'cpu',
    save_results: bool = False,
    output_path: str = None
):
    """
    Prédire les émotions sur un lot d'images
    """
    print("\n" + "="*80)
    print("🎯 PRÉDICTION PAR LOT AVEC GNN")
    print("="*80)
    
    # Charger le modèle
    print(f"\n📂 Chargement du modèle GNN...")
    model = ModelManager.load_model(model_path, device=device)
    
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
    
    print(f"\n🔍 Traitement des images...")
    
    for image_path in tqdm(image_files, desc="Prédiction", ncols=100):
        result = predict_emotion_gnn(
            image_path=str(image_path),
            model=model,
            face_part=face_part,
            device=device
        )
        
        results.append(result)
        if result['success']:
            success_count += 1
    
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
            output_path = image_dir / "predictions_gnn.json"
        
        output_path = Path(output_path)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Résultats sauvegardés : {output_path}")
    
    print("="*80)
    
    return results


def evaluate_on_test_set(
    model_path: str,
    test_dir: str,
    face_part: str = 'eyes_mouth',
    device: str = 'cpu'
):
    """
    Évaluer le modèle GNN sur un ensemble de test organisé par émotions
    
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
    print("📊 ÉVALUATION GNN SUR ENSEMBLE DE TEST")
    print("="*80)
    
    # Charger le modèle
    print(f"\n📂 Chargement du modèle GNN...")
    model = ModelManager.load_model(model_path, device=device)
    
    test_dir = Path(test_dir)
    
    # Trouver tous les dossiers d'émotions
    emotion_dirs = [d for d in test_dir.iterdir() if d.is_dir()]
    
    if len(emotion_dirs) == 0:
        print(f"❌ Aucun dossier trouvé dans {test_dir}")
        return
    
    print(f"\n📁 {len(emotion_dirs)} classes d'émotions trouvées")
    
    # Évaluer sur chaque émotion
    total_correct = 0
    total_images = 0
    confusion = {}
    
    for emotion_dir in emotion_dirs:
        true_emotion = emotion_dir.name.capitalize()
        
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
            result = predict_emotion_gnn(
                image_path=str(image_path),
                model=model,
                face_part=face_part,
                device=device
            )
            
            if result['success']:
                pred_emotion = result['predicted_emotion']
                
                if pred_emotion == true_emotion:
                    correct += 1
                
                emotion_confusion[pred_emotion] = emotion_confusion.get(pred_emotion, 0) + 1
        
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
        description="Prédiction d'émotions avec modèle GNN",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation :
------------------------
# Prédiction sur une image unique
python scripts/predict_gnn_script.py --model_path ./models/emotion_gnn/emotion_gnn.pth --image_path ./test.jpg --show_image

# Prédiction sur un dossier d'images
python scripts/predict_gnn_script.py --model_path ./models/emotion_gnn/emotion_gnn.pth --image_dir ./test_images/ --save_results

# Évaluation sur un test set organisé par émotions
python scripts/predict_gnn_script.py --model_path ./models/emotion_gnn/emotion_gnn.pth --test_dir ./datasets/fer2013/test/
        """
    )
    parser.add_argument("--model_path", default='./models/emotion_gnn_full_face_medium/emotion_gnn_full_face_medium.pth', type=str,
                       help="Chemin vers le modèle .pth (REQUIS)")
    parser.add_argument("--image_path", type=str, default=None,
                       help="Chemin vers une image (mode image unique)")
    parser.add_argument("--image_dir", type=str, default=None,
                       help="Chemin vers un dossier d'images (mode batch)")
    parser.add_argument("--test_dir", type=str, default='./datasets/fer2013/test/',
                       help="Chemin vers un dossier de test organisé par émotions (mode évaluation)")
    parser.add_argument("--face_part", type=str, default="full_face",
                       choices=['full_face', 'eyes', 'mouth', 'eyes_mouth'],
                       help="Partie du visage à utiliser (défaut: full_face)")
    parser.add_argument("--device", type=str, default="cuda",
                       choices=['cpu', 'cuda'],
                       help="Device à utiliser (défaut: cuda)")
    parser.add_argument("--show_image", action="store_true",
                       help="Afficher l'image avec la prédiction (mode image unique)")
    parser.add_argument("--save_results", action="store_true",
                       help="Sauvegarder les résultats en JSON (mode batch)")
    parser.add_argument("--output_path", type=str, default=None,
                       help="Chemin de sauvegarde des résultats")
    
    args = parser.parse_args()
    
    # Vérifier que model_path est fourni
    if args.model_path is None:
        print("\n" + "="*80)
        print("❌ ERREUR : Le chemin du modèle est requis (--model_path)")
        print("="*80)
        parser.print_help()
        sys.exit(1)
    
    # Vérifier que le modèle existe
    if not Path(args.model_path).exists():
        print("\n" + "="*80)
        print(f"❌ ERREUR : Le fichier modèle n'existe pas : {args.model_path}")
        print("="*80)
        sys.exit(1)
    
    # Vérifier le device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA non disponible, utilisation du CPU")
        args.device = 'cpu'
    
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
            face_part=args.face_part,
            device=args.device,
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
            face_part=args.face_part,
            device=args.device,
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
            face_part=args.face_part,
            device=args.device
        )
