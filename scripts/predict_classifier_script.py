"""
🎯 SCRIPT DE PRÉDICTION CLASSIFIER POUR CLASSIFICATION D'ÉMOTIONS
==============================================================================
Ce script permet de prédire les émotions à partir d'images en utilisant un modèle pré-entraîné.
COMMANDE D'UTILISATION :
-----------------------
# Prédiction sur une image unique
python scripts/predict_classifier_script.py \
    --image_path ./test_images/happy_face.jpg \
    --model_path ./models/emotion_classifier/emotion_classifier.pth \
    --visualize \
    --device cuda

# Prédiction sur un dossier d'images
python scripts/predict_classifier_script.py \
    --folder_path ./test_images/ \
    --model_path ./models/emotion_classifier/emotion_classifier.pth \
    --device cuda

# Évaluation sur un test set
python scripts/predict_classifier_script.py \
    --test_data_dir ./datasets/fer2013_eyes_mouth_features/ \
    --model_path ./models/emotion_classifier/emotion_classifier.pth \
    --device cuda
"""
import numpy as np
import torch
import cv2
from pathlib import Path
from typing import Dict, List
import json
import argparse
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from classes.EmotionClassifierModel import ModelManager, predict_emotion

MODEL_PATH = "./models/emotion_classifier/emotion_classifier.pth"
THRESHOLD_FACE_DETECTION = 0.6

def predict_single_image(
    image_path: str,
    model_path: str,
    visualize: bool = True,
    device: str = 'cuda'
) -> Dict:
    """
    Prédire l'émotion d'une seule image
    
    Args:
        image_path: Chemin vers l'image
        model_path: Chemin vers le modèle .pth
        visualize: Afficher l'image avec la prédiction
        device: 'cuda' ou 'cuda'
    
    Returns:
        Dictionnaire avec les résultats de la prédiction
    """
    
    print("="*80)
    print("🎭 PRÉDICTION D'ÉMOTION")
    print("="*80)
    
    # Charger le modèle
    print(f"📦 Chargement du modèle...")
    model = ModelManager.load_model(model_path, device=device)
    
    # Prédire
    print(f"\n🔍 Analyse de l'image : {image_path}")
    result = predict_emotion(image_path, model, device=device)

    if not result['success']:
        print(f"\n❌ {result['error']}")
        return result
    
    # Afficher les résultats
    print("\n" + "="*80)
    print("📊 RÉSULTATS")
    print("="*80)
    print(f"Émotion prédite : {result['predicted_emotion']}")
    print(f"Confiance       : {result['confidence']*100:.2f}%")
    print(f"\n📈 Probabilités pour toutes les classes :")
    
    # Trier par probabilité décroissante
    sorted_probs = sorted(
        result['all_probabilities'].items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    for emotion, prob in sorted_probs:
        bar = '█' * int(prob * 50)
        print(f"   {emotion:12} : {prob*100:5.2f}% {bar}")
    
    print("="*80)
    
    # Visualisation
    if visualize:
        visualize_prediction(image_path, result)
    
    return result

def predict_batch(
    image_paths: List[str],
    model_path: str,
    device: str = 'cuda',
    save_results: bool = True,
    output_file: str = "predictions.json"
) -> List[Dict]:
    """
    Prédire les émotions pour plusieurs images
    
    Args:
        image_paths: Liste de chemins d'images
        model_path: Chemin vers le modèle .pth
        device: 'cuda' ou 'cuda'
        save_results: Sauvegarder les résultats dans un JSON
        output_file: Nom du fichier de sortie
    
    Returns:
        Liste de dictionnaires avec les prédictions
    """
    
    print("="*80)
    print("🎭 PRÉDICTION PAR BATCH")
    print("="*80)
    print(f"Nombre d'images : {len(image_paths)}\n")
    
    # Charger le modèle
    print(f"📦 Chargement du modèle...")
    model = ModelManager.load_model(model_path, device=device)
    
    # Prédire pour chaque image
    results = []
    successful = 0
    failed = 0
    
    for i, img_path in enumerate(image_paths, 1):
        print(f"\n[{i}/{len(image_paths)}] {img_path}")

        result = predict_emotion(img_path, model, device=device)
        results.append(result)
        
        if result['success']:
            print(f"  ✅ {result['predicted_emotion']} ({result['confidence']*100:.2f}%)")
            successful += 1
        else:
            print(f"  ❌ {result['error']}")
            failed += 1
    
    # Statistiques
    print("\n" + "="*80)
    print("📊 RÉSUMÉ")
    print("="*80)
    print(f"Total images     : {len(image_paths)}")
    print(f"Succès           : {successful} ({successful/len(image_paths)*100:.1f}%)")
    print(f"Échecs           : {failed} ({failed/len(image_paths)*100:.1f}%)")
    
    # Distribution des émotions prédites
    if successful > 0:
        emotion_counts = {}
        for result in results:
            if result['success']:
                emotion = result['predicted_emotion']
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        print(f"\n📈 Distribution des émotions prédites :")
        for emotion, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = count / successful * 100
            bar = '█' * int(percentage / 2)
            print(f"   {emotion:12} : {count:4} ({percentage:5.1f}%) {bar}")
    
    print("="*80)
    
    # Sauvegarder les résultats
    if save_results:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Résultats sauvegardés : {output_file}")
    
    return results

def predict_from_folder(
    folder_path: str,
    model_path: str,
    device: str = 'cuda',
    output_file: str = "predictions.json"
) -> List[Dict]:
    """
    Prédire toutes les images d'un dossier
    
    Args:
        folder_path: Chemin vers le dossier contenant les images
        model_path: Chemin vers le modèle .pth
        device: 'cuda' ou 'cuda'
        output_file: Nom du fichier de sortie
    
    Returns:
        Liste de dictionnaires avec les prédictions
    """
    
    # Extensions d'images supportées
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    # Collecter les images
    folder = Path(folder_path)
    image_paths = [
        str(img) for img in folder.iterdir()
        if img.is_file() and img.suffix.lower() in image_extensions
    ]
    
    print(f"📁 Dossier : {folder_path}")
    print(f"📊 Images trouvées : {len(image_paths)}")
    
    if not image_paths:
        print("❌ Aucune image trouvée")
        return []
    
    # Prédire
    return predict_batch(image_paths, model_path, device, save_results=True, output_file=output_file)

def visualize_prediction(image_path: str, result: Dict):
    """
    Afficher l'image avec la prédiction
    
    Args:
        image_path: Chemin vers l'image
        result: Résultat de la prédiction
    """
    import matplotlib.pyplot as plt
    
    # Charger l'image
    image = cv2.imread(image_path)
    if image is None:
        print(f"⚠️  Impossible de charger l'image pour visualisation")
        return
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Créer la figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Image
    axes[0].imshow(image_rgb)
    axes[0].set_title(f"Émotion prédite : {result['predicted_emotion']}\nConfiance : {result['confidence']*100:.2f}%", 
                     fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Barres de probabilités
    emotions = list(result['all_probabilities'].keys())
    probabilities = [result['all_probabilities'][e] * 100 for e in emotions]
    
    colors = ['red' if e == result['predicted_emotion'] else 'steelblue' for e in emotions]
    
    axes[1].barh(emotions, probabilities, color=colors)
    axes[1].set_xlabel('Probabilité (%)', fontsize=12)
    axes[1].set_title('Distribution des Probabilités', fontsize=14, fontweight='bold')
    axes[1].set_xlim(0, 100)
    
    # Ajouter les valeurs sur les barres
    for i, (emotion, prob) in enumerate(zip(emotions, probabilities)):
        axes[1].text(prob + 1, i, f'{prob:.1f}%', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.show()

def evaluate_on_test_set(
    data_dir: str,
    model_path: str,
    device: str = 'cuda'
) -> Dict:
    """
    Évaluer le modèle sur le set de test
    
    Args:
        data_dir: Dossier contenant X_test.npy et y_test.npy
        model_path: Chemin vers le modèle .pth
        device: 'cuda' ou 'cuda'
    
    Returns:
        Dictionnaire avec les métriques d'évaluation
    """
    from sklearn.metrics import classification_report, confusion_matrix
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    print("="*80)
    print("📊 ÉVALUATION SUR LE SET DE TEST")
    print("="*80)
    
    # Charger les données de test
    X_test = np.load(Path(data_dir) / "X_test.npy")
    y_test = np.load(Path(data_dir) / "y_test.npy")
    
    print(f"\n📁 Test set shape : {X_test.shape}")
    
    # Charger le modèle
    print(f"\n📦 Chargement du modèle...")
    model = ModelManager.load_model(model_path, device=device)
    
    # Convertir en tensors
    X_test_tensor = torch.from_numpy(X_test).to(device)
    
    # Prédictions
    print(f"\n🔄 Prédictions en cours...")
    predictions, probabilities = model.predict(X_test_tensor)
    
    y_pred = predictions.cpu().numpy()
    y_true = y_test
    
    # Calcul de l'accuracy
    accuracy = np.mean(y_pred == y_true)
    
    print(f"\n✅ Test Accuracy : {accuracy*100:.2f}%")
    
    # Classification report
    emotion_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    
    print("\n" + "="*80)
    print("📈 CLASSIFICATION REPORT")
    print("="*80)
    print(classification_report(y_true, y_pred, target_names=emotion_names, digits=4))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=emotion_names, yticklabels=emotion_names,
                cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.title('Confusion Matrix - Test Set', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Sauvegarder
    cm_path = Path(model_path).parent / "confusion_matrix_test.png"
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    print(f"\n💾 Matrice de confusion sauvegardée : {cm_path}")
    
    plt.show()
    
    results = {
        'test_accuracy': float(accuracy),
        'test_samples': len(y_test),
        'confusion_matrix': cm.tolist(),
        'classification_report': classification_report(y_true, y_pred, target_names=emotion_names, output_dict=True)
    }
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script de prédiction pour classification d'émotions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="""
Exemples d'utilisation :
# Prédiction sur une image unique
python scripts/predict_classifier_script.py --image_path ./datasets/test_images/happy_face.jpg --model
_path ./models/emotion_classifier/emotion_classifier.pth --visualize --device cuda

# Prédiction sur un dossier d'images
python scripts/predict_classifier_script.py --folder_path ./datasets/test_images/ --model_path ./models
/emotion_classifier/emotion_classifier.pth --device cuda
    
# Évaluation sur un test set
python scripts/predict_classifier_script.py --test_data_dir ./datasets/fer2013/test/ --model_path ./models
/emotion_classifier/emotion_classifier.pth --device cuda
        """
    )
    parser.add_argument("--image_path", type=str, default=None,
                        help="Chemin vers une image unique pour la prédiction")
    parser.add_argument("--folder_path", type=str, default=None,
                        help="Chemin vers un dossier d'images pour la prédiction par batch")
    parser.add_argument("--test_data_dir", type=str, default=None,
                        help="Dossier contenant X_test.npy et y_test.npy pour évaluation")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Chemin vers le modèle pré-entraîné (.pth)")
    parser.add_argument("--visualize", action='store_true',
                        help="Afficher l'image avec la prédiction (seulement pour image unique)")
    parser.add_argument("--device", type=str, default='cuda',
                        help="Device à utiliser : 'cuda' ou 'cpu'")
    
    args = parser.parse_args()
    
    if args.image_path:
        predict_single_image(
            image_path=args.image_path,
            model_path=args.model_path,
            visualize=args.visualize,
            device=args.device
        )
    elif args.folder_path:
        predict_from_folder(
            folder_path=args.folder_path,
            model_path=args.model_path,
            device=args.device,
            output_file="predictions_batch.json"
        )
    elif args.test_data_dir:
        evaluate_on_test_set(
            data_dir=args.test_data_dir,
            model_path=args.model_path,
            device=args.device
        )
    else:
        print("❌ Veuillez fournir --image_path, --folder_path ou --test_data_dir pour la prédiction.")