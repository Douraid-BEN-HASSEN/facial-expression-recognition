import cv2
import mediapipe as mp
import numpy as np
from typing import List, Tuple, Optional

class FaceLandmarkExtractor:
    """
    Classe pour extraire les landmarks faciaux avec MediaPipe Face Mesh
    """
    
    def __init__(self, 
                 static_image_mode: bool = True,
                 max_num_faces: int = 1,
                 refine_landmarks: bool = True,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        """
        Initialise MediaPipe Face Mesh
        
        Args:
            static_image_mode: True pour images statiques, False pour vidéo
            max_num_faces: Nombre maximum de visages à détecter
            refine_landmarks: Affine les landmarks autour des yeux et lèvres
            min_detection_confidence: Seuil de confiance détection
            min_tracking_confidence: Seuil de confiance tracking
        """
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        self.landmark_regions = {
            'left_eye': [33, 160, 158, 133, 153, 144],
            'right_eye': [362, 385, 387, 263, 373, 380],
            'left_eyebrow': [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],
            'right_eyebrow': [300, 293, 334, 296, 336, 285, 295, 282, 283, 276],
            'nose': [1, 2, 98, 327, 205, 4],
            'mouth_outer': [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88],
            'mouth_inner': [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191],
            'jaw': [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109],

            'FACEMESH_LEFT_EYE': [263, 249, 249, 390, 390, 373, 373, 374,
                               374, 380, 380, 381, 381, 382, 382, 362,
                               263, 466, 466, 388, 388, 387, 387, 386,
                               386, 385, 385, 384, 384, 398, 398, 362],

            'FACEMESH_LEFT_IRIS': [474, 475, 475, 476, 476, 477,
                                            477, 474],

            'FACEMESH_LEFT_EYEBROW': [276, 283, 283, 282, 282, 295,
                                   295, 285, 300, 293, 293, 334,
                                   334, 296, 296, 336],

            'FACEMESH_RIGHT_EYE': [33, 7, 7, 163, 163, 144, 144, 145,
                                            145, 153, 153, 154, 154, 155, 155, 133,
                                            33, 246, 246, 161, 161, 160, 160, 159,
                                            159, 158, 158, 157, 157, 173, 173, 133],

            'FACEMESH_RIGHT_EYEBROW':  [46, 53, 53, 52, 52, 65, 65, 55,
                                                70, 63, 63, 105, 105, 66, 66, 107],

            'FACEMESH_RIGHT_IRIS': [469, 470, 470, 471, 471, 472,
                                            472, 469],

            'FACEMESH_LIPS': [61, 146, 146, 91, 91, 181, 181, 84, 84, 17,
                           17, 314, 314, 405, 405, 321, 321, 375,
                           375, 291, 61, 185, 185, 40, 40, 39, 39, 37,
                           37, 0, 0, 267,
                           267, 269, 269, 270, 270, 409, 409, 291,
                           78, 95, 95, 88, 88, 178, 178, 87, 87, 14,
                           14, 317, 317, 402, 402, 318, 318, 324,
                           324, 308, 78, 191, 191, 80, 80, 81, 81, 82,
                           82, 13, 13, 312, 312, 311, 311, 310,
                           310, 415, 415, 308]
        }

    def extract_landmarks(self, image: np.ndarray) -> Optional[List[Tuple[float, float, float]]]:
        """
        Extrait les 468 landmarks d'une image
        
        Args:
            image: Image BGR (OpenCV format)
            
        Returns:
            Liste de tuples (x, y, z) normalisés [0, 1] ou None si pas de visage
        """
        # Conversion BGR -> RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Détection
        results = self.face_mesh.process(image_rgb)
        
        if not results.multi_face_landmarks:
            return None
        
        landmarks = []

        # Récupération des landmarks (premier visage)
        face_landmarks = results.multi_face_landmarks[0]
        
        for landmark in face_landmarks.landmark:
            landmarks.append((landmark.x, landmark.y, landmark.z))
        
        return landmarks
    
    def visualize_landmarks(self, image: np.ndarray, show_tesselation: bool = True) -> np.ndarray:
        """
        Visualise les landmarks sur l'image
        
        Args:
            image: Image BGR
            show_tesselation: Afficher le maillage complet
            
        Returns:
            Image annotée
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)
        
        annotated_image = image.copy()
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                if show_tesselation:
                    # Dessine le maillage complet
                    self.mp_drawing.draw_landmarks(
                        image=annotated_image,
                        landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
                    )
                
                # Dessine les contours
                self.mp_drawing.draw_landmarks(
                    image=annotated_image,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style()
                )
                
                # Dessine les iris
                self.mp_drawing.draw_landmarks(
                    image=annotated_image,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_iris_connections_style()
                )
        
        return annotated_image

    def annotate_landmarks(self, image: np.ndarray, landmarks: List[Tuple]) -> np.ndarray:
        """
        Annotate the image with landmark points.
        
        Args:
            image: BGR image
            landmarks: List of (x, y, z) tuples of landmarks
            
        Returns:
            Annotated image
        """
        annotated_image = image.copy()
        h, w, _ = image.shape
        
        for idx, (x, y, z) in enumerate(landmarks):
            cx, cy = int(x * w), int(y * h)
            cv2.circle(annotated_image, (cx, cy), 1, (0, 255, 0), -1)
        
        # draw eye centers
        left_eye_indices = self.landmark_regions['left_eye']
        right_eye_indices = self.landmark_regions['right_eye']
        left_eye_points = np.array([ (int(landmarks[i][0] * w), int(landmarks[i][1] * h)) for i in left_eye_indices ])
        right_eye_points = np.array([ (int(landmarks[i][0] * w), int(landmarks[i][1] * h)) for i in right_eye_indices ])
        left_eye_center = tuple(np.mean(left_eye_points, axis=0).astype(int))
        right_eye_center = tuple(np.mean(right_eye_points, axis=0).astype(int))
        cv2.circle(annotated_image, left_eye_center, 3, (255, 0, 0), -1)
        cv2.circle(annotated_image, right_eye_center, 3, (255, 0, 0), -1)

        return annotated_image
    
    def normalize_landmarks(self, landmarks: List[Tuple]) -> List[Tuple]:
        """
        Normalisation complète : translation + rotation + mise à l'échelle
        
        Args:
            landmarks: Liste de tuples (x, y, z) des 468 landmarks
            
        Returns:
            Liste de tuples normalisés avec yeux alignés horizontalement
        """
        # Conversion en array numpy pour faciliter les calculs
        landmarks_array = np.array(landmarks)
        
        # Aligner les yeux en utilisant les bons indices MediaPipe
        left_eye_indices = self.landmark_regions['left_eye']
        right_eye_indices = self.landmark_regions['right_eye']
        
        left_eye_points = landmarks_array[left_eye_indices]
        right_eye_points = landmarks_array[right_eye_indices]
        
        left_eye_center = np.mean(left_eye_points, axis=0)
        right_eye_center = np.mean(right_eye_points, axis=0)
        eye_center = (left_eye_center + right_eye_center) / 2
        
        # ========== ÉTAPE 1 : TRANSLATION (centrage sur le milieu des yeux) ==========
        landmarks_array[:, 0] -= eye_center[0]
        landmarks_array[:, 1] -= eye_center[1]
        
        # Recalculer les centres des yeux après translation
        left_eye_center_translated = np.mean(landmarks_array[left_eye_indices], axis=0)
        right_eye_center_translated = np.mean(landmarks_array[right_eye_indices], axis=0)
        
        # ========== ÉTAPE 2 : ROTATION (aligner les yeux horizontalement) ==========
        # Calculer l'angle entre les yeux
        dx = right_eye_center_translated[0] - left_eye_center_translated[0]
        dy = right_eye_center_translated[1] - left_eye_center_translated[1]
        angle = np.arctan2(dy, dx)  # Angle en radians
        
        # Matrice de rotation pour ramener à l'horizontale
        cos_angle = np.cos(-angle)
        sin_angle = np.sin(-angle)
        rotation_matrix = np.array([
            [cos_angle, -sin_angle],
            [sin_angle, cos_angle]
        ])
        
        # Appliquer la rotation (seulement sur x, y)
        landmarks_xy = landmarks_array[:, :2]  # Extraire x, y
        landmarks_xy_rotated = landmarks_xy @ rotation_matrix.T
        landmarks_array[:, :2] = landmarks_xy_rotated
        
        # ========== ÉTAPE 3 : MISE À L'ÉCHELLE (normalisation Min-Max vers [0, 1]) ==========
        x_coords = landmarks_array[:, 0]
        y_coords = landmarks_array[:, 1]
        
        x_min, x_max = np.min(x_coords), np.max(x_coords)
        y_min, y_max = np.min(y_coords), np.max(y_coords)
        
        x_range = x_max - x_min
        y_range = y_max - y_min
        
        # Normaliser vers [0, 1]
        landmarks_array[:, 0] = (landmarks_array[:, 0] - x_min) / (x_range + 1e-6)
        landmarks_array[:, 1] = (landmarks_array[:, 1] - y_min) / (y_range + 1e-6)
        
        # Reconvertir en liste de tuples
        normalized_landmarks = [(x, y, z) for x, y, z in landmarks_array]
        
        return normalized_landmarks

    def normalize_face_part_landmarks(self, landmarks: List[Tuple]) -> List[Tuple]:
        """
        Normalisation complète : mise à l'échelle
        
        Args:
            landmarks: Liste de tuples (x, y, z) des landmarks
            
        Returns:
            Liste de tuples normalisés dans [0, 1] pour x, y, z
        """
        normalized_landmarks = []

        # Conversion en array numpy pour faciliter les calculs
        landmarks_array = np.array(landmarks, dtype=np.float32)
        
        # ========== MISE À L'ÉCHELLE (normalisation Min-Max vers [0, 1]) ==========
        x_coords = landmarks_array[:, 0]
        y_coords = landmarks_array[:, 1]
        z_coords = landmarks_array[:, 2]
        
        x_min, x_max = np.min(x_coords), np.max(x_coords)
        y_min, y_max = np.min(y_coords), np.max(y_coords)
        z_min, z_max = np.min(z_coords), np.max(z_coords)
        
        x_range = x_max - x_min
        y_range = y_max - y_min
        z_range = z_max - z_min
        
        # Normaliser x, y, z vers [0, 1]
        landmarks_array[:, 0] = (landmarks_array[:, 0] - x_min) / (x_range + 1e-6)
        landmarks_array[:, 1] = (landmarks_array[:, 1] - y_min) / (y_range + 1e-6)
        landmarks_array[:, 2] = (landmarks_array[:, 2] - z_min) / (z_range + 1e-6)
        
        # Reconvertir en liste de tuples
        normalized_landmarks = [(float(x), float(y), float(z)) for x, y, z in landmarks_array]
        
        return normalized_landmarks

    def landmarks_to_features(self, landmarks: List[Tuple[float, float]]) -> np.ndarray:
        """
        Convertit une liste de landmarks en un vecteur de features géométriques.
        
        Args:
            landmarks: Liste de tuples (x, y, z)
        
        Returns:
            Array numpy de shape (n_landmarks * 3,) avec dtype float32
        """
        features = []
        for (x, y, z) in landmarks:
            features.extend([x, y, z])

        return np.array(features, dtype=np.float32)

    def __del__(self):
        """Ferme proprement MediaPipe"""
        self.face_mesh.close()

def example_single_image():
    """Exemple : traiter une seule image"""
    # Charger une image
    image = cv2.imread("Maha_Habib_0001.jpg")
    
    # Initialiser l'extracteur
    extractor = FaceLandmarkExtractor()
    
    # Extraire les landmarks
    landmarks = extractor.extract_landmarks(image)

    if landmarks:
        print(f"✅ {len(landmarks)} landmarks détectés")
        
        # Features géométriques
        features = extractor.extract_geometric_features(landmarks)
        print(f"Features géométriques : {features}")
        
        # Visualisation
        annotated = extractor.annotate_landmarks(image, landmarks)
        cv2.imshow("Landmarks", annotated)
        
        # Normalisation
        normalized_landmarks = extractor.normalize_landmarks(landmarks)
         # Image noire pour visualisation 256 x 256
        normalized_image = np.zeros((256, 256, 3), dtype=np.uint8)
        annotated_normalized = extractor.annotate_landmarks(normalized_image, normalized_landmarks)
        cv2.imshow("Landmarks Normalisés", annotated_normalized)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("❌ Aucun visage détecté")
