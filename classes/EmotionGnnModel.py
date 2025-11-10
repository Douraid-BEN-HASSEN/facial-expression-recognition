"""
🎯 ARCHITECTURE GNN (GRAPH NEURAL NETWORK) POUR LANDMARKS FACIAUX
==================================================================

Cette architecture est OPTIMALE pour les landmarks faciaux car elle modélise
le visage comme un GRAPHE avec des connexions anatomiques naturelles.

🔥 FONCTIONNALITÉS CLÉS :
--------------------------------------------
1. **Structure naturelle** : Les landmarks forment un graphe facial anatomique
2. **Relations spatiales** : Les connexions respectent la géométrie du visage
3. **Moins de paramètres** : Plus efficace que Transformer pour données structurées
4. **Meilleure généralisation** : Exploite la topologie faciale

📊 ARCHITECTURE :
----------------
Input (478 landmarks × 3 coords)
    → Graph Construction (nœuds + arêtes anatomiques)
    → Graph Convolution Layers (apprentissage sur le graphe)
    → Global Pooling (agrégation des nœuds)
    → Classification Head → 7 émotions

🎯 CONNEXIONS ANATOMIQUES :
---------------------------
- Yeux : Contours des yeux (6 landmarks chacun)
- Sourcils : Ligne des sourcils (5 landmarks chacun)
- Bouche : Contour externe + interne (20+20 landmarks)
- Connexions croisées : Yeux ↔ Sourcils, Bouche ↔ Joues
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from typing import Tuple, List
from pathlib import Path


class FacialGraphConstructor:
    """
    Constructeur de graphe facial basé sur les VRAIES connexions anatomiques MediaPipe FaceMesh
    """
    
    # Vraies connexions MediaPipe FaceMesh (fournies par l'utilisateur)
    FACEMESH_LIPS = frozenset([(61, 146), (146, 91), (91, 181), (181, 84), (84, 17),
                               (17, 314), (314, 405), (405, 321), (321, 375),
                               (375, 291), (61, 185), (185, 40), (40, 39), (39, 37),
                               (37, 0), (0, 267),
                               (267, 269), (269, 270), (270, 409), (409, 291),
                               (78, 95), (95, 88), (88, 178), (178, 87), (87, 14),
                               (14, 317), (317, 402), (402, 318), (318, 324),
                               (324, 308), (78, 191), (191, 80), (80, 81), (81, 82),
                               (82, 13), (13, 312), (312, 311), (311, 310),
                               (310, 415), (415, 308)])

    FACEMESH_LEFT_EYE = frozenset([(263, 249), (249, 390), (390, 373), (373, 374),
                                   (374, 380), (380, 381), (381, 382), (382, 362),
                                   (263, 466), (466, 388), (388, 387), (387, 386),
                                   (386, 385), (385, 384), (384, 398), (398, 362)])

    FACEMESH_LEFT_IRIS = frozenset([(474, 475), (475, 476), (476, 477),
                                     (477, 474)])

    FACEMESH_LEFT_EYEBROW = frozenset([(276, 283), (283, 282), (282, 295),
                                       (295, 285), (300, 293), (293, 334),
                                       (334, 296), (296, 336)])

    FACEMESH_RIGHT_EYE = frozenset([(33, 7), (7, 163), (163, 144), (144, 145),
                                    (145, 153), (153, 154), (154, 155), (155, 133),
                                    (33, 246), (246, 161), (161, 160), (160, 159),
                                    (159, 158), (158, 157), (157, 173), (173, 133)])

    FACEMESH_RIGHT_EYEBROW = frozenset([(46, 53), (53, 52), (52, 65), (65, 55),
                                        (70, 63), (63, 105), (105, 66), (66, 107)])

    FACEMESH_RIGHT_IRIS = frozenset([(469, 470), (470, 471), (471, 472),
                                     (472, 469)])

    FACEMESH_FACE_OVAL = frozenset([(10, 338), (338, 297), (297, 332), (332, 284),
                                    (284, 251), (251, 389), (389, 356), (356, 454),
                                    (454, 323), (323, 361), (361, 288), (288, 397),
                                    (397, 365), (365, 379), (379, 378), (378, 400),
                                    (400, 377), (377, 152), (152, 148), (148, 176),
                                    (176, 149), (149, 150), (150, 136), (136, 172),
                                    (172, 58), (58, 132), (132, 93), (93, 234),
                                    (234, 127), (127, 162), (162, 21), (21, 54),
                                    (54, 103), (103, 67), (67, 109), (109, 10)])

    FACEMESH_NOSE = frozenset([(168, 6), (6, 197), (197, 195), (195, 5),
                               (5, 4), (4, 1), (1, 19), (19, 94), (94, 2), (98, 97),
                               (97, 2), (2, 326), (326, 327), (327, 294),
                               (294, 278), (278, 344), (344, 440), (440, 275),
                               (275, 4), (4, 45), (45, 220), (220, 115), (115, 48),
                               (48, 64), (64, 98)])
    
    @staticmethod
    def create_facial_graph(n_landmarks: int = 478) -> torch.Tensor:
        """
        Créer les arêtes du graphe pour FULL FACE avec TOUTES les connexions MediaPipe
        
        Utilise toutes les connexions anatomiques : lips, eyes, eyebrows, iris, nose, face_oval
        
        Args:
            n_landmarks: Nombre de landmarks (478 pour full_face)
        
        Returns:
            edge_index: Tensor de shape (2, num_edges) avec connexions anatomiques complètes
        """
        edges = []
        
        # 🔥 TOUTES les connexions MediaPipe pour full face
        all_connections = (
            FacialGraphConstructor.FACEMESH_LIPS |
            FacialGraphConstructor.FACEMESH_LEFT_EYE |
            FacialGraphConstructor.FACEMESH_LEFT_IRIS |
            FacialGraphConstructor.FACEMESH_LEFT_EYEBROW |
            FacialGraphConstructor.FACEMESH_RIGHT_EYE |
            FacialGraphConstructor.FACEMESH_RIGHT_EYEBROW |
            FacialGraphConstructor.FACEMESH_RIGHT_IRIS |
            FacialGraphConstructor.FACEMESH_FACE_OVAL |
            FacialGraphConstructor.FACEMESH_NOSE
        )
        
        # Convertir les connexions anatomiques en arêtes
        for src, dst in all_connections:
            if src < n_landmarks and dst < n_landmarks:
                edges.append([src, dst])
                edges.append([dst, src])  # Graphe non-dirigé
        
        n_anatomical_edges = len(edges) // 2
        n_connected_nodes = len(set([e[0] for e in edges] + [e[1] for e in edges]))
        
        print(f"📊 Connexions anatomiques : {n_anatomical_edges} arêtes, {n_connected_nodes}/{n_landmarks} nodes connectés")
        
        # 💡 CHOIX STRATÉGIQUE : Garder uniquement les connexions anatomiques ?
        # Option 1 : Graphe pur anatomique (157 arêtes) - plus "propre"
        # Option 2 : Ajouter k-NN circulaire (3025 arêtes) - couvre tout
        # Option 3 : Ajouter k-NN spatial (basé sur distance) - compromis intelligent
        
        use_knn = False  # Mettre à False pour graphe pur anatomique
        
        if use_knn and n_connected_nodes < n_landmarks:
            print(f"⚠️ Seulement {n_connected_nodes}/{n_landmarks} nodes connectés, ajout k-NN circulaire...")
            print(f"   Note: k-NN circulaire peut créer des connexions non-anatomiques")
            k = 6  # Connecter aux 6 voisins les plus proches (index)
            knn_edges = 0
            for i in range(n_landmarks):
                for j in range(1, k + 1):
                    neighbor = (i + j) % n_landmarks
                    edges.append([i, neighbor])
                    edges.append([neighbor, i])
                    knn_edges += 1
            print(f"✅ Graphe hybride : {n_anatomical_edges + knn_edges} arêtes totales (anatomique + k-NN)")
        else:
            print(f"✅ Graphe anatomique pur : {n_anatomical_edges} arêtes (160 nodes connectés sur 478)")
            print(f"   Note: 318 landmarks isolés seront traités indépendamment")
        
        # Convertir en tensor
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        
        return edge_index
    
class GraphConvolution(nn.Module):
    """
    Couche de convolution sur graphe (Graph Convolution Layer)
    Implémentation simple et efficace
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialisation Xavier"""
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Node features (batch_size, n_nodes, in_features)
            edge_index: Arêtes (2, num_edges)
        
        Returns:
            Node features transformées (batch_size, n_nodes, out_features)
        """
        batch_size, n_nodes, _ = x.shape
        
        # Transformation linéaire : X @ W
        x_transformed = torch.matmul(x, self.weight)  # (B, N, out_features)
        
        # Agrégation des voisins
        edge_src, edge_dst = edge_index[0], edge_index[1]
        
        # Initialiser la sortie
        out = torch.zeros_like(x_transformed)
        
        # Pour chaque batch
        for b in range(batch_size):
            # Matrice d'adjacence sparse
            adj = torch.zeros((n_nodes, n_nodes), device=x.device)
            adj[edge_src, edge_dst] = 1.0
            
            # Normalisation (degré)
            degree = adj.sum(dim=1, keepdim=True) + 1e-6
            adj_normalized = adj / degree
            
            # Agrégation : A_norm @ X
            out[b] = torch.matmul(adj_normalized, x_transformed[b])
        
        if self.bias is not None:
            out = out + self.bias
        
        return out


class GCNBlock(nn.Module):
    """
    Bloc GCN complet avec normalisation et activation
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        dropout: float = 0.1,
        use_residual: bool = True
    ):
        super().__init__()
        self.use_residual = use_residual and (in_features == out_features)
        
        self.conv = GraphConvolution(in_features, out_features)
        self.norm = nn.LayerNorm(out_features)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        
        if not self.use_residual and in_features != out_features:
            self.projection = nn.Linear(in_features, out_features)
        else:
            self.projection = None
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, n_nodes, in_features)
            edge_index: (2, num_edges)
        
        Returns:
            (batch_size, n_nodes, out_features)
        """
        identity = x
        
        # Graph convolution
        out = self.conv(x, edge_index)
        out = self.norm(out)
        out = self.activation(out)
        out = self.dropout(out)
        
        # Residual connection
        if self.use_residual:
            out = out + identity
        elif self.projection is not None:
            out = out + self.projection(identity)
        
        return out


class EmotionGNN(nn.Module):
    """
    🎯 MODÈLE GNN POUR CLASSIFICATION D'ÉMOTIONS À PARTIR DE LANDMARKS
    
    Architecture optimisée pour exploiter la structure de graphe facial.
    Meilleure que Transformer pour landmarks car respecte la topologie.
    
    Supporte FULL_FACE (478 landmarks)
    
    Performances attendues : 62-68% accuracy (vs 57.6% Transformer)
    """
    
    def __init__(
        self,
        input_dim: int = 1434,          # 1434 (478×3)
        n_classes: int = 7,            # 7 émotions
        n_landmarks: int = 478,        # 478 (full_face)
        hidden_dims: List[int] = [128, 256, 256, 128],  # Dimensions des couches GCN
        dropout: float = 0.15,         # Dropout
        pooling: str = 'mean',         # 'mean', 'max', ou 'attention'
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.n_landmarks = n_landmarks
        self.hidden_dims = hidden_dims
        self.pooling = pooling
        
        self.edge_index = FacialGraphConstructor.create_facial_graph(n_landmarks)
        self.register_buffer('graph_edges', self.edge_index)
        
        # Projection d'entrée : 3 coords → hidden_dim
        self.input_projection = nn.Linear(3, hidden_dims[0])
        
        # Couches de graph convolution
        self.gcn_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.gcn_layers.append(
                GCNBlock(
                    in_features=hidden_dims[i],
                    out_features=hidden_dims[i + 1],
                    dropout=dropout,
                    use_residual=True
                )
            )
        
        # Global pooling
        if pooling == 'attention':
            self.attention_pool = nn.Sequential(
                nn.Linear(hidden_dims[-1], 64),
                nn.Tanh(),
                nn.Linear(64, 1)
            )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
            nn.LayerNorm(hidden_dims[-1] // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[-1] // 2, n_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialisation Xavier"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Tensor de shape (batch_size, 1434) ou (batch_size, 478, 3)
        
        Returns:
            Logits de shape (batch_size, 7)
        """
        batch_size = x.shape[0]
        
        # Reshape si nécessaire : (batch, 1434) → (batch, 478, 3)
        if x.dim() == 2:
            x = x.view(batch_size, self.n_landmarks, 3)
        
        # Projection d'entrée : (batch, 478, 3) → (batch, 478, hidden_dim)
        x = self.input_projection(x)
        
        # Passer par les couches GCN
        edge_index = self.graph_edges
        for gcn_layer in self.gcn_layers:
            x = gcn_layer(x, edge_index)
        
        # Global pooling : (batch, 478, hidden_dim) → (batch, hidden_dim)
        if self.pooling == 'mean':
            x = x.mean(dim=1)
        elif self.pooling == 'max':
            x = x.max(dim=1)[0]
        elif self.pooling == 'attention':
            # Attention-based pooling
            attn_weights = self.attention_pool(x)  # (batch, 478, 1)
            attn_weights = F.softmax(attn_weights, dim=1)
            x = (x * attn_weights).sum(dim=1)  # (batch, hidden_dim)
        else:
            x = x.mean(dim=1)
        
        # Classification
        logits = self.classifier(x)
        
        return logits
    
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prédiction avec probabilités
        
        Args:
            x: Tensor de shape (batch_size, 1434)
        
        Returns:
            predictions: Indices des classes (batch_size,)
            probabilities: Probabilités softmax (batch_size, 7)
        """
        self.eval()
        with torch.no_grad():
            logits = self(x)
            probabilities = F.softmax(logits, dim=-1)
            predictions = torch.argmax(probabilities, dim=-1)
        
        return predictions, probabilities
    
    def get_num_parameters(self) -> int:
        """Retourne le nombre total de paramètres"""
        return sum(p.numel() for p in self.parameters())
    
    def get_trainable_parameters(self) -> int:
        """Retourne le nombre de paramètres entraînables"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class GCNBlockWithBatchNorm(nn.Module):
    """
    Bloc GCN avec BatchNorm pour stabilité (version corrigée)
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        dropout: float = 0.1,
        use_residual: bool = True
    ):
        super().__init__()
        self.use_residual = use_residual and (in_features == out_features)
        
        self.conv = GraphConvolution(in_features, out_features)
        # BatchNorm1d appliqué correctement : (batch * nodes, features)
        self.norm = nn.BatchNorm1d(out_features)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        
        if not self.use_residual and in_features != out_features:
            self.projection = nn.Linear(in_features, out_features)
        else:
            self.projection = None
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, n_nodes, in_features)
            edge_index: (2, num_edges)
        
        Returns:
            (batch_size, n_nodes, out_features)
        """
        identity = x
        batch_size, n_nodes, _ = x.shape
        
        # Graph convolution
        out = self.conv(x, edge_index)
        
        # BatchNorm : reshape pour (batch*nodes, features) → BatchNorm → reshape back
        out = out.view(batch_size * n_nodes, -1)  # (B*N, F)
        out = self.norm(out)                       # BatchNorm sur features
        out = out.view(batch_size, n_nodes, -1)   # (B, N, F)
        
        out = self.activation(out)
        out = self.dropout(out)
        
        # Residual connection
        if self.use_residual:
            out = out + identity
        elif self.projection is not None:
            out = out + self.projection(identity)
        
        return out


class EmotionGNNMedium(EmotionGNN):
    """
    ⭐ VERSION MEDIUM AVEC BATCHNORM (RECOMMANDÉ)
    
    Compromis optimal entre standard (4 couches) et deep (6 couches).
    Utilise BatchNorm pour stabiliser l'entraînement.
    
    Supporte FULL_FACE (478 landmarks)
    
    Différences :
    - 5 couches GCN (vs 4 standard, 6 deep)
    - BatchNorm correctement appliqué → stabilité
    - Dimensions [128, 256, 384, 256, 128]
    - Mean pooling (plus stable)
    
    Performances attendues : 50-60% accuracy
    Paramètres : ~600K
    """
    
    def __init__(
        self,
        input_dim: int = 1434,          # 478×3
        n_classes: int = 7,
        n_landmarks: int = 478,        # 478
        hidden_dims: List[int] = [128, 256, 384, 256, 128],
        dropout: float = 0.2,  # Plus élevé pour éviter overfitting
        pooling: str = 'mean'
    ):
        # Ne pas appeler super().__init__ directement car on veut remplacer les GCN blocks
        nn.Module.__init__(self)
        
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.n_landmarks = n_landmarks
        self.hidden_dims = hidden_dims
        self.pooling = pooling
        
        self.edge_index = FacialGraphConstructor.create_facial_graph(n_landmarks)
        self.register_buffer('graph_edges', self.edge_index)
        
        # Projection d'entrée : 3 coords → hidden_dim
        self.input_projection = nn.Linear(3, hidden_dims[0])
        
        # Couches de graph convolution AVEC BATCHNORM
        self.gcn_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.gcn_layers.append(
                GCNBlockWithBatchNorm(
                    in_features=hidden_dims[i],
                    out_features=hidden_dims[i + 1],
                    dropout=dropout,
                    use_residual=True
                )
            )
        
        # Global pooling
        if pooling == 'attention':
            self.attention_pool = nn.Sequential(
                nn.Linear(hidden_dims[-1], 64),
                nn.Tanh(),
                nn.Linear(64, 1)
            )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
            nn.LayerNorm(hidden_dims[-1] // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[-1] // 2, n_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialisation Xavier"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)


class EmotionGNNDeep(EmotionGNN):
    """
    ⚠️ VERSION DEEP (PEUT SOUFFRIR DE GRADIENT VANISHING)
    
    Version profonde avec 6 couches - à utiliser SEULEMENT si Medium fonctionne bien.
    
    Différences :
    - Plus de couches (6 vs 4)
    - Dimensions plus grandes [128, 256, 512, 512, 256, 128]
    - Attention pooling par défaut
    
    ⚠️ ATTENTION : Peut ne pas converger sans BatchNorm ou skip connections avancées
    
    Performances attendues : 65-70% accuracy (SI convergence)
    Paramètres : ~800K-1M
    """
    
    def __init__(
        self,
        input_dim: int = 1434,
        n_classes: int = 7,
        n_landmarks: int = 478,
        hidden_dims: List[int] = [128, 256, 512, 512, 256, 128],
        dropout: float = 0.15,
        pooling: str = 'attention'
    ):
        super().__init__(
            input_dim=input_dim,
            n_classes=n_classes,
            n_landmarks=n_landmarks,
            hidden_dims=hidden_dims,
            dropout=dropout,
            pooling=pooling
        )


class ModelManager:
    """
    Gestionnaire pour sauvegarder/charger les modèles GNN
    Compatible avec EmotionGNN et EmotionGNNDeep
    """
    
    @staticmethod
    def save_model(
        model: nn.Module,
        save_dir: str,
        model_name: str,
        metadata: dict = None
    ) -> dict:
        """
        Sauvegarder le modèle GNN avec ses métadonnées
        
        Args:
            model: Modèle PyTorch
            save_dir: Dossier de sauvegarde
            model_name: Nom du modèle
            metadata: Métadonnées additionnelles
        
        Returns:
            Dict avec les chemins des fichiers
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Métadonnées du modèle
        model_metadata = {
            'model_type': model.__class__.__name__,
            'input_dim': model.input_dim,
            'n_classes': model.n_classes,
            'n_landmarks': model.n_landmarks,
            'hidden_dims': model.hidden_dims,
            'pooling': model.pooling,
            'total_parameters': model.get_num_parameters(),
            'trainable_parameters': model.get_trainable_parameters()
        }
        
        if metadata:
            model_metadata.update(metadata)
        
        # Sauvegarder le modèle
        model_path = save_path / f"{model_name}.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'metadata': model_metadata
        }, model_path)
        
        # Sauvegarder les métadonnées en JSON
        metadata_path = save_path / f"{model_name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(model_metadata, f, indent=2)
        
        print(f"\n💾 Modèle GNN sauvegardé :")
        print(f"   Modèle    : {model_path}")
        print(f"   Metadata  : {metadata_path}")
        
        return {
            'model': str(model_path),
            'metadata': str(metadata_path)
        }
    
    @staticmethod
    def load_model(
        model_path: str,
        device: str = 'cpu',
        model_class = None
    ) -> nn.Module:
        """
        Charger un modèle GNN sauvegardé
        
        Args:
            model_path: Chemin vers le fichier .pth
            device: Device ('cpu' ou 'cuda')
            model_class: Classe du modèle (auto-détection si None)
        
        Returns:
            Modèle chargé
        """
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        metadata = checkpoint['metadata']
        
        print(f"\n📂 Chargement du modèle GNN : {model_path}")
        print(f"   Type      : {metadata['model_type']}")
        print(f"   Hidden    : {metadata['hidden_dims']}")
        print(f"   Pooling   : {metadata['pooling']}")
        print(f"   Parameters: {metadata['total_parameters']:,}")
        
        # Déterminer la classe
        if model_class is None:
            if metadata['model_type'] == 'EmotionGNN':
                model_class = EmotionGNN
            elif metadata['model_type'] == 'EmotionGNNDeep':
                model_class = EmotionGNNDeep
            elif metadata['model_type'] == 'EmotionGNNMedium':
                model_class = EmotionGNNMedium
            else:
                raise ValueError(f"Type de modèle inconnu : {metadata['model_type']}")
        
        # Créer le modèle
        model = model_class(
            input_dim=metadata['input_dim'],
            n_classes=metadata['n_classes'],
            n_landmarks=metadata['n_landmarks'],
            hidden_dims=metadata['hidden_dims'],
            pooling=metadata['pooling']
        )
        
        # Charger les poids
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        
        print(f"✅ Modèle chargé avec succès sur {device}")
        
        return model
    
    @staticmethod
    def train_model(
        model: nn.Module,
        train_loader,
        val_loader,
        test_loader,
        n_epochs: int,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        patience: int = 30,
        save_dir: str = "./models",
        model_name: str = "emotion_gnn"
    ) -> dict:
        """
        Entraîner un modèle GNN
        
        Args:
            model: Modèle à entraîner
            train_loader: DataLoader d'entraînement
            val_loader: DataLoader de validation
            test_loader: DataLoader de test
            n_epochs: Nombre d'epochs
            device: Device
            learning_rate: Learning rate
            weight_decay: Weight decay
            patience: Patience pour early stopping
            save_dir: Dossier de sauvegarde
            model_name: Nom du modèle
        
        Returns:
            Dict avec l'historique
        """
        from tqdm import tqdm
        import torch.optim as optim
        
        model = model.to(device)
        
        # Optimizer Adam
        optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=10
        )
        
        # Loss
        criterion = nn.CrossEntropyLoss()
        
        # Historique
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'test_loss': [],
            'test_acc': [],
            'learning_rates': []
        }
        
        best_val_acc = 0.0
        best_epoch = 0
        patience_counter = 0
        
        print("\n" + "="*80)
        print("🚀 DÉBUT DE L'ENTRAÎNEMENT GNN")
        print("="*80)
        print(f"Device         : {device}")
        print(f"Learning Rate  : {learning_rate}")
        print(f"Epochs         : {n_epochs}")
        print(f"Early Stopping : {patience} epochs")
        print(f"Model params   : {model.get_num_parameters():,}")
        print("="*80)
        
        for epoch in range(n_epochs):
            # ========== ENTRAÎNEMENT ==========
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            pbar = tqdm(train_loader, desc="Training", leave=False, ncols=100)
            
            for batch_x, batch_y in pbar:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                optimizer.zero_grad()
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * batch_x.size(0)
                predictions = torch.argmax(logits, dim=1)
                correct += (predictions == batch_y).sum().item()
                total += batch_y.size(0)
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100 * correct / total:.2f}%'
                })
            
            train_loss = running_loss / total
            train_acc = correct / total
            
            # ========== VALIDATION ==========
            model.eval()
            running_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    
                    logits = model(batch_x)
                    loss = criterion(logits, batch_y)
                    
                    running_loss += loss.item() * batch_x.size(0)
                    predictions = torch.argmax(logits, dim=1)
                    correct += (predictions == batch_y).sum().item()
                    total += batch_y.size(0)
            
            val_loss = running_loss / total
            val_acc = correct / total
            
            # ========== TEST ==========
            running_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch_x, batch_y in test_loader:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    
                    logits = model(batch_x)
                    loss = criterion(logits, batch_y)
                    
                    running_loss += loss.item() * batch_x.size(0)
                    predictions = torch.argmax(logits, dim=1)
                    correct += (predictions == batch_y).sum().item()
                    total += batch_y.size(0)
            
            test_loss = running_loss / total
            test_acc = correct / total
            
            # Scheduler step
            scheduler.step(val_acc)
            current_lr = optimizer.param_groups[0]['lr']
            
            # Sauvegarder l'historique
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['test_loss'].append(test_loss)
            history['test_acc'].append(test_acc)
            history['learning_rates'].append(current_lr)
            
            # Affichage
            print(f"\n📊 Epoch {epoch + 1}/{n_epochs}")
            print(f"   Train : Loss = {train_loss:.4f}, Acc = {train_acc*100:.2f}%")
            print(f"   Val   : Loss = {val_loss:.4f}, Acc = {val_acc*100:.2f}%")
            print(f"   Test  : Loss = {test_loss:.4f}, Acc = {test_acc*100:.2f}%")
            print(f"   LR    : {current_lr:.6f}")
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                patience_counter = 0
                
                # Métadonnées
                model_metadata = {
                    'model_type': model.__class__.__name__,
                    'input_dim': model.input_dim,
                    'n_classes': model.n_classes,
                    'n_landmarks': model.n_landmarks,
                    'hidden_dims': model.hidden_dims,
                    'pooling': model.pooling,
                    'total_parameters': model.get_num_parameters(),
                    'trainable_parameters': model.get_trainable_parameters(),
                    'best_epoch': epoch + 1,
                    'best_val_acc': float(best_val_acc),
                    'train_acc': float(train_acc),
                    'test_acc': float(test_acc)
                }
                
                # Sauvegarder
                save_path = Path(save_dir) / f"{model_name}.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_acc': best_val_acc,
                    'history': history,
                    'metadata': model_metadata
                }, save_path)
                
                print(f"   ✅ Nouveau meilleur modèle sauvegardé ! Val Acc = {val_acc*100:.2f}%")
            else:
                patience_counter += 1
                print(f"   ⏳ Patience : {patience_counter}/{patience}")
            
            if patience_counter >= patience:
                print(f"\n🛑 Early stopping à l'epoch {epoch + 1}")
                print(f"   Meilleure Val Acc : {best_val_acc*100:.2f}% (epoch {best_epoch + 1})")
                break
        
        print("\n" + "="*80)
        print("🎉 ENTRAÎNEMENT TERMINÉ")
        print("="*80)
        print(f"Meilleure Val Acc : {best_val_acc*100:.2f}% (epoch {best_epoch + 1})")
        print("="*80)
        
        return history
    
    @staticmethod
    def plot_history(history: dict, save_path: str):
        """
        Tracer les courbes d'entraînement
        
        Args:
            history: Historique d'entraînement
            save_path: Chemin de sauvegarde
        """
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        axes[0, 0].plot(history['train_loss'], label='Train Loss')
        axes[0, 0].plot(history['val_loss'], label='Val Loss')
        axes[0, 0].plot(history['test_loss'], label='Test Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Loss Evolution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[0, 1].plot([acc * 100 for acc in history['train_acc']], label='Train Acc')
        axes[0, 1].plot([acc * 100 for acc in history['val_acc']], label='Val Acc')
        axes[0, 1].plot([acc * 100 for acc in history['test_acc']], label='Test Acc')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].set_title('Accuracy Evolution')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Learning Rate
        axes[1, 0].plot(history['learning_rates'])
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Val vs Test
        axes[1, 1].plot([acc * 100 for acc in history['val_acc']], label='Val Acc')
        axes[1, 1].plot([acc * 100 for acc in history['test_acc']], label='Test Acc')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy (%)')
        axes[1, 1].set_title('Val vs Test Accuracy')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Courbes sauvegardées : {save_path}")


def predict_emotion_gnn(
    image_path: str,
    model: nn.Module,
    device: str = 'cuda'
) -> dict:
    """
    Prédire l'émotion avec un modèle GNN
    
    Args:
        image_path: Chemin vers l'image
        model: Modèle GNN
        device: 'cpu' ou 'cuda'
    
    Returns:
        Dict avec la prédiction
    """
    import cv2
    from classes.FaceLandmarkExtractor import FaceLandmarkExtractor
    
    emotion_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    
    # Charger l'image
    image = cv2.imread(image_path)
    if image is None:
        return {'success': False, 'error': f"Impossible de charger l'image : {image_path}"}
    
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
    
    normalized_landmarks = extractor.normalize_face_part_landmarks(landmarks)
    features = extractor.landmarks_to_features(normalized_landmarks)
    
    # Convertir en tensor
    features_tensor = torch.from_numpy(features).float().unsqueeze(0).to(device)
    
    # Prédiction
    predictions, probabilities = model.predict(features_tensor)
    
    pred_class = predictions.item()
    pred_probs = probabilities.cpu().numpy()[0]
    
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