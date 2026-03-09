"""
anomaly_model.py — OpenVAD-inspired architecture for video anomaly detection.

Implements the key components of the "Towards Open Set Video Anomaly Detection"
framework (ECCV 2022) from scratch using PyTorch, without importing any
external OpenVAD library.

Architecture Components:
  1. FeatureExtractor   — CNN backbone that converts person crops into features.
  2. GraphConvolution    — Single GCN layer: H' = σ(A·H·W).
  3. GCNFeatureExtractor — Two stacked GCN layers for relational feature learning.
  4. PlanarFlow          — Single planar normalizing flow transformation.
  5. NormalizingFlow     — Stacked planar flows for pseudo-anomaly generation.
  6. EDLClassifier       — Evidential Deep Learning head for uncertainty-aware
                           classification (outputs Dirichlet parameters).
  7. OpenVADModel        — Top-level model composing all components.

Anomaly Scoring:
  Instead of reconstruction error (MSE), the anomaly score is derived from the
  EDL uncertainty: higher uncertainty = more likely anomalous.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import config


# =============================================================================
# 1. CNN FEATURE EXTRACTOR
# =============================================================================
class FeatureExtractor(nn.Module):
    """
    CNN backbone that converts person crop images into flat feature vectors.

    Reuses the convolutional encoder design from the original autoencoder,
    followed by adaptive pooling and a projection to FEATURE_DIM.

    Input:  (batch, 3, 128, 64)
    Output: (batch, FEATURE_DIM)
    """

    def __init__(self, feature_dim=None):
        super(FeatureExtractor, self).__init__()
        if feature_dim is None:
            feature_dim = config.FEATURE_DIM

        self.conv_layers = nn.Sequential(
            # (3, 128, 64) → (32, 64, 32)
            nn.Conv2d(config.INPUT_CHANNELS, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            # (32, 64, 32) → (64, 32, 16)
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            # (64, 32, 16) → (128, 16, 8)
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # (128, 16, 8) → (256, 8, 4)
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Adaptive pooling to fixed size regardless of input resolution
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Project to feature_dim
        self.fc = nn.Linear(256, feature_dim)
        self.bn = nn.BatchNorm1d(feature_dim)

    def forward(self, x):
        """
        Args:
            x: (batch, 3, 128, 64) — person crop tensors.
        Returns:
            (batch, feature_dim) — flat feature vectors.
        """
        x = self.conv_layers(x)
        x = self.adaptive_pool(x)       # (batch, 256, 1, 1)
        x = x.view(x.size(0), -1)       # (batch, 256)
        x = self.fc(x)                   # (batch, feature_dim)
        x = self.bn(x)
        return x


# =============================================================================
# 2. GRAPH CONVOLUTION LAYER
# =============================================================================
class GraphConvolution(nn.Module):
    """
    Single Graph Convolutional Network layer.

    Implements: H' = σ(A · H · W + b)
    where A is the adjacency matrix, H is the node feature matrix,
    and W is a learnable weight matrix.
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize parameters using Xavier uniform initialization."""
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, h, adj):
        """
        Args:
            h:   (num_nodes, in_features)  — node feature matrix.
            adj: (num_nodes, num_nodes)    — adjacency matrix (normalized).
        Returns:
            (num_nodes, out_features) — updated node features.
        """
        # H · W
        support = torch.mm(h, self.weight)
        # A · (H · W)
        output = torch.mm(adj, support)
        if self.bias is not None:
            output = output + self.bias
        return output


# =============================================================================
# 3. GCN FEATURE EXTRACTOR (two stacked GCN layers)
# =============================================================================
class GCNFeatureExtractor(nn.Module):
    """
    Two-layer GCN for refining instance-level features by modeling
    relationships between instances in a bag (e.g., persons in a video segment).

    The adjacency matrix is computed dynamically from feature similarity
    (cosine similarity → softmax normalization).
    """

    def __init__(self, feature_dim=None, hidden_dim=None):
        super(GCNFeatureExtractor, self).__init__()
        if feature_dim is None:
            feature_dim = config.FEATURE_DIM
        if hidden_dim is None:
            hidden_dim = config.GCN_HIDDEN_DIM

        self.gc1 = GraphConvolution(feature_dim, hidden_dim)
        self.gc2 = GraphConvolution(hidden_dim, feature_dim)
        self.dropout = nn.Dropout(0.3)

    @staticmethod
    def compute_adjacency(features):
        """
        Compute a normalized adjacency matrix from feature similarity.

        Uses cosine similarity between all pairs of features, then applies
        softmax row-wise to get a stochastic adjacency matrix.

        Args:
            features: (N, D) — feature matrix for N instances.
        Returns:
            (N, N) — normalized adjacency matrix.
        """
        # Normalize features for cosine similarity
        norm_features = F.normalize(features, p=2, dim=1)
        # Cosine similarity matrix
        sim = torch.mm(norm_features, norm_features.t())
        # Softmax normalization (row-wise) to get a stochastic matrix
        adj = F.softmax(sim, dim=1)
        return adj

    def forward(self, features):
        """
        Args:
            features: (N, feature_dim) — instance features from FeatureExtractor.
        Returns:
            (N, feature_dim) — GCN-refined features.
        """
        # For single instance, GCN is a no-op (identity with self-loop)
        if features.size(0) == 1:
            return features

        # Compute dynamic adjacency matrix
        adj = self.compute_adjacency(features)

        # Two-layer GCN with residual connection
        h = F.leaky_relu(self.gc1(features, adj), 0.2)
        h = self.dropout(h)
        h = self.gc2(h, adj)

        # Residual connection
        h = h + features
        return h


# =============================================================================
# 4. PLANAR NORMALIZING FLOW
# =============================================================================
class PlanarFlow(nn.Module):
    """
    Single planar normalizing flow transformation.

    Transforms z → z' = z + u · tanh(w^T z + b)

    This is one of the simplest normalizing flow forms, suitable for
    learning simple density transformations.
    """

    def __init__(self, dim):
        super(PlanarFlow, self).__init__()
        self.u = nn.Parameter(torch.randn(1, dim) * 0.01)
        self.w = nn.Parameter(torch.randn(1, dim) * 0.01)
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, z):
        """
        Apply the planar flow transformation.

        Args:
            z: (batch, dim) — input latent vectors.
        Returns:
            z_prime: (batch, dim) — transformed vectors.
            log_det: (batch,) — log determinant of the Jacobian.
        """
        # w^T z + b → (batch, 1)
        linear = torch.mm(z, self.w.t()) + self.b
        # z' = z + u · tanh(w^T z + b)
        z_prime = z + self.u * torch.tanh(linear)

        # Log determinant of Jacobian: log|1 + u^T ψ(z)|
        # where ψ(z) = (1 - tanh²(w^T z + b)) · w
        psi = (1.0 - torch.tanh(linear) ** 2) * self.w  # (batch, dim)
        det = 1.0 + torch.mm(psi, self.u.t())             # (batch, 1)
        log_det = torch.log(torch.abs(det) + 1e-8).squeeze(1)

        return z_prime, log_det


class NormalizingFlow(nn.Module):
    """
    Stack of planar normalizing flows for learning the normal data distribution
    and generating pseudo-anomaly samples.

    The flow is trained to map normal features to a base Gaussian distribution.
    Pseudo-anomalies are generated by sampling from the tails of this distribution.
    """

    def __init__(self, feature_dim=None, num_flows=None):
        super(NormalizingFlow, self).__init__()
        if feature_dim is None:
            feature_dim = config.FEATURE_DIM
        if num_flows is None:
            num_flows = config.NF_NUM_FLOWS

        self.flows = nn.ModuleList([PlanarFlow(feature_dim) for _ in range(num_flows)])
        self.feature_dim = feature_dim

    def forward(self, z):
        """
        Pass features through the normalizing flow chain.

        Args:
            z: (batch, feature_dim) — input features.
        Returns:
            z_k: (batch, feature_dim) — transformed features.
            sum_log_det: (batch,) — sum of log-determinants across all flows.
        """
        sum_log_det = torch.zeros(z.size(0), device=z.device)
        z_k = z
        for flow in self.flows:
            z_k, log_det = flow(z_k)
            sum_log_det += log_det
        return z_k, sum_log_det

    def log_prob(self, z):
        """
        Compute the log-probability of features under the learned distribution.

        Maps z through the flow, then evaluates under a standard Gaussian base.

        Args:
            z: (batch, feature_dim) — input features.
        Returns:
            log_prob: (batch,) — log-probability of each input.
        """
        z_k, sum_log_det = self.forward(z)
        # Log-probability under standard Gaussian base distribution
        log_p_base = -0.5 * (z_k ** 2 + math.log(2 * math.pi)).sum(dim=1)
        return log_p_base + sum_log_det

    def generate_pseudo_anomalies(self, normal_features, scale=3.0):
        """
        Generate pseudo-anomaly features by perturbing normal features
        in the flow-transformed space.

        Strategy: Map normal features through the flow, add noise scaled
        to push them into the tails of the distribution, then invert
        (approximately) by using the noisy transformed features directly
        as pseudo-anomalous instances in feature space.

        Args:
            normal_features: (batch, feature_dim) — normal features.
            scale: float — scale of perturbation (higher = more anomalous).
        Returns:
            pseudo_anomalies: (batch, feature_dim) — pseudo-anomaly features.
        """
        with torch.no_grad():
            z_k, _ = self.forward(normal_features)
            # Perturb in the transformed space (tail sampling)
            noise = torch.randn_like(z_k) * scale
            perturbed = z_k + noise
        return perturbed


# =============================================================================
# 5. EVIDENTIAL DEEP LEARNING (EDL) CLASSIFIER
# =============================================================================
class EDLClassifier(nn.Module):
    """
    Evidential Deep Learning classifier that outputs Dirichlet concentration
    parameters (evidence) instead of softmax probabilities.

    The uncertainty is computed as:  u = num_classes / (total_evidence + num_classes)

    Higher uncertainty → more likely an unknown/anomalous instance.
    """

    def __init__(self, feature_dim=None, num_classes=None):
        super(EDLClassifier, self).__init__()
        if feature_dim is None:
            feature_dim = config.FEATURE_DIM
        if num_classes is None:
            num_classes = config.EDL_NUM_CLASSES

        self.num_classes = num_classes
        self.fc1 = nn.Linear(feature_dim, feature_dim // 2)
        self.fc2 = nn.Linear(feature_dim // 2, feature_dim // 4)
        self.evidence_layer = nn.Linear(feature_dim // 4, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, features):
        """
        Compute evidence (Dirichlet parameters) from features.

        Args:
            features: (batch, feature_dim) — input features.
        Returns:
            evidence: (batch, num_classes) — non-negative evidence values.
            alpha:    (batch, num_classes) — Dirichlet concentrations (evidence + 1).
            uncertainty: (batch,) — epistemic uncertainty for each instance.
        """
        x = F.leaky_relu(self.fc1(features), 0.2)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = self.dropout(x)

        # Evidence must be non-negative → use softplus activation
        evidence = F.softplus(self.evidence_layer(x))

        # Dirichlet concentration parameters: α = evidence + 1
        alpha = evidence + 1.0

        # Dirichlet strength (total evidence)
        S = alpha.sum(dim=1, keepdim=True)

        # Epistemic uncertainty: u = K / S  where K = num_classes
        uncertainty = (self.num_classes / S).squeeze(1)

        return evidence, alpha, uncertainty


# =============================================================================
# 6. OPEN-VAD MODEL (Top-level model)
# =============================================================================
class OpenVADModel(nn.Module):
    """
    OpenVAD-inspired model for video anomaly detection.

    Composes:
      - FeatureExtractor:     CNN backbone for crop → feature conversion.
      - GCNFeatureExtractor:  Two-layer GCN for relational feature refinement.
      - NormalizingFlow:      Planar flows for density estimation & pseudo-anomaly generation.
      - EDLClassifier:        Evidential DL head for uncertainty-aware classification.

    Input:  (batch, 3, 128, 64) — person crop tensors.
    Output: evidence, alpha, uncertainty — EDL outputs.
    """

    def __init__(self):
        super(OpenVADModel, self).__init__()
        self.feature_extractor = FeatureExtractor()
        self.gcn = GCNFeatureExtractor()
        self.normalizing_flow = NormalizingFlow()
        self.edl_classifier = EDLClassifier()

    def extract_features(self, x):
        """
        Extract GCN-refined features from person crop images.

        Args:
            x: (batch, 3, 128, 64) — person crop tensors.
        Returns:
            (batch, feature_dim) — refined feature vectors.
        """
        cnn_features = self.feature_extractor(x)
        gcn_features = self.gcn(cnn_features)
        return gcn_features

    def forward(self, x):
        """
        Full forward pass: CNN → GCN → EDL.

        Args:
            x: (batch, 3, 128, 64) — person crop tensors.
        Returns:
            evidence:    (batch, num_classes)
            alpha:       (batch, num_classes)
            uncertainty: (batch,) — anomaly score (higher = more anomalous).
        """
        features = self.extract_features(x)
        evidence, alpha, uncertainty = self.edl_classifier(features)
        return evidence, alpha, uncertainty

    def compute_nf_log_prob(self, x):
        """
        Compute normalizing flow log-probability for input features.

        Args:
            x: (batch, 3, 128, 64) — person crop tensors.
        Returns:
            log_prob: (batch,) — log-probability under the learned NF distribution.
        """
        features = self.extract_features(x)
        return self.normalizing_flow.log_prob(features)

    def generate_pseudo_anomalies(self, x, scale=3.0):
        """
        Generate pseudo-anomaly features from normal input crops.

        Args:
            x: (batch, 3, 128, 64) — normal person crop tensors.
            scale: float — perturbation scale.
        Returns:
            pseudo_features: (batch, feature_dim) — pseudo-anomaly features.
        """
        with torch.no_grad():
            features = self.feature_extractor(x)
        return self.normalizing_flow.generate_pseudo_anomalies(features, scale)


# =============================================================================
# BACKWARD COMPATIBILITY ALIAS
# =============================================================================
# Keep the old name so that existing imports still work
AnomalyAutoencoder = OpenVADModel


# =============================================================================
# EDL LOSS FUNCTIONS
# =============================================================================
def edl_mse_loss(alpha, target, epoch, num_classes=None, annealing_epochs=None):
    """
    Compute the EDL loss: MSE-based Dirichlet loss + KL divergence regularization.

    The MSE loss encourages correct classification through the Dirichlet posterior.
    The KL term regularizes the evidence for incorrect classes to be small.

    Args:
        alpha:  (batch, K) — Dirichlet concentration parameters.
        target: (batch, K) — one-hot encoded target labels.
        epoch:  int — current training epoch (for KL annealing).
        num_classes: int — number of classes.
        annealing_epochs: int — epochs over which to anneal KL weight.

    Returns:
        loss: scalar tensor — the combined EDL loss.
    """
    if num_classes is None:
        num_classes = config.EDL_NUM_CLASSES
    if annealing_epochs is None:
        annealing_epochs = config.EDL_ANNEALING_EPOCHS

    S = alpha.sum(dim=1, keepdim=True)  # Dirichlet strength

    # Expected probability under the Dirichlet: p_k = α_k / S
    prob = alpha / S

    # MSE loss with Dirichlet variance correction
    err = (target - prob) ** 2
    var = alpha * (S - alpha) / (S * S * (S + 1))
    mse_loss = (err + var).sum(dim=1).mean()

    # KL divergence term: regularize evidence for non-target classes
    # Anneal the KL weight from 0 → 1 over annealing_epochs
    kl_weight = min(1.0, epoch / max(annealing_epochs, 1))

    # Remove evidence from the target class for KL computation
    alpha_tilde = target + (1.0 - target) * (alpha - 1.0) + 1.0
    kl = kl_divergence_dirichlet(alpha_tilde, num_classes)
    kl_loss = kl.mean()

    return mse_loss + kl_weight * kl_loss


def kl_divergence_dirichlet(alpha, num_classes):
    """
    Compute KL divergence between Dirichlet(alpha) and Dirichlet(1, ..., 1).

    Args:
        alpha: (batch, K) — Dirichlet concentration parameters.
        num_classes: int — number of classes K.
    Returns:
        kl: (batch,) — KL divergence for each sample.
    """
    ones = torch.ones_like(alpha)
    S_alpha = alpha.sum(dim=1, keepdim=True)
    S_ones = ones.sum(dim=1, keepdim=True)

    ln_B_alpha = torch.lgamma(alpha).sum(dim=1, keepdim=True) - torch.lgamma(S_alpha)
    ln_B_ones = torch.lgamma(ones).sum(dim=1, keepdim=True) - torch.lgamma(S_ones)

    dg_S = torch.digamma(S_alpha)
    dg_alpha = torch.digamma(alpha)

    kl = (ln_B_alpha - ln_B_ones +
          ((alpha - ones) * (dg_alpha - dg_S)).sum(dim=1, keepdim=True))
    return kl.squeeze(1)


def mil_ranking_loss(normal_scores, anomaly_scores, top_k_ratio=None):
    """
    Multiple Instance Learning ranking loss.

    Selects the top-k instances from each bag (normal and anomaly) and applies
    a hinge-based ranking loss that encourages anomaly instances to have higher
    uncertainty scores than normal instances.

    Args:
        normal_scores:  (N,) — uncertainty scores for normal instances.
        anomaly_scores: (M,) — uncertainty scores for anomaly instances.
        top_k_ratio:    float — fraction of instances to select as top-k.
    Returns:
        loss: scalar tensor — MIL ranking loss.
    """
    if top_k_ratio is None:
        top_k_ratio = config.MIL_TOP_K_RATIO

    # Select top-k from each bag
    k_normal = max(1, int(len(normal_scores) * top_k_ratio))
    k_anomaly = max(1, int(len(anomaly_scores) * top_k_ratio))

    # Top-k normal scores (should be LOW → select highest as hard examples)
    top_normal, _ = torch.topk(normal_scores, k_normal)
    # Top-k anomaly scores (should be HIGH → select highest)
    top_anomaly, _ = torch.topk(anomaly_scores, k_anomaly)

    # Ranking loss: max(0, 1 - (anomaly_score - normal_score))
    # Average over all pairs of top-k normal and anomaly instances
    loss = torch.clamp(
        1.0 - (top_anomaly.mean() - top_normal.mean()),
        min=0.0
    )

    # Smoothness regularization: encourage consistent scores within bags
    smooth_normal = (top_normal[1:] - top_normal[:-1]).pow(2).mean() if k_normal > 1 else 0.0
    smooth_anomaly = (top_anomaly[1:] - top_anomaly[:-1]).pow(2).mean() if k_anomaly > 1 else 0.0

    return loss + 0.1 * (smooth_normal + smooth_anomaly)


def triplet_loss(anchor, positive, negative, margin=None):
    """
    Triplet loss for discriminative feature learning.

    Encourages features of the same class to be closer together
    and features of different classes to be farther apart.

    Args:
        anchor:   (batch, dim) — anchor features.
        positive: (batch, dim) — same-class features.
        negative: (batch, dim) — different-class features.
        margin:   float — minimum desired separation.
    Returns:
        loss: scalar tensor.
    """
    if margin is None:
        margin = config.TRIPLET_MARGIN

    loss_fn = nn.TripletMarginLoss(margin=margin, p=2)
    return loss_fn(anchor, positive, negative)


# =============================================================================
# ANOMALY SCORING
# =============================================================================
def compute_anomaly_score(model, crop_tensor, device=None):
    """
    Compute the anomaly score for a single cropped person region.

    The anomaly score is the EDL uncertainty: a value in [0, 1] where
    higher values indicate more anomalous (uncertain) instances.

    Args:
        model (OpenVADModel): Trained OpenVAD model.
        crop_tensor (torch.Tensor): Preprocessed person crop tensor (C, H, W).
        device (torch.device): Device to run computation on.

    Returns:
        float: Anomaly score (EDL uncertainty, higher = more anomalous).
    """
    if device is None:
        device = config.DEVICE

    model.eval()
    with torch.no_grad():
        # Add batch dimension: (C, H, W) → (1, C, H, W)
        input_tensor = crop_tensor.unsqueeze(0).to(device)

        # Forward pass through the OpenVAD model
        evidence, alpha, uncertainty = model(input_tensor)

        # The anomaly score is the EDL uncertainty
        score = uncertainty.item()

    return score


# =============================================================================
# MODEL LOADING
# =============================================================================
def load_model(model_path=None):
    """
    Load a trained OpenVAD model from disk.

    Args:
        model_path (str): Path to the saved model weights.
                          Defaults to config.MODEL_SAVE_PATH.

    Returns:
        OpenVADModel: The loaded model in eval mode on the configured device.
    """
    if model_path is None:
        model_path = config.MODEL_SAVE_PATH

    model = OpenVADModel().to(config.DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
    model.eval()
    print(f"[OpenVAD] Loaded model from: {model_path}")
    return model
