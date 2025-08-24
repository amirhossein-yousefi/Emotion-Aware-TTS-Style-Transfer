# Minimal style adaptor & fusion blocks for Emotion-aware TTS style transfer.

from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class StyleAdaptor(nn.Module):
    """
    Turns a pooled SSL (WavLM) embedding into a compact style latent.
    Also predicts emotion logits for an auxiliary loss.
    """
    def __init__(self, in_dim: int = 768, latent_dim: int = 256,
                 num_emotions: int = 5, dropout: float = 0.1):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, 512),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(512, latent_dim),
            nn.SiLU(),
        )
        self.emotion_head = nn.Linear(latent_dim, num_emotions)

    def forward(self, style_vec: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            style_vec: [B, in_dim] pooled WavLM features (mean pooled).
        Returns:
            latent: [B, latent_dim]
            logits: [B, num_emotions]
        """
        latent = self.backbone(style_vec)
        logits = self.emotion_head(latent)
        return latent, logits


class StyleSpeakerFusion(nn.Module):
    """
    Fuses speaker x-vector (512) with style latent (e.g., 256) to SpeechT5's expected 512-D.
    """
    def __init__(self, spk_dim: int = 512, style_dim: int = 256, out_dim: int = 512):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(spk_dim + style_dim, 512),
            nn.SiLU(),
            nn.Linear(512, out_dim),
        )
        self.out_dim = out_dim

    def forward(self, spk_vec: torch.Tensor, style_latent: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spk_vec: [B, spk_dim]
            style_latent: [B, style_dim]
        Returns:
            fused: [B, out_dim] -> feed into SpeechT5 speaker_embeddings
        """
        fused = torch.cat([spk_vec, style_latent], dim=-1)
        return self.proj(fused)


@dataclass
class LossWeights:
    emo_ce: float = 0.2  # weight for emotion aux loss
