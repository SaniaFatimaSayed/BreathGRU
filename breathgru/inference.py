"""
BreathGRU Inference Module

Loads a pretrained BreathGRU model and performs speech–breath
segmentation on an audio recording.
"""

import numpy as np
import torch

from .config import HIDDEN, HOP_MS, INPUT_DIM
from .model import BreathGRUSegmenter
from .features import extract_features
from .decoder import segmental_viterbi, collapse_segments
from .utils import load_duration_priors


class BreathGRU:
    """
    BreathGRU inference class.

    Example
    -------
    >>> from breathgru import BreathGRU
    >>> model = BreathGRU(
    ...     "pretrained/breathgru.pt",
    ...     "pretrained/duration_priors.json"
    ... )
    >>> segments = model.predict("example.wav")
    """

    def __init__(
        self,
        checkpoint_path,
        duration_priors_path,
        device="cpu",
    ):
        self.device = torch.device(device)

        # Load duration priors
        self.dur_priors = load_duration_priors(duration_priors_path)

        # Transition probabilities
        transition = np.array(
            [
                [0.90, 0.10],
                [0.10, 0.90],
            ],
            dtype=np.float64,
        )

        self.log_trans = np.log(transition + 1e-12)

        # Build model
        self.model = BreathGRUSegmenter(
            in_dim=INPUT_DIM,
            hidden=HIDDEN,
        )

        # Load checkpoint
        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
        )

        # Support both checkpoint formats
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]

        self.model.load_state_dict(checkpoint)

        self.model.to(self.device)
        self.model.eval()

    def predict(self, audio_path):
        """
        Predict speech and breath segments.

        Parameters
        ----------
        audio_path : str or Path
            Path to the audio recording.

        Returns
        -------
        list
            List of tuples:
            (segment_type, start_time, end_time, duration)
        """

        # Feature extraction
        features = extract_features(audio_path)

        x = (
            torch.tensor(features, dtype=torch.float32)
            .unsqueeze(0)
            .to(self.device)
        )

        with torch.no_grad():
            logits = self.model(x)[0]
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

        labels = segmental_viterbi(
            np.log(probs + 1e-12),
            self.dur_priors,
            self.log_trans,
        )

        segments = collapse_segments(
            labels,
            HOP_MS / 1000.0,
        )

        return segments
