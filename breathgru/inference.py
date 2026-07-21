from pathlib import Path

import numpy as np
import torch

from .config import HIDDEN, HOP_MS
from .model import BreathGRUSegmenter
from .features import extract_features
from .decoder import segmental_viterbi, collapse_segments
from .utils import load_duration_priors


class BreathGRU:
   
    #BreathGRU inference class.

    model = BreathGRU(
      "pretrained/breathgru.pt",
      "pretrained/duration_priors.json"
     )
    ## segments = model.predict("example.wav")
  

    def __init__(
        self,
        checkpoint_path,
        duration_priors_path,
        device="cpu"
    ):

        self.device = torch.device(device)

        self.dur_priors = load_duration_priors(duration_priors_path)

        # Transition matrix
        trans = np.array([
            [0.90, 0.10],
            [0.10, 0.90]
        ])

        self.log_trans = np.log(trans + 1e-12)

        # Build model
        dummy = extract_features(Path(__file__).parent.parent /
                                 "examples/example.wav")

        in_dim = dummy.shape[1]

        self.model = BreathGRUSegmenter(
            in_dim=in_dim,
            hidden=HIDDEN
        )

        state = torch.load(
            checkpoint_path,
            map_location=self.device
        )

        self.model.load_state_dict(state)

        self.model.to(self.device)

        self.model.eval()

    def predict(self, audio_path):

        features = extract_features(audio_path)

        x = (
            torch.tensor(features)
            .float()
            .unsqueeze(0)
            .to(self.device)
        )

        with torch.no_grad():

            logits = self.model(x)[0]

            probs = torch.softmax(
                logits,
                dim=-1
            ).cpu().numpy()

        labels = segmental_viterbi(
            np.log(probs + 1e-12),
            self.dur_priors,
            self.log_trans
        )

        return collapse_segments(
            labels,
            HOP_MS / 1000.0
        )
