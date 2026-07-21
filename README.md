# BreathGRU

**BreathGRU** is a semi-supervised Bidirectional Gated Recurrent Unit (BiGRU) framework for automatic speech–breath segmentation from respiratory audio recordings.

Unlike general-purpose Voice Activity Detection (VAD) systems, BreathGRU is specifically designed to detect respiratory events by combining frame-level acoustic modelling with duration-constrained sequence decoding. The framework was developed for pulmonary health applications, including digital biomarker extraction, lung function prediction, and respiratory disease monitoring.

---

## Features

- Semi-supervised BiGRU architecture for speech–breath segmentation
- Temporal modelling using bidirectional recurrent networks
- Duration-aware Segmental Viterbi decoding
- Robust segmentation of speech and breath intervals
- Lightweight PyTorch implementation
- Designed for respiratory healthcare research
- Compatible with external respiratory audio datasets

---

## Repository Structure

```
BreathGRU/
│
├── breathgru/
│   ├── __init__.py
│   ├── config.py
│   ├── model.py
│   ├── features.py
│   ├── decoder.py
│   ├── inference.py
│   └── utils.py
│
├── pretrained/
│   ├── breathgru.pt
│   └── duration_priors.json
│
├── examples/
│   ├── example.wav
│   └── run_inference.py
│
├── docs/
│
├── tests/
│
├── requirements.txt
├── LICENSE
├── CITATION.cff
└── README.md
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/<username>/BreathGRU.git
cd BreathGRU
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Quick Start

```python
from breathgru import BreathGRU

model = BreathGRU(
    checkpoint_path="pretrained/breathgru.pt",
    duration_priors_path="pretrained/duration_priors.json"
)

segments = model.predict("examples/example.wav")

print(segments)
```

or from the command line

```bash
python examples/run_inference.py \
    --audio examples/example.wav \
    --model pretrained/breathgru.pt \
    --priors pretrained/duration_priors.json
```

---

## Output

The model returns a list of segmented speech and breath intervals.

Example

```text
[
    ('speech', 0.00, 2.31, 2.31),
    ('breath', 2.31, 2.86, 0.55),
    ('speech', 2.86, 5.02, 2.16),
    ('breath', 5.02, 5.61, 0.59)
]
```

Each tuple contains

- Segment label
- Start time (seconds)
- End time (seconds)
- Duration (seconds)

---

## Method Overview

The BreathGRU pipeline consists of the following stages:

```
Audio Recording
        │
        ▼
Feature Extraction
(MFCC + Δ + ΔΔ + Mel Spectrogram)
        │
        ▼
BiGRU Network
(Frame-wise Speech/Breath Probabilities)
        │
        ▼
Segmental Viterbi Decoder
(Duration Constraints)
        │
        ▼
Speech–Breath Segments
```

---

## Model Architecture

The framework consists of

- Acoustic feature extraction
- Two-layer Bidirectional GRU
- Fully connected classification layer
- Softmax prediction
- Duration-aware Segmental Viterbi decoding

An architecture diagram is available in the accompanying publication.

---

## Applications

BreathGRU can be used for

- Respiratory audio preprocessing
- Breath cycle detection
- Pulmonary digital biomarker extraction
- Lung function prediction
- Asthma monitoring
- COPD monitoring
- Clinical speech analysis

---

## Citation

If you use BreathGRU in your research, please cite

```bibtex
@article{sayed2026breathgru,
  title={BreathGRU: A Semi-Supervised BiGRU Framework for Speech--Breath Segmentation in Respiratory Audio},
  author={Sayed, S.F, Holloway, J.W., Zwiggelaar, R., Rezwan, F.I.},
  journal={...},
  year={2026}
}
```

---

## License

This project is released under the MIT License.

---

## Acknowledgements

This work was developed as part of the **Pulmono.AI** project at **Aberystwyth University**, investigating speech and breathing as digital biomarkers for pulmonary health assessment.

---

## Contact

For questions, bug reports or collaborations, please open an issue or submit a pull request.
