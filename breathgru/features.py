def extract_features(path: Path) -> np.ndarray:
    try:
        y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    except Exception as e:
        raise RuntimeError(f"Could not read audio file {path}: {e}")

    if y is None or len(y) == 0:
        raise RuntimeError(f"Audio file is empty: {path}")

    n_fft = max(256, int(SAMPLE_RATE * FRAME_MS / 1000.0))
    hop = max(1, int(SAMPLE_RATE * HOP_MS / 1000.0))

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=SAMPLE_RATE,
        n_fft=n_fft,
        hop_length=hop,
        n_mels=40,
        fmin=50,
        fmax=SAMPLE_RATE // 2,
    )
    mel_db = librosa.power_to_db(mel + 1e-10)

    mfcc = librosa.feature.mfcc(S=mel_db, n_mfcc=13)
    d1 = librosa.feature.delta(mfcc, width=9)
    d2 = librosa.feature.delta(mfcc, order=2, width=9)

    feats = np.concatenate([mfcc, d1, d2, mel_db[:20]], axis=0).T.astype(np.float32)

    mu = feats.mean(axis=0, keepdims=True)
    std = feats.std(axis=0, keepdims=True) + 1e-6
    return (feats - mu) / std
