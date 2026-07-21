def find_audio_files(audio_dir: Path) -> Dict[str, Path]:
    exts = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}
    out = {}
    for p in audio_dir.rglob("*"):
        if p.suffix.lower() in exts:
            out[p.stem] = p
    return out

def write_csv(path: Path, rows: List[List], header: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

def load_duration_priors(path: Path) -> Dict[int, Dict[str, float]]:
    with path.open("r") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}
