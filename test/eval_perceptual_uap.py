import argparse
import json
import math
from pathlib import Path
import numpy as np
import torch
import torchaudio
from pesq import pesq
from torchmetrics.audio.nisqa import NonIntrusiveSpeechQualityAssessment
import os
NUM_THREADS = 48  
torch.set_num_threads(NUM_THREADS)
os.environ["OMP_NUM_THREADS"] = str(NUM_THREADS)
os.environ["MKL_NUM_THREADS"] = str(NUM_THREADS)
SAMPLE_RATE = 16000
EPS = 1e-12
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean_dir", type=str, required=True,)
    parser.add_argument("--delta_path", type=str, required=True)
    parser.add_argument("--output_json", type=str, default="perceptual_metrics_uap.json")
    parser.add_argument("--file_ext", type=str, default="wav")
    return parser.parse_args()

def load_wav_mono_16k(path: Path) -> np.ndarray:
    wav, sr = torchaudio.load(str(path))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    wav = wav.squeeze(0).numpy().astype(np.float32)
    return wav

def load_delta(delta_path: Path) -> np.ndarray:
    suffix = delta_path.suffix.lower()
    if suffix in [".pt", ".pth"]:
        ckpt = torch.load(str(delta_path), map_location="cpu")
        if isinstance(ckpt, dict):
            if "delta_universal" in ckpt:
                delta = ckpt["delta_universal"]
            elif "delta" in ckpt:
                delta = ckpt["delta"]
            else:
                one_d_tensors = [v for v in ckpt.values() if torch.is_tensor(v) and v.ndim in (1, 2)]
                if not one_d_tensors:
                    raise RuntimeError("Cannot find delta_universal")
                delta = one_d_tensors[0]
        elif torch.is_tensor(ckpt):
            delta = ckpt
        else:
            raise RuntimeError("Unsupported .pt format")
        if delta.ndim == 2:
            delta = delta.squeeze(0)
        delta_np = delta.detach().cpu().numpy().astype(np.float32)

    elif suffix in [".wav", ".flac"]:
        delta_np = load_wav_mono_16k(delta_path)
    else:
        raise RuntimeError(f"Unsupported perturbation file extension: {suffix}")

    return delta_np

def apply_delta(clean: np.ndarray, delta: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    L = min(len(clean), len(delta))
    clean_seg = clean[:L]
    delta_seg = delta[:L]
    adv_seg = clean_seg + delta_seg
    adv_seg = np.clip(adv_seg, -1.0, 1.0)
    return clean_seg, adv_seg, delta_seg

def compute_snr_db(clean: np.ndarray, delta: np.ndarray) -> float:
    assert len(clean) == len(delta)
    p_clean = np.mean(clean ** 2)
    p_noise = np.mean(delta ** 2)
    snr = 10 * math.log10((p_clean + EPS) / (p_noise + EPS))
    return float(snr)

def compute_pesq_score(clean: np.ndarray, adv: np.ndarray) -> float:
    assert len(clean) == len(adv)
    score = pesq(SAMPLE_RATE, clean, adv, "wb")
    return float(score)

def build_nisqa_metric(device: str = "cpu"):
    metric = NonIntrusiveSpeechQualityAssessment(fs=SAMPLE_RATE).to(device)
    metric.eval()
    return metric

def compute_nisqa_overall(metric, wav_np: np.ndarray, device: str = "cpu") -> float:
    wav_tensor = torch.from_numpy(wav_np).float().to(device)
    with torch.no_grad():
        scores = metric(wav_tensor)
    scores = scores.detach().cpu().numpy()
    overall = float(scores[0])
    return overall

def main():
    args = parse_args()
    clean_dir = Path(args.clean_dir)
    delta_path = Path(args.delta_path)
    output_json = Path(args.output_json)
    assert clean_dir.is_dir(), f"clean_dir does not exist: {clean_dir}"
    assert delta_path.is_file(), f"delta_path does not exist: {delta_path}"
    delta = load_delta(delta_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    nisqa_metric = build_nisqa_metric(device=device)
    pattern = f"*.{args.file_ext}"
    clean_files = sorted(clean_dir.glob(pattern))
    if not clean_files:
        raise RuntimeError(f"No {pattern} files found in {clean_dir}")
    print(f"==> Found {len(clean_files)} clean audio files")
    per_file_results = []
    snr_list = []
    pesq_list = []
    nisqa_clean_list = []
    nisqa_adv_list = []
    for idx, clean_path in enumerate(clean_files, 1):
        print(f"[{idx}/{len(clean_files)}] Processing {clean_path.name} ...")
        clean = load_wav_mono_16k(clean_path)
        clean_seg, adv_seg, delta_seg = apply_delta(clean, delta)
        snr_db = compute_snr_db(clean_seg, delta_seg)
        snr_list.append(snr_db)
        pesq_score = compute_pesq_score(clean_seg, adv_seg)
        pesq_list.append(pesq_score)
        nisqa_clean = compute_nisqa_overall(nisqa_metric, clean_seg, device=device)
        nisqa_adv = compute_nisqa_overall(nisqa_metric, adv_seg, device=device)
        nisqa_clean_list.append(nisqa_clean)
        nisqa_adv_list.append(nisqa_adv)
        per_file_results.append({
            "filename": clean_path.name,
            "snr_db": snr_db,
            "pesq": pesq_score,
            "nisqa_clean": nisqa_clean,
            "nisqa_adv": nisqa_adv,
            "delta_nisqa": nisqa_adv - nisqa_clean,
            "clean_len_sec": len(clean_seg) / SAMPLE_RATE,
        })

    def mean_or_nan(arr):
        return float(np.mean(arr)) if arr else float("nan")
    overall_stats = {
        "num_files": len(per_file_results),
        "avg_snr_db": mean_or_nan(snr_list),
        "avg_pesq": mean_or_nan(pesq_list),
        "avg_nisqa_clean": mean_or_nan(nisqa_clean_list),
        "avg_nisqa_adv": mean_or_nan(nisqa_adv_list),
        "avg_delta_nisqa": mean_or_nan(nisqa_adv_list) - mean_or_nan(nisqa_clean_list),
    }
    for k, v in overall_stats.items():
        print(f"{k:20s}: {v:.4f}" if isinstance(v, float) else f"{k:20s}: {v}")
    result = {
        "config": {
            "clean_dir": str(clean_dir),
            "delta_path": str(delta_path),
            "sample_rate": SAMPLE_RATE,
            "delta_len_samples": len(delta),
            "delta_len_sec": len(delta) / SAMPLE_RATE,
        },
        "overall": overall_stats,
        "per_file": per_file_results,
    }
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
if __name__ == "__main__":
    main()