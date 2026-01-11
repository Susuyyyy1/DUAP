import os
import re
import subprocess
import tempfile
import shutil
import numpy as np
import torch
import torchaudio
from pathlib import Path

try:
    from speechbrain.pretrained import EncoderClassifier
except ImportError:
    pass

class ECAPAModel:
    def __init__(self, model_dir, device='cuda'):
        self.device = device
        print(f"Loading ECAPA-TDNN model: {model_dir}")
        self.model = EncoderClassifier.from_hparams(
            source=model_dir,
            savedir=model_dir,
            run_opts={"device": device}
        )
        self.model.eval()
    
    def extract_embedding(self, audio_tensor):
        with torch.no_grad():
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
            emb = self.model.encode_batch(audio_tensor.to(self.device))
            emb = emb.squeeze()
            emb = torch.nn.functional.normalize(emb, p=2, dim=0)
            return emb
    
    def compute_score(self, emb1, emb2):
        if isinstance(emb1, torch.Tensor):
            return torch.dot(emb1, emb2).item()
        else:
            return np.dot(emb1, emb2)

class XVectorModel:
    def __init__(self, model_dir, device='cuda'):
        self.device = device
        print(f"Loading X-vector model: {model_dir}")
        self.model = EncoderClassifier.from_hparams(
            source=model_dir,
            savedir=model_dir,
            run_opts={"device": device}
        )
        self.model.eval()

    def extract_embedding(self, audio_tensor):
        with torch.no_grad():
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
            embedding = self.model.encode_batch(audio_tensor.to(self.device))
            embedding = embedding.squeeze()
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=0)
            return embedding
    
    def compute_score(self, emb1, emb2):
        if isinstance(emb1, torch.Tensor):
            return torch.dot(emb1, emb2).item()
        return np.dot(emb1, emb2)

        if transform.shape[1] == ivector.shape[0]:
            ivector = transform @ ivector
        elif transform.shape[1] == ivector.shape[0] + 1:
            ivector_aug = np.concatenate([ivector, np.array([1.0], dtype=ivector.dtype)])
            ivector = transform @ ivector_aug
        norm = np.linalg.norm(ivector)
        if norm > 0:
            ivector = ivector / norm
        return ivector
    
    def _read_kaldi_vector(self, path):
        with open(path, "r") as f:
            content = f.read().strip()
        parts = content.replace('[', '').replace(']', '').split()
        try:
            float(parts[0])
        except ValueError:
            parts = parts[1:]
        vec = np.array([float(x) for x in parts], dtype=np.float32)
        return vec
    
    def _read_kaldi_matrix(self, path):
        cmd = [os.path.join(self.MATRIXBIN, "copy-matrix"), "--binary=false", str(path), "-"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        lines = result.stdout.strip().splitlines()
        rows = []
        for line in lines:
            line = line.strip()
            if not line: continue
            line = line.replace('[', '').replace(']', '')
            if not line: continue
            parts = line.split()
            try:
                float(parts[0])
            except (ValueError, IndexError):
                if len(parts) > 1: parts = parts[1:]
                else: continue
            if not parts: continue
            rows.append([float(x) for x in parts])
        mat = np.array(rows, dtype=np.float32)
        return mat
    
    def compute_score(self, ivec1, ivec2):
        with tempfile.TemporaryDirectory() as tmpdir:
            ivec1_path = os.path.join(tmpdir, "ivec1.ark")
            ivec2_path = os.path.join(tmpdir, "ivec2.ark")
            score_path = os.path.join(tmpdir, "scores.txt")
            trials_path = os.path.join(tmpdir, "trials.txt")
            
            self._write_single_ivector_ark(ivec1, ivec1_path, utt_id="utt1")
            self._write_single_ivector_ark(ivec2, ivec2_path, utt_id="utt2")
            
            with open(trials_path, "w") as f:
                f.write("utt1 utt2\n")
            
            cmd = [
                os.path.join(self.IVECTORBIN, "ivector-plda-scoring"),
                "--num-utts=ark:echo utt1 1; echo utt2 1|",
                str(self.plda),
                f"ark:{ivec1_path}",
                f"ark:{ivec2_path}",
                trials_path,
                score_path
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            
            with open(score_path, "r") as f:
                line = f.readline().strip()
                score = float(line.split()[-1])
            return score
            
    def _write_single_ivector_ark(self, ivector, ark_path, utt_id):
        txt_path = ark_path + '.txt'
        with open(txt_path, 'w') as f:
            values_str = ' '.join([f"{x:.6f}" for x in ivector.tolist()])
            f.write(f"{utt_id}  [ {values_str} ]\n")
        cmd = [os.path.join(self.MATRIXBIN, 'copy-vector'), f'ark,t:{txt_path}', f'ark:{ark_path}']
        subprocess.run(cmd, check=True, capture_output=True)
