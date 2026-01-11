import os
import sys
import json
import argparse
import numpy as np
import torch
import torchaudio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import subprocess
import tempfile
import shutil
import re
import pickle

try:
    from sr_model_loader import ECAPAModel, ResNet34Model, XVectorModel, HuBERTModel, WavLMModel, IVectorPLDAModel
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent))
    from sr_model_loader import ECAPAModel, ResNet34Model, XVectorModel, HuBERTModel, WavLMModel, IVectorPLDAModel

def load_and_preprocess_audio(audio_path, sample_rate=16000):
    waveform, sr = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    waveform = waveform.squeeze()
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        waveform = resampler(waveform)
    return waveform

def insert_perturbation(audio, delta, position='cover_all'):
    if position == 'cover_all':
        T = audio.size(0)
        delta = delta.to(audio.device)
        delta_len = delta.size(0)
        if T <= delta_len:
            adv_audio = audio + delta[:T]
        else:
            adv_audio = audio.clone()
            adv_audio[:delta_len] = audio[:delta_len] + delta
        adv_audio = torch.clamp(adv_audio, -1.0, 1.0)
        return adv_audio
    else:
        raise NotImplementedError

def build_speaker_prototypes(enroll_dir, model, device='cuda', cache_dir=None):
    enroll_path = Path(enroll_dir)
    prototypes = {}
    non_target_dirs = sorted([d for d in enroll_path.iterdir() if d.is_dir() and d.name != 'enroll_tgt'])
        
    for spk_dir in tqdm(non_target_dirs, desc="Processing non-target"):
        audio_files = sorted(list(spk_dir.glob('*.wav')))
        embeddings = []
        for audio_file in audio_files[:5]:
            waveform = load_and_preprocess_audio(audio_file)
            waveform = waveform.to(device)
            if hasattr(model, 'extract_ivector_from_path'):
                emb = model.extract_embedding(waveform, cache_dir=cache_dir)
            else:
                emb = model.extract_embedding(waveform)
            embeddings.append(emb)
        
        if hasattr(model, 'compute_score') and 'IVector' in model.__class__.__name__:
             prototypes[spk_dir.name] = np.mean(embeddings, axis=0)
        else:
            avg_emb = torch.stack(embeddings).mean(dim=0)
            prototypes[spk_dir.name] = torch.nn.functional.normalize(avg_emb, p=2, dim=0)

    tgt_dir = enroll_path / 'enroll_tgt'
    if tgt_dir.exists():
        audio_files = sorted(list(tgt_dir.glob('*.wav')))
        embeddings = []
        for audio_file in audio_files[:5]:
            waveform = load_and_preprocess_audio(audio_file)
            waveform = waveform.to(device)
            if hasattr(model, 'extract_ivector_from_path'):
                 emb = model.extract_embedding(waveform, cache_dir=cache_dir)
            else:
                 emb = model.extract_embedding(waveform)
            embeddings.append(emb)
        
        if hasattr(model, 'compute_score') and 'IVector' in model.__class__.__name__:
             prototypes['tgt'] = np.mean(embeddings, axis=0)
        else:
             avg_emb = torch.stack(embeddings).mean(dim=0)
             prototypes['tgt'] = torch.nn.functional.normalize(avg_emb, p=2, dim=0)

    speaker_ids = sorted([k for k in prototypes.keys() if k != 'tgt']) + ['tgt']
    return prototypes, speaker_ids

def evaluate_test_set(test_dir, prototypes, speaker_ids, model, delta=None, device='cuda', cache_dir=None):
    test_path = Path(test_dir)
    records = []
    test_spk_dirs = sorted([d for d in test_path.iterdir() if d.is_dir()])
    
    for spk_dir in tqdm(test_spk_dirs, desc="Testing"):
        true_spk = spk_dir.name
        is_target_dir = (true_spk == 'enroll_tgt' or true_spk == 'tgt')
        if is_target_dir: continue

        audio_files = sorted(list(spk_dir.glob('*.wav')))
        for audio_file in audio_files[:10]:
            record = {'file': str(audio_file), 'true_spk': true_spk}
            
            waveform = load_and_preprocess_audio(audio_file)
            waveform = waveform.to(device)
            
            if hasattr(model, 'extract_ivector_from_path'):
                emb_clean = model.extract_embedding(waveform, cache_dir=cache_dir)
            else:
                emb_clean = model.extract_embedding(waveform)

            max_score = -float('inf')
            pred_spk = None
            for spk_id in speaker_ids:
                score = model.compute_score(emb_clean, prototypes[spk_id])
                if score > max_score:
                    max_score = score
                    pred_spk = spk_id
            
            record['clean_pred'] = pred_spk
            record['clean_correct'] = (pred_spk == true_spk)
            record['score_clean_tgt'] = model.compute_score(emb_clean, prototypes['tgt'])
            
            if delta is not None:
                waveform_adv = insert_perturbation(waveform, delta)
                if hasattr(model, 'extract_ivector_from_path'):
                    emb_adv = model.extract_embedding(waveform_adv, cache_dir=cache_dir)
                else:
                    emb_adv = model.extract_embedding(waveform_adv)
                
                max_score_adv = -float('inf')
                pred_spk_adv = None
                for spk_id in speaker_ids:
                    score = model.compute_score(emb_adv, prototypes[spk_id])
                    if score > max_score_adv:
                        max_score_adv = score
                        pred_spk_adv = spk_id
                
                record['adv_pred'] = pred_spk_adv
                record['score_adv_tgt'] = model.compute_score(emb_adv, prototypes['tgt'])
            else:
                record['adv_pred'] = None
                record['score_adv_tgt'] = None
            
            records.append(record)
    return records

def compute_metrics(records):
    N_total = len(records)
    if N_total == 0: return {}
    
    N_clean_correct = sum(1 for r in records if r['clean_correct'])
    acc_clean = N_clean_correct / N_total
    
    N_adv_correct = sum(1 for r in records if r['adv_pred'] is not None and r['adv_pred'] == r['true_spk'])
    acc_adv = N_adv_correct / N_total
    
    N_succ = sum(1 for r in records if r['adv_pred'] == 'tgt')
    asr_csi = N_succ / N_total
    
    scores_clean_tgt = [r['score_clean_tgt'] for r in records]
    scores_adv_tgt = [r['score_adv_tgt'] for r in records if r['score_adv_tgt'] is not None]
    
    mean_score_clean_tgt = np.mean(scores_clean_tgt) if scores_clean_tgt else 0.0
    mean_score_adv_tgt = np.mean(scores_adv_tgt) if scores_adv_tgt else 0.0
    mean_score_shift = mean_score_adv_tgt - mean_score_clean_tgt
    
    return {
        'num_tests': N_total,
        'acc_clean': acc_clean,
        'acc_adv': acc_adv,
        'SRoA': asr_csi,
        'mean_score_adv_tgt': mean_score_adv_tgt,
    }



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, required=True, choices=['ecapa', 'hubert', 'ivector', 'resnet34', 'wavlm', 'xvector'])
    parser.add_argument('--dataset_root', type=str, default='./datasets/sr_eval_vctk')
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--perturbation_path', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='./results/sr_eval')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--cache_dir', type=str, default='./cache/ivector_cache')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.model_type == 'ecapa':
        model = ECAPAModel(args.model_dir, args.device)
    elif args.model_type == 'hubert':
        model = HuBERTModel(args.model_dir, args.device)
    elif args.model_type == 'ivector':
        model = IVectorPLDAModel(args.model_dir, args.device)
    elif args.model_type == 'resnet34':
        model = ResNet34Model(args.model_dir, args.device)
    elif args.model_type == 'wavlm':
        model = WavLMModel(args.model_dir, args.device)
    elif args.model_type == 'xvector':
        model = XVectorModel(args.model_dir, args.device)

    delta = None
    if args.perturbation_path and os.path.exists(args.perturbation_path):
        checkpoint = torch.load(args.perturbation_path, map_location=args.device)
        if isinstance(checkpoint, dict):
            for key in ['delta', 'delta_universal', 'perturbation', 'universal_perturbation']:
                if key in checkpoint:
                    delta = checkpoint[key]
                    break
            if delta is None: 
                 delta = next(v for v in checkpoint.values() if isinstance(v, torch.Tensor))
        else:
            delta = checkpoint
        
        if delta.dim() == 2:
            delta = delta.squeeze(0)
    
    prototypes, speaker_ids = build_speaker_prototypes(
        os.path.join(args.dataset_root, 'enroll'),
        model,
        device=args.device,
        cache_dir=args.cache_dir
    )
    
    records = evaluate_test_set(
        os.path.join(args.dataset_root, 'test'),
        prototypes,
        speaker_ids,
        model,
        delta=delta,
        device=args.device,
        cache_dir=args.cache_dir
    )
    
    metrics = compute_metrics(records)
    print(json.dumps(metrics, indent=2))
    
    with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    

if __name__ == '__main__':
    main()
