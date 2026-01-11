import os
import sys
import time
import json
import argparse
import datetime
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchaudio
import numpy as np
import soundfile as sf
from tqdm import tqdm
from speechbrain.inference.speaker import EncoderClassifier
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
import torch.nn.functional as F
import tempfile
from whisper_attacker import Attacker, normalize_audio, set_seed
from sr_models import SRModelManager


class UniversalAudioDataset(Dataset):
    def __init__(self, audio_dir, sample_rate=16000, max_length_sec=10):
        self.audio_dir = Path(audio_dir)
        self.sample_rate = sample_rate
        self.max_length = int(max_length_sec * sample_rate)
        
        self.audio_files = sorted(list(self.audio_dir.glob('*.wav')))
        
        if len(self.audio_files) == 0:
            raise ValueError(f"No .wav files found in {audio_dir}")
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        wav, sr = torchaudio.load(audio_path)
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        wav = wav.squeeze(0) 
        if wav.size(0) > self.max_length:
            start = torch.randint(0, wav.size(0) - self.max_length + 1, (1,)).item()
            wav = wav[start:start + self.max_length]
        elif wav.size(0) < self.max_length:
            pad_len = self.max_length - wav.size(0)
            wav = torch.nn.functional.pad(wav, (0, pad_len))
        wav = normalize_audio(wav.unsqueeze(0)).squeeze(0) 
        return wav, str(audio_path)

class UniversalPerturbationTrainer:
    def __init__(self, args, attacker):
        self.args = args
        self.attacker = attacker
        self.device = attacker.device
        self.perturbation_duration = 10.0  
        self.sample_rate = 16000
        self.perturbation_len = int(self.perturbation_duration * self.sample_rate) 
        self.delta_universal = nn.Parameter(
            torch.randn(1, self.perturbation_len, device=self.device) * 0.01
        )
        print(f"\n Loading SR Models...")
        self.sr_manager = SRModelManager(self.device)
        self.sr_manager.load_ecapa(args.ecapa_dir)
        self.sr_manager.load_wavlm(args.wavlm_dir)
        self.sr_manager.load_resnet34(args.resnet34_dir)
        self._load_speaker_prototypes(args)
        self.sr_mu_ecapa = 0.0
        self.sr_sigma_ecapa = 1.0
        self.sr_mu_wavlm = 0.0
        self.sr_sigma_wavlm = 1.0
        self.sr_mu_resnet = 0.0
        self.sr_sigma_resnet = 1.0
        self.sr_step = 0  
        self.use_psy = getattr(args, 'use_psy', False) or (getattr(args, 'psy_weight', 0.0) > 0)
        self.psy_margin_db = getattr(args, 'psy_margin_db', 15.0)
        self.psy_weight = getattr(args, 'psy_weight', 0.0)
        self.psy_n_fft = 512
        self.psy_hop_length = 256
        self.psy_window = torch.hann_window(self.psy_n_fft, device=self.device)
        self.psy_eps = 1e-12
        self.optimizer = torch.optim.Adam(
            [self.delta_universal], 
            lr=args.lr,
            betas=(0.9, 0.999)
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=args.epochs,
            eta_min=args.lr * 0.1
        )
        self.train_stats = {
            'epoch_losses': [],
            'asr_losses': [],
            'sr_losses': [],
            'best_loss': float('inf'),
            'best_epoch': 0
        }
        self.global_step = 0  
    
    def _load_speaker_prototypes(self, args):

        if not hasattr(args, 'enroll_dir') or args.enroll_dir is None:
            self.spk_protos_ecapa = None
            self.spk_protos_wavlm = None
            self.spk_protos_resnet = None
            self.target_spk_idx = -1
            self.num_speakers = 0
            return
        enroll_dir = Path(args.enroll_dir)
        if not enroll_dir.exists():
            raise FileNotFoundError(f"Enroll directory not found: {enroll_dir}")
        enroll_tgt_dir = enroll_dir / 'enroll_tgt'
        has_special_target = enroll_tgt_dir.exists() and enroll_tgt_dir.is_dir()
        speaker_dirs = sorted([d for d in enroll_dir.iterdir() if d.is_dir()])
        
        if has_special_target:
            speaker_dirs = [d for d in speaker_dirs if d.name != 'enroll_tgt']
            self.target_spk_idx = len(speaker_dirs)
            self.num_speakers = len(speaker_dirs) + 1  
            speaker_dirs.append(enroll_tgt_dir)
        else:
            self.num_speakers = len(speaker_dirs)
            if hasattr(args, 'target_speaker_id') and args.target_speaker_id is not None:
                self.target_spk_idx = args.target_speaker_id
            else:
                self.target_spk_idx = self.num_speakers - 1
        protos_ecapa = []
        protos_wavlm = []
        protos_resnet = []
        for idx, spk_dir in enumerate(tqdm(speaker_dirs, desc="Extracting speaker prototypes")):
            audio_files = sorted(list(spk_dir.glob('*.wav')))[:5]
            if len(audio_files) == 0:
                raise ValueError(f"No audio files found in {spk_dir}")
            embs_ecapa = []
            embs_wavlm = []
            embs_resnet = []
            
            with torch.no_grad():
                for audio_file in audio_files:
                    wav, sr = torchaudio.load(audio_file)
                    if sr != self.sample_rate:
                        wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
                    if wav.size(0) > 1:
                        wav = wav.mean(dim=0, keepdim=True)
                    wav = normalize_audio(wav).squeeze(0).to(self.device)
                    embs_ecapa.append(self.sr_manager.embed(wav, 'ecapa'))
                    embs_wavlm.append(self.sr_manager.embed(wav, 'wavlm'))
                    embs_resnet.append(self.sr_manager.embed(wav, 'resnet'))
            
            proto_ecapa = torch.stack(embs_ecapa).mean(dim=0)
            proto_wavlm = torch.stack(embs_wavlm).mean(dim=0)
            proto_resnet = torch.stack(embs_resnet).mean(dim=0)
            proto_ecapa = F.normalize(proto_ecapa, p=2, dim=0)
            proto_wavlm = F.normalize(proto_wavlm, p=2, dim=0)
            proto_resnet = F.normalize(proto_resnet, p=2, dim=0)
            protos_ecapa.append(proto_ecapa)
            protos_wavlm.append(proto_wavlm)
            protos_resnet.append(proto_resnet)
        self.spk_protos_ecapa = torch.stack(protos_ecapa)
        self.spk_protos_wavlm = torch.stack(protos_wavlm)
        self.spk_protos_resnet = torch.stack(protos_resnet)

    def compute_scores(self, audios):
        B = audios.size(0)
        embs_ecapa = []
        embs_wavlm = []
        embs_resnet = []
        for i in range(B):
            emb_ecapa = self.sr_manager.embed(audios[i], 'ecapa')  
            emb_wavlm = self.sr_manager.embed(audios[i], 'wavlm')  
            emb_resnet = self.sr_manager.embed(audios[i], 'resnet') 
            embs_ecapa.append(emb_ecapa)
            embs_wavlm.append(emb_wavlm)
            embs_resnet.append(emb_resnet)
        embs_ecapa = torch.stack(embs_ecapa)  
        embs_wavlm = torch.stack(embs_wavlm)  
        embs_resnet = torch.stack(embs_resnet)
        S_ecapa = torch.matmul(embs_ecapa, self.spk_protos_ecapa.T)  
        S_wavlm = torch.matmul(embs_wavlm, self.spk_protos_wavlm.T)  
        S_resnet = torch.matmul(embs_resnet, self.spk_protos_resnet.T)  
        return S_ecapa, S_wavlm, S_resnet
        
    def insert_perturbation(self, audios, position='cover_all'):
        B, T = audios.shape
        delta_len = self.perturbation_len  
        perturbed = audios.clone()
        
        if position == 'cover_all':
            if T <= delta_len:
                perturbed += self.delta_universal[0, :T].unsqueeze(0)
            else:
                num_repeats = (T // delta_len) + 1
                repeated_delta = self.delta_universal.repeat(1, num_repeats)[:, :T]
                perturbed += repeated_delta
        
        elif position == 'random':
            for i in range(B):
                if T <= delta_len:
                    max_start = delta_len - T
                    start = torch.randint(0, max_start + 1, (1,)).item()
                    perturbed[i] += self.delta_universal[0, start:start+T]
                else:
                    num_repeats = (T // delta_len) + 1
                    repeated_delta = self.delta_universal.repeat(1, num_repeats)[:, :T]
                    perturbed[i] += repeated_delta[0]
        
        perturbed = torch.clamp(perturbed, -1.0, 1.0)
        
        return perturbed
    
    def compute_psychoacoustic_loss(self, clean_audios, adv_audios):
        delta = adv_audios - clean_audios
        X = torch.stft(
            clean_audios, 
            n_fft=self.psy_n_fft, 
            hop_length=self.psy_hop_length,
            window=self.psy_window, 
            return_complex=True
        ) 
        
        D = torch.stft(
            delta, 
            n_fft=self.psy_n_fft, 
            hop_length=self.psy_hop_length,
            window=self.psy_window, 
            return_complex=True
        )  
        Px = 10.0 * torch.log10(X.abs().pow(2) + self.psy_eps) 
        Pd = 10.0 * torch.log10(D.abs().pow(2) + self.psy_eps) 
        theta_x = Px - self.psy_margin_db
        over = Pd - theta_x  
        over_clamped = torch.clamp(over, min=0.0)
        L_psy = over_clamped.mean()
        return L_psy
    
    def calc_batch_loss(self, audios, orig_audios, inner_iter=None, batch_idx=None):
        B = audios.size(0)
        asr_loss, pred_texts = self.attacker.calc_ds_ctc_loss(audios, self.args.tgt_text)
        if self.spk_protos_ecapa is not None:
            S_ecapa, S_wavlm, S_resnet = self.compute_scores(audios)
            target_labels = torch.full(
                (B,), 
                self.target_spk_idx, 
                dtype=torch.long, 
                device=self.device
            )
            loss_sr_ecapa_raw = F.cross_entropy(S_ecapa, target_labels, reduction='mean')
            loss_sr_wavlm_raw = F.cross_entropy(S_wavlm, target_labels, reduction='mean')
            loss_sr_resnet_raw = F.cross_entropy(S_resnet, target_labels, reduction='mean')
            self.sr_step += 1
            momentum = 0.9
            loss_e = loss_sr_ecapa_raw.item()
            self.sr_mu_ecapa = momentum * self.sr_mu_ecapa + (1 - momentum) * loss_e
            self.sr_sigma_ecapa = momentum * self.sr_sigma_ecapa + (1 - momentum) * (loss_e ** 2)
            std_ecapa = np.sqrt(max(self.sr_sigma_ecapa - self.sr_mu_ecapa ** 2, 1e-8))
            loss_w = loss_sr_wavlm_raw.item()
            self.sr_mu_wavlm = momentum * self.sr_mu_wavlm + (1 - momentum) * loss_w
            self.sr_sigma_wavlm = momentum * self.sr_sigma_wavlm + (1 - momentum) * (loss_w ** 2)
            std_wavlm = np.sqrt(max(self.sr_sigma_wavlm - self.sr_mu_wavlm ** 2, 1e-8))
            loss_r = loss_sr_resnet_raw.item()
            self.sr_mu_resnet = momentum * self.sr_mu_resnet + (1 - momentum) * loss_r
            self.sr_sigma_resnet = momentum * self.sr_sigma_resnet + (1 - momentum) * (loss_r ** 2)
            std_resnet = np.sqrt(max(self.sr_sigma_resnet - self.sr_mu_resnet ** 2, 1e-8))
            loss_sr_ecapa = (loss_sr_ecapa_raw - self.sr_mu_ecapa) / std_ecapa
            loss_sr_wavlm = (loss_sr_wavlm_raw - self.sr_mu_wavlm) / std_wavlm
            loss_sr_resnet = (loss_sr_resnet_raw - self.sr_mu_resnet) / std_resnet
            loss_sr_ecapa = torch.clamp(loss_sr_ecapa, min=0.0)
            loss_sr_wavlm = torch.clamp(loss_sr_wavlm, min=0.0)
            loss_sr_resnet = torch.clamp(loss_sr_resnet, min=0.0)
            sr_loss_total = (loss_sr_ecapa + loss_sr_wavlm + loss_sr_resnet) / 3.0
            pred_ecapa = torch.argmax(S_ecapa, dim=1)  
            pred_wavlm = torch.argmax(S_wavlm, dim=1)
            pred_resnet = torch.argmax(S_resnet, dim=1)
            
            success_ecapa = (pred_ecapa == self.target_spk_idx).cpu().numpy()
            success_wavlm = (pred_wavlm == self.target_spk_idx).cpu().numpy()
            success_resnet = (pred_resnet == self.target_spk_idx).cpu().numpy()
            
            success_joint = success_ecapa & success_wavlm & success_resnet
            sr_success_count = int(success_joint.sum())
            
            probs_ecapa = F.softmax(S_ecapa, dim=1)[:, self.target_spk_idx].detach().cpu().numpy()
            probs_wavlm = F.softmax(S_wavlm, dim=1)[:, self.target_spk_idx].detach().cpu().numpy()
            probs_resnet = F.softmax(S_resnet, dim=1)[:, self.target_spk_idx].detach().cpu().numpy()
            
            cos_ecapa_list = probs_ecapa.tolist()  
            cos_wavlm_list = probs_wavlm.tolist()
            cos_resnet_list = probs_resnet.tolist()
        else:
            sr_loss_total = torch.tensor(0.0, device=self.device)
            loss_sr_ecapa_raw = torch.tensor(0.0, device=self.device)
            loss_sr_wavlm_raw = torch.tensor(0.0, device=self.device)
            loss_sr_resnet_raw = torch.tensor(0.0, device=self.device)
            loss_sr_ecapa = torch.tensor(0.0, device=self.device)
            loss_sr_wavlm = torch.tensor(0.0, device=self.device)
            loss_sr_resnet = torch.tensor(0.0, device=self.device)
            cos_ecapa_list = [0.0] * B
            cos_wavlm_list = [0.0] * B
            cos_resnet_list = [0.0] * B
            sr_success_count = 0

        T = self.delta_universal.numel() 
        reg_raw = (self.delta_universal.pow(2).sum() / T) 
        l2_reg = reg_raw * self.args.lambda_reg
        psy_loss_raw = torch.tensor(0.0, device=self.device)
        
        if self.use_psy and self.psy_weight > 0:
            psy_loss_raw = self.compute_psychoacoustic_loss(orig_audios, audios)
            psy_reg = self.psy_weight * psy_loss_raw
        else:
            psy_reg = torch.tensor(0.0, device=self.device)
        total_loss = (self.args.asr_weight * asr_loss + 
                     self.args.sr_weight * sr_loss_total + 
                     l2_reg +
                     psy_reg)
        delta_linf = self.delta_universal.abs().max().item()
        delta_rms = torch.sqrt(reg_raw).item()
        sr_success_rate = sr_success_count / B if B > 0 else 0.0
        asr_success_count = sum(1 for pred in pred_texts if pred.strip() == self.args.tgt_text.strip())
        asr_success_rate = asr_success_count / B if B > 0 else 0.0
        detailed_metrics = {
            'loss_asr': asr_loss.item(),
            'loss_sr_total': sr_loss_total.item(),
            'loss_psy_raw': psy_loss_raw.item() if self.use_psy else 0.0,
            'weighted_asr': (self.args.asr_weight * asr_loss).item(),
            'weighted_sr': (self.args.sr_weight * sr_loss_total).item(),
            'weighted_psy': psy_reg.item() if self.use_psy else 0.0,
            'loss_total': total_loss.item(),
            'asr_success_rate': asr_success_rate,
            'sr_success_rate': sr_success_rate,
        }
        sr_success_metrics = {
            'ecapa': cos_ecapa_list,
            'wavlm': cos_wavlm_list,
            'resnet': cos_resnet_list,
            'success_count': sr_success_count,
            'success_rate': sr_success_rate
        }
        return total_loss, asr_loss, sr_loss_total, pred_texts, sr_success_metrics, detailed_metrics
    
    def train_epoch(self, dataloader, epoch):
        self.attacker.whisper.train()
        self.sr_manager.set_eval()
        epoch_loss = 0.0
        epoch_asr_loss = 0.0
        epoch_sr_loss = 0.0
        total_batches = 0
        successful_batches = 0
        total_samples = 0
        successful_samples = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.args.epochs}")
        
        for batch_idx, (audios, paths) in enumerate(pbar):
            audios = audios.to(self.device)  
            B = audios.size(0)
            max_inner_iters = self.args.max_inner_iters  
            asr_success_threshold = self.args.success_threshold  
            sr_success_threshold = self.args.sr_success_threshold  
            batch_success = False
            for inner_iter in range(max_inner_iters):
                adv_audios = self.insert_perturbation(audios, position='cover_all')

                total_loss, asr_loss, sr_loss_total, pred_texts, sr_metrics, detailed_metrics = self.calc_batch_loss(
                    adv_audios, audios, inner_iter=inner_iter, batch_idx=batch_idx
                )
                asr_success_count = sum(1 for pred in pred_texts if pred.strip() == self.args.tgt_text.strip())
                asr_success_rate = asr_success_count / B
                sr_success_count = sr_metrics['success_count']
                sr_success_rate = sr_metrics['success_rate']
                self.optimizer.zero_grad()
                total_loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_([self.delta_universal], max_norm=1.0)
                self.global_step += 1
                self.optimizer.step()
                with torch.no_grad():
                    self.delta_universal.data = torch.clamp(
                        self.delta_universal.data, -self.args.delta_max, self.args.delta_max
                    )
                if self.spk_protos_ecapa is not None:
                    both_success = (asr_success_rate >= asr_success_threshold and 
                                   sr_success_rate >= sr_success_threshold)
                    if both_success:
                        batch_success = True
                        break
                else:
                    if asr_success_rate >= asr_success_threshold:
                        batch_success = True
                        break
                postfix_dict = {
                    'loss': f'{total_loss.item():.2f}',
                    'asr_succ': f'{asr_success_count}/{B}',
                    'sr_succ': f'{sr_success_count}/{B}' if self.spk_protos_ecapa is not None else 'N/A'
                }
                pbar.set_postfix(postfix_dict)
            epoch_loss += total_loss.item()
            epoch_asr_loss += asr_loss.item()
            epoch_sr_loss += sr_loss_total.item()
            total_batches += 1
            total_samples += B
            successful_samples += asr_success_count  
            if batch_success:
                successful_batches += 1
        avg_loss = epoch_loss / len(dataloader)
        avg_asr = epoch_asr_loss / len(dataloader)
        avg_sr = epoch_sr_loss / len(dataloader)
        print(f"Epoch {epoch+1} Summary: Avg Loss={avg_loss:.4f} | Avg ASR Loss={avg_asr:.4f} | Avg SR Loss={avg_sr:.4f}")
        return avg_loss, avg_asr, avg_sr
    
    def train(self, dataloader):
        start_time = time.time()
        
        for epoch in range(self.args.epochs):
            epoch_start = time.time()
            avg_loss, avg_asr, avg_sr = self.train_epoch(dataloader, epoch)
            self.scheduler.step()
            self.train_stats['epoch_losses'].append(avg_loss)
            self.train_stats['asr_losses'].append(avg_asr)
            self.train_stats['sr_losses'].append(avg_sr)
            # Save best model
            if avg_loss < self.train_stats['best_loss']:
                self.train_stats['best_loss'] = avg_loss
                self.train_stats['best_epoch'] = epoch + 1
                self.save_checkpoint(epoch + 1, is_best=True)
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(epoch + 1, is_best=False)
            epoch_time = time.time() - epoch_start
            total_time = time.time() - start_time
            
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{self.args.epochs} Summary")
            print(f"{'='*60}")
            print(f"Avg Loss: {avg_loss:.6f} | ASR: {avg_asr:.6f} | SR: {avg_sr:.6f}")
            print(f"Time: {epoch_time:.1f}s | Total: {str(datetime.timedelta(seconds=int(total_time)))}")
            print(f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            print(f"Best Loss: {self.train_stats['best_loss']:.6f} (Epoch {self.train_stats['best_epoch']})")
            print(f"{'='*60}\n")
        total_time = time.time() - start_time
        print("\n" + "="*60)
        print("Training Completed!")
        print("="*60)
        print(f"Total Time: {str(datetime.timedelta(seconds=int(total_time)))}")
        print(f"Best Loss: {self.train_stats['best_loss']:.6f} at Epoch {self.train_stats['best_epoch']}")
        print(f"Final perturbation saved to: {self.args.output_dir}")
        print("="*60 + "\n")
        self.save_checkpoint(self.args.epochs, is_best=False, is_final=True)
        self.save_training_stats()
    def save_checkpoint(self, epoch, is_best=False, is_final=False):
        checkpoint_dir = Path(self.args.output_dir) / 'checkpoints'
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'delta_universal': self.delta_universal.data.cpu(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'train_stats': self.train_stats,
            'args': vars(self.args)
        }
        
        if not is_best and not is_final:
            path = checkpoint_dir / f'delta_universal_epoch{epoch}.pt'
            torch.save(checkpoint, path)
            print(f"Checkpoint saved: {path}")
        if is_best:
            path = checkpoint_dir / 'delta_universal_best.pt'
            torch.save(checkpoint, path)
            print(f"Best model saved: {path}")
        if is_final:
            path = Path(self.args.output_dir) / 'delta_universal_final.pt'
            torch.save(checkpoint, path)
            print(f"Final model saved: {path}")
    
    def save_training_stats(self):
        stats_path = Path(self.args.output_dir) / 'training_stats.json'
        with open(stats_path, 'w') as f:
            json.dump(self.train_stats, f, indent=2)
        print(f"  Training stats saved: {stats_path}")

def parse_args():
    parser = argparse.ArgumentParser(description='Train Universal Adversarial Perturbation')
    parser.add_argument('--audio_dir', type=str, default='./datasets/train_set')
    parser.add_argument('--tgt_text', type=str,default='OPEN THE DOOR')
    parser.add_argument('--enroll_dir', type=str,default=None)
    parser.add_argument('--target_speaker_id', type=int,default=80)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--max_inner_iters', type=int, default=1000)
    parser.add_argument('--success_threshold', type=float, default=0.8)
    parser.add_argument('--sr_success_threshold', type=float, default=0.8)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--asr_weight', type=float, default=1.0)
    parser.add_argument('--sr_weight', type=float, default=1.0)
    parser.add_argument('--lambda_reg', type=float, default=5e-4)
    parser.add_argument('--delta_max', type=float, default=0.5)
    parser.add_argument('--use_psy', action='store_true')
    parser.add_argument('--psy_margin_db', type=float, default=15.0)
    parser.add_argument('--psy_weight', type=float, default=1e-3)
    parser.add_argument('--ecapa_dir', type=str, default='./sr_model/ecapa')
    parser.add_argument('--wavlm_dir', type=str, default='./sr_model/wavlm_base_plus_sv')
    parser.add_argument('--resnet34_dir', type=str, default='./sr_model/resnet34_voxceleb')
    parser.add_argument('--output_dir', type=str,default='./universal_perturbation_output')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--wav_input', type=str, default=None)
    parser.add_argument('--wav_tgt', type=str, default=None)
    parser.add_argument('--attack_iters', type=int, default=500)
    parser.add_argument('--segment_size', type=int, default=48000)
    parser.add_argument('--if_slm_loss', type=bool, default=False)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    attacker = Attacker(args)
    print(f"\n Loading dataset from {args.audio_dir}...")
    dataset = UniversalAudioDataset(
        audio_dir=args.audio_dir,
        sample_rate=16000,
        max_length_sec=10 
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    trainer = UniversalPerturbationTrainer(args, attacker)
    trainer.train(dataloader)

if __name__ == '__main__':
    main()