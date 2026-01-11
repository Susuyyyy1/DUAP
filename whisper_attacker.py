import os
import sys
import time
import json
import math
import datetime
import random
from pathlib import Path
import torch
import torchaudio
import numpy as np
import soundfile as sf
from torch import nn
from torch.nn import functional as F
import torchaudio.transforms as Trans_audio
from torch.cuda.amp import autocast
from tqdm import tqdm
from transformers import (
    pipeline,
    WhisperForConditionalGeneration,
    WhisperTokenizer,
    WhisperFeatureExtractor,
)
import whisper  

def set_seed(seed: int = 1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def normalize_audio(x: torch.Tensor) -> torch.Tensor:
    if x.dtype != torch.float32:
        x = x.float()
    if x.dim() == 3 and x.size(1) == 1:
        x = x.squeeze(1)
    return x


def clamp_perturbation(ptb: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.clamp(ptb, -eps, eps)


def safe_cosine_distance(feat_a: torch.Tensor, feat_b: torch.Tensor) -> torch.Tensor:
    if feat_a.dim() == 2:
        feat_a = feat_a.unsqueeze(0) 
    if feat_b.dim() == 2:
        feat_b = feat_b.unsqueeze(0)
    if feat_a.size(0) != feat_b.size(0):
        if feat_a.size(0) == 1:
            feat_a = feat_a.expand(feat_b.size(0), *feat_a.shape[1:])
        elif feat_b.size(0) == 1:
            feat_b = feat_b.expand(feat_a.size(0), *feat_b.shape[1:])
        else:
            raise ValueError(f"Batch sizes differ and cannot broadcast: {feat_a.size(0)} vs {feat_b.size(0)}")

    B = feat_a.size(0)
    a_flat = feat_a.reshape(B, -1)
    b_flat = feat_b.reshape(B, -1)

    a_norm = F.normalize(a_flat, p=2, dim=1)
    b_norm = F.normalize(b_flat, p=2, dim=1)

    cos = (a_norm * b_norm).sum(dim=1)  
    dist = 1.0 - cos                    
    return dist

class Attacker:
    def __init__(self, args):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.tgt_model = getattr(args, 'tgt_model', 'whisper')
        self.target_layer = getattr(args, 'tgt_layer', 'avg')
        self.wav_input = getattr(args, 'wav_input', None)
        self.wav_tgt = getattr(args, 'wav_tgt', None)
        self.attack_iters = getattr(args, 'attack_iters', 0)
        self.segment_size = getattr(args, 'segment_size', 48000)
        self.tgt_text = getattr(args, 'tgt_text', '')
        self.if_slm_loss = getattr(args, 'if_slm_loss', False)
        self.asr_weight = getattr(args, 'asr_weight', 1.0)
        self.sr_weight = getattr(args, 'sr_weight', 1.0)
        self.sample_rate = 16000
        self.window_size = .02
        self.window_stride = .01
        self.precision = 32
        default_model_path = "openai/whisper-small" 
        if os.path.exists("./models/whisper"):
             default_model_path = "./models/whisper"
        self.model_path = getattr(args, 'whisper_model_path', default_model_path)
        self.whisper = WhisperForConditionalGeneration.from_pretrained(self.model_path)
        self.whisper_tokenizer = WhisperTokenizer.from_pretrained(self.model_path)
        self.whisper_feature_extractor = WhisperFeatureExtractor.from_pretrained(self.model_path)
        device_idx = 0 if self.device.startswith('cuda') else -1
        self.whisper_pipe = pipeline(
            task="automatic-speech-recognition",
            model=self.whisper,
            tokenizer=self.whisper_tokenizer,
            feature_extractor=self.whisper_feature_extractor,
            device=device_idx
        )
        self.whisper.to(self.device)
        self.whisper_encoder = self.whisper.get_encoder()
        self.whisper.train()
        for param in self.whisper.parameters():
            param.requires_grad = False
        self.attack_criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=-100)

    def _stable_eval_decode(self, wav_1c: torch.Tensor) -> str:
        with torch.no_grad():
            wav_cpu = wav_1c.squeeze(0).detach().cpu().numpy()
            inputs = self.whisper_feature_extractor(
                wav_cpu,
                sampling_rate=self.sample_rate,
                return_tensors="pt"
            )
            input_features = inputs.input_features.to(self.device)
            forced_decoder_ids = self.whisper_tokenizer.get_decoder_prompt_ids(
                language="en", 
                task="transcribe"
            )
            generated_ids = self.whisper.generate(
                input_features,
                forced_decoder_ids=forced_decoder_ids,
                max_length=50
            )
            txt = self.whisper_tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        return txt

    def get_feat(self, wav, tgt_model):
        if wav.dim() == 1:
            wav_b = wav.unsqueeze(0) 
        else:
            wav_b = wav
        if 'whisper' in tgt_model:
            if wav_b.size(0) > 1:
                w1 = wav_b[0]
            else:
                w1 = wav_b[0]
            mel = whisper.log_mel_spectrogram(w1, device=self.device) 
            mel_b = mel.unsqueeze(0) 
            enc = self.whisper_encoder(mel_b, output_hidden_states=True)
            hs = enc.hidden_states 
            if self.target_layer == 'avg':
                wav_feat = torch.mean(torch.stack(hs), axis=0).squeeze(0) 
            else:
                wav_feat = hs[self.target_layer].squeeze(0) 
            return wav_feat

        elif 'hubert' in tgt_model:
            raise RuntimeError("Hubert path requested but Hubert not loaded. Uncomment Hubert init block to enable.")
        elif 'wavlm' in tgt_model:
            raise RuntimeError("WavLM path requested but WavLM not loaded. Uncomment WavLM init block to enable.")
        elif 'wav2vec2' in tgt_model:
            raise RuntimeError("wav2vec2 path requested but wav2vec2 not loaded. Uncomment wav2vec2 init block to enable.")
        else:
            raise ValueError(f"Unknown tgt_model requested: {tgt_model}. Use 'whisper'/'deepspeech' or enable other models in __init__.")

    @staticmethod
    def save_wav(wav_np: np.ndarray, path: str, sr: int = 16000):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        sf.write(path, wav_np, sr)

    def test_wav_whisper(self, x_1d: torch.Tensor):
        txt = self.whisper_pipe(x_1d.detach().cpu().numpy())['text']
        return txt
    
    def calc_whisper_loss(self, syn_audio: torch.Tensor, tgt_text: str):
        B = syn_audio.size(0)
        input_features_list = []
        for i in range(B):
            wav_cpu = syn_audio[i].detach().cpu().numpy()
            inputs = self.whisper_feature_extractor(
                wav_cpu,
                sampling_rate=self.sample_rate,
                return_tensors="pt"
            )
            input_features_list.append(inputs.input_features)
        
        input_features = torch.cat(input_features_list, dim=0).to(self.device)
        forced_decoder_ids = self.whisper_tokenizer.get_decoder_prompt_ids(
            language="en", 
            task="transcribe"
        )
        
        target_tokens = self.whisper_tokenizer(
            tgt_text,
            return_tensors="pt",
            add_special_tokens=False
        ).input_ids.to(self.device) 
        
        bos_token_id = self.whisper_tokenizer.bos_token_id
        eos_token_id = self.whisper_tokenizer.eos_token_id
        
        prefix_ids = torch.tensor(
            [[id_pair[1] for id_pair in forced_decoder_ids]], 
            device=self.device
        )  
        
        decoder_input_ids = torch.cat([prefix_ids, target_tokens], dim=1)  
        decoder_input_ids = decoder_input_ids.repeat(B, 1) 
        labels = decoder_input_ids.clone()
        labels[:, :-1] = decoder_input_ids[:, 1:]  
        labels[:, -1] = eos_token_id  
        labels[:, :len(forced_decoder_ids)] = -100
        outputs = self.whisper(
            input_features=input_features,
            decoder_input_ids=decoder_input_ids,
            labels=labels
        )
        loss = outputs.loss
        with torch.no_grad():
            generated_ids = self.whisper.generate(
                input_features,
                forced_decoder_ids=forced_decoder_ids,
                max_length=50
            )
            pred_texts = self.whisper_tokenizer.batch_decode(
                generated_ids, 
                skip_special_tokens=True
            )
        return loss, pred_texts
    
    def calc_ds_ctc_loss(self, syn_audio: torch.Tensor, tgt_text: str):
        return self.calc_whisper_loss(syn_audio, tgt_text)

    def get_feat(self, wav: torch.Tensor, tgt_model: str):

        if 'whisper' in tgt_model:
            if wav.dim() == 2:
                assert wav.size(0) == 1, "Whisper path expects batch=1"
                w1d = wav[0]
            else:
                w1d = wav
            w1d = w1d.to(self.device)
            target_length = 30 * self.sample_rate  
            
            if w1d.size(0) < target_length:
                pad_length = target_length - w1d.size(0)
                w1d = torch.nn.functional.pad(w1d, (0, pad_length), mode='constant', value=0)
            elif w1d.size(0) > target_length:
                w1d = w1d[:target_length]

            mel = whisper.log_mel_spectrogram(w1d) 
            mel_b = mel.unsqueeze(0)
            enc_out = self.whisper_encoder(mel_b, output_hidden_states=True)
            hs = enc_out.hidden_states 
            if self.target_layer == 'avg':
                feat = torch.mean(torch.stack(hs), dim=0).squeeze(0)  
            else:
                feat = hs[self.target_layer].squeeze(0)              
            return feat  

        else:
            raise ValueError(f"Unknown tgt_model: {tgt_model}")

    def attack(self):
        sr = self.sample_rate
        target_len = 30 * sr 
        wav, fs = torchaudio.load(self.wav_input)  
        if fs != sr:
            wav = torchaudio.functional.resample(wav, fs, sr)
        if wav.size(-1) < target_len:
            wav = F.pad(wav, (0, target_len - wav.size(-1)))
        else:
            wav = wav[:, :target_len]
        wav = wav.to(self.device)
        wav = normalize_audio(wav) 
        if wav.dim() == 2:
            wav = wav  
        if wav.size(0) == 1:
            wav_b = wav  
        else:
            wav_b = wav[:1, :] 
        wav_tgt, fs2 = torchaudio.load(self.wav_tgt)
        if fs2 != sr:
            wav_tgt = torchaudio.functional.resample(wav_tgt, fs2, sr)
        if wav_tgt.size(-1) < target_len:
            wav_tgt = F.pad(wav_tgt, (0, target_len - wav_tgt.size(-1)))
        else:
            wav_tgt = wav_tgt[:, :target_len]
        wav_tgt = wav_tgt.to(self.device)
        wav_tgt = normalize_audio(wav_tgt)
        if wav_tgt.size(0) == 1:
            wav_tgt_b = wav_tgt
        else:
            wav_tgt_b = wav_tgt[:1, :]

        print("Input shapes -> src:", tuple(wav_b.shape), "tgt:", tuple(wav_tgt_b.shape))
        _ = self.test_wav_whisper(wav_b.squeeze(0))
        org_feat = self.get_feat(wav_b, self.tgt_model).detach()      
        tgt_feat = self.get_feat(wav_tgt_b, self.tgt_model).detach() 
        eps = 0.02      
        lr = 5e-3
        ptb = torch.zeros_like(wav_b, device=self.device).uniform_(-1e-6, 1e-6)
        ptb = torch.nn.Parameter(ptb, requires_grad=True)
        optimizer = torch.optim.Adam([ptb], lr=lr, betas=(0.9, 0.999), weight_decay=0.0)
        tot_loss = 0.0
        start_time = time.time()
        last_log = start_time
        consec_eval_hits = 0
        need_hits = 2 
        for i in range(self.attack_iters):
            adv_wav = normalize_audio(wav_b + ptb)

            asr_loss, txts = self.calc_ds_ctc_loss(adv_wav, self.tgt_text)

            wav_feat = self.get_feat(adv_wav, self.tgt_model)        
            sim_dist = safe_cosine_distance(tgt_feat, wav_feat).mean() 
            asr_term = asr_loss                 
            sr_term = sim_dist * 5.0         

            total_loss = self.asr_weight * asr_term + self.sr_weight * sr_term

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_([ptb], max_norm=0.1)
            optimizer.step()
            with torch.no_grad():
                ptb.data = clamp_perturbation(ptb.data, eps)
            if (i + 1) % 10 == 0:
                et = str(datetime.timedelta(seconds=int(time.time() - start_time)))
                print(f"[{et}] iter {i+1}/{self.attack_iters} | total={total_loss.item():.6f} "
                      f"| asr={asr_loss.item():.6f} | sr={sim_dist.item():.6f} | pred='{txts[0]}'")
            pred_train = txts[0].strip().upper()
            pred_eval = self._stable_eval_decode(adv_wav).strip().upper()
            suc = (pred_eval == self.tgt_text.upper())
            if self.if_slm_loss:
                suc = suc and (sim_dist.item() < 0.2)
            consec_eval_hits = consec_eval_hits + 1 if suc else 0
            if consec_eval_hits >= need_hits:
                print(f"attack success (stable) at iter {i+1}: ASR_train='{pred_train}', ASR_eval='{pred_eval}', SR_dist={sim_dist.item():.4f}")
                break
            if (i + 1) % 10 == 0:
                print("Whisper probe on adv:")
                print(f"[{et}] iter {i+1}/{self.attack_iters} | total={total_loss.item():.6f} "
                      f"| asr={asr_loss.item():.6f} | sr={sim_dist.item():.6f} | pred_train='{pred_train}' | pred_eval='{pred_eval}'")
        final_adv_audio = adv_wav.squeeze(0).detach().cpu().numpy()
        final_output_dir = './final_adversarial_audio'
        os.makedirs(final_output_dir, exist_ok=True)
        input_filename = os.path.basename(self.wav_input)
        sample_id = os.path.splitext(input_filename)[0]
        final_path = os.path.join(final_output_dir, f'{sample_id}_adv_{self.tgt_text.replace(" ", "_")}.wav')
        self.save_wav(final_adv_audio, final_path, sr=self.sample_rate)
        print(f'Final adversarial audio saved: {final_path}')

if __name__ == '__main__':
    set_seed(1337)
    class ArgsNamespace:
        def __init__(self,
                     wav_input='',
                     wav_tgt='',
                     tgt_layer=11,
                     tgt_model='whisper',
                     attack_iters=500,
                     segment_size=48000,
                     tgt_text='OPEN THE DOOR',
                     if_slm_loss=True,
                     asr_weight=1.0,
                     sr_weight=1.0):
            self.wav_input = wav_input
            self.wav_tgt = wav_tgt
            self.attack_iters = attack_iters
            self.segment_size = segment_size
            self.tgt_layer = tgt_layer
            self.tgt_model = tgt_model
            self.tgt_text = tgt_text
            self.if_slm_loss = if_slm_loss
            self.asr_weight = asr_weight
            self.sr_weight = sr_weight

    args = ArgsNamespace(
        wav_input='',
        wav_tgt='',
        tgt_text='OPEN THE DOOR',
        if_slm_loss=True,
        tgt_layer=11,
        tgt_model='whisper',
        attack_iters=500,
        segment_size=48000,
        asr_weight=1.0,
        sr_weight=1.0
    )
    attacker = Attacker(args)
    attacker.attack()
