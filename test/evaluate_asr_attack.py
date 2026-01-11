import argparse
import sys
import torch
import torchaudio
import json
import datetime
from pathlib import Path
from tqdm import tqdm
import jiwer
import warnings
import numpy as np
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir.parent))
sys.path.append(str(current_dir.parent.parent))
warnings.filterwarnings('ignore')

try:
    from asr_model_loader import setup_whisper, setup_deepspeech
except ImportError:
    sys.path.append(str(current_dir))
    from asr_model_loader import setup_whisper, setup_deepspeech

def normalize_audio(audio):
    if audio.abs().max() > 0:
        audio = audio / audio.abs().max()
    return audio

def load_universal_perturbation(delta_path):
    if not Path(delta_path).exists():
        raise FileNotFoundError(f"Delta file not found: {delta_path}")
    checkpoint = torch.load(delta_path, map_location='cuda')
    delta_universal = checkpoint['delta_universal']
    return delta_universal

def apply_perturbation(audio, delta_universal):
    T = audio.size(-1)
    delta_len = delta_universal.size(-1)
    if T <= delta_len:
        perturbed = audio + delta_universal[:, :T]
    else:
        num_repeats = (T // delta_len) + 1
        repeated_delta = delta_universal.repeat(1, num_repeats)[:, :T]
        perturbed = audio + repeated_delta
    return torch.clamp(perturbed, -1, 1)

def calculate_wer(reference, hypothesis):
    try:
        return jiwer.wer(reference, hypothesis)
    except:
        return 1.0

def calculate_cer(reference, hypothesis):
    try:
        return jiwer.cer(reference, hypothesis)
    except:
        return 1.0

def evaluate(args):
    if args.backend == 'whisper':
        predict_fn, model_name = setup_whisper(args)
    elif args.backend == 'deepspeech':
        predict_fn, model_name = setup_deepspeech(args)
    else:
        raise ValueError(f"Unknown backend: {args.backend}")
    delta_universal = load_universal_perturbation(args.delta_path)
    test_dir = Path(args.test_dir)
    test_audios = sorted(test_dir.glob("*.wav"))
    
    if len(test_audios) == 0:
        return 
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target_text_upper = args.target_text.strip().upper()
    results = []
    success_count = 0
    cer_success_count = 0
    total_count = 0
    cer_orig_to_adv = []
    cer_target_to_adv = []
    pbar = tqdm(test_audios, desc="Evaluating")
    for audio_path in pbar:
        try:
            wav, sr = torchaudio.load(audio_path)
            if sr != 16000:
                wav = torchaudio.functional.resample(wav, sr, 16000)
            wav = normalize_audio(wav)
            orig_text = predict_fn(wav.clone())
            adv_wav = apply_perturbation(wav.cuda(), delta_universal)
            adv_text = predict_fn(adv_wav.cpu()) 
            adv_text = predict_fn(adv_wav) 
            is_success = (adv_text == target_text_upper)
            if is_success:
                success_count += 1
            cer_orig = calculate_cer(orig_text, adv_text)
            cer_target = calculate_cer(target_text_upper, adv_text)
            cer_orig_to_adv.append(cer_orig)
            cer_target_to_adv.append(cer_target)
            cer_success = (cer_orig > 0.5)
            if cer_success:
                cer_success_count += 1  
            total_count += 1
            results.append({
                'audio': audio_path.name,
                'duration': wav.size(-1) / 16000,
                'original_text': orig_text,
                'adversarial_text': adv_text,
                'target_text': target_text_upper,
                'success': is_success,
                'cer_success': cer_success,
                'cer_orig_to_adv': cer_orig,
                'cer_target_to_adv': cer_target  
            })

            success_rate = success_count / total_count * 100
            cer_success_rate = cer_success_count / total_count * 100
            pbar.set_postfix({
                'acc': f'{success_rate:.1f}%',
                'cer_acc': f'{cer_success_rate:.1f}%'
            })
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            continue
            
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    target_clean = args.target_text.replace(' ', '_').lower()
    filename = f"{args.backend}_{target_clean}_{timestamp}.json"
    results_json = output_dir / filename
    avg_cer_orig = np.mean(cer_orig_to_adv) if cer_orig_to_adv else 0
    avg_cer_target = np.mean(cer_target_to_adv) if cer_target_to_adv else 0
    
    stats = {
        'model_name': model_name,
        'delta_path': str(args.delta_path),
        'SRoA': cer_success_count / total_count if total_count > 0 else 0,
        'avg_cer_orig_to_adv': float(avg_cer_orig),
        'avg_cer_target_to_adv': float(avg_cer_target),
        'results': results
    }
    with open(results_json, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"SRoA: {cer_success_count}/{total_count} ({cer_success_count/total_count*100:.2f}%)")
    return success_count / total_count if total_count > 0 else 0

def main():
    parser = argparse.ArgumentParser(description='ASR Attack Evaluation')
    parser.add_argument('--backend', type=str, required=True, choices=['whisper', 'deepspeech'])
    parser.add_argument('--delta_path', type=str, 
                       default='./checkpoints/delta_universal_best.pt')
    parser.add_argument('--test_dir', type=str,
                       default='./datasets/universal_attack_testset')
    parser.add_argument('--target_text', type=str,
                       default='OPEN THE DOOR')
    parser.add_argument('--output_dir', type=str,
                       default='./eval_results_asr')
    parser.add_argument('--whisper_model_path', type=str,
                       default='./asr_model/whisper')
    parser.add_argument('--model_size', type=str,
                       default='base',
                       choices=['tiny', 'base', 'small', 'medium', 'large'])        
    args = parser.parse_args()
    evaluate(args)

if __name__ == '__main__':
    main()