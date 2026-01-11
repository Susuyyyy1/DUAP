import torch
import torchaudio
from pathlib import Path
import argparse
from tqdm import tqdm
import json
import warnings
import time
import os
import jiwer
from typing import Optional
try:
    import dashscope
    from dashscope.audio.asr import Recognition
except ImportError:
    print("Error: DashScope SDK not installed!")
    print("Please install: pip install dashscope")
    exit(1)
warnings.filterwarnings('ignore')
class DashScopeASRClient:
    def __init__(self, api_key: str):
        dashscope.api_key = api_key
    def transcribe_file(self, audio_path: str, model: str = 'paraformer-realtime-v2', sample_rate: int = 16000) -> Optional[str]:
        try:
            recognition = Recognition(model=model,
                                     format='wav',
                                     sample_rate=sample_rate,
                                     callback=None)
            result = recognition.call(audio_path)
            if isinstance(result, dict):
                if result.get('code') or result.get('status_code') != 200:
                    print(f"API Error Response: {result}")
                    return None
                output = result.get('output')
                if output:
                    if output.get('text'):
                        text = output['text'].strip().upper()
                        return text

                    if output.get('sentence') and len(output['sentence']) > 0:
                        sentences = [s['text'] for s in output['sentence'] if s.get('text')]
                        if sentences:
                            text = ' '.join(sentences).strip().upper()
                            return text
                if output is None:
                    return ""       
            elif hasattr(result, 'output'):
                if hasattr(result.output, 'text'):
                    text = result.output.text.strip().upper()
                    return text
                elif hasattr(result.output, 'sentence') and len(result.output.sentence) > 0:
                    sentences = [s.text for s in result.output.sentence if hasattr(s, 'text')]
                    if sentences:
                        text = ' '.join(sentences).strip().upper()
                        return text
            return None
        except Exception as e:
            print(f"API Error: {e}")
            import traceback
            traceback.print_exc()
            return None
def normalize_audio(audio):
    if audio.abs().max() > 0:
        audio = audio / audio.abs().max()
    return audio
def load_universal_perturbation(delta_path):
    checkpoint = torch.load(delta_path, map_location='cuda')
    delta_universal = checkpoint['delta_universal']
    print(f"Loaded universal perturbation from {delta_path}")
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
    perturbed = torch.clamp(perturbed, -1, 1)
    return perturbed
def save_audio_temp(audio_tensor, sr=16000):
    import tempfile
    if audio_tensor.dim() == 1:
        audio_tensor = audio_tensor.unsqueeze(0)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    temp_path = temp_file.name
    temp_file.close()
    torchaudio.save(temp_path, audio_tensor.cpu(), sr)
    return temp_path
def calculate_wer(reference, hypothesis):
    try:
        wer = jiwer.wer(reference, hypothesis)
        return wer
    except:
        return 1.0
def calculate_cer(reference, hypothesis):
    try:
        cer = jiwer.cer(reference, hypothesis)
        return cer
    except:
        return 1.0
def evaluate_on_testset(
    delta_path, 
    test_dir, 
    target_text, 
    api_key,
    output_dir,
    rate_limit=0.5
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    delta_universal = load_universal_perturbation(delta_path)
    asr_client = DashScopeASRClient(api_key=api_key)
    test_audios = sorted(Path(test_dir).glob("*.wav"))
    print(f"\nFound {len(test_audios)} test audios")
    if len(test_audios) == 0:
        print(f"No audio files found in {test_dir}")
        return
    results = []
    success_count = 0  
    cer_success_count = 0  
    total_count = 0
    api_error_count = 0
    cer_orig_to_adv = []
    cer_target_to_adv = []
    target_text_upper = target_text.strip().upper()
    print(f"\nTarget text: '{target_text_upper}'")
    print(f"Testing on {len(test_audios)} audios...")
    print(f"API rate limit: {rate_limit}s between calls\n")
    pbar = tqdm(test_audios, desc="Evaluating")
    for audio_path in pbar:
        temp_orig = None
        temp_adv = None
        try:
            wav, sr = torchaudio.load(audio_path)
            if sr != 16000:
                wav = torchaudio.functional.resample(wav, sr, 16000)
            wav = normalize_audio(wav)
            temp_orig = save_audio_temp(wav, sr=16000)
            time.sleep(rate_limit)  
            orig_text = asr_client.transcribe_file(temp_orig)
            if orig_text is None:
                api_error_count += 1
                print(f"\nAPI Error for {audio_path.name} (original)")
                continue
            wav_cuda = wav.cuda()
            adv_wav = apply_perturbation(wav_cuda, delta_universal)
            temp_adv = save_audio_temp(adv_wav, sr=16000)
            time.sleep(rate_limit)  
            adv_text = asr_client.transcribe_file(temp_adv)
            if adv_text is None:
                api_error_count += 1
                print(f"\n API Error for {audio_path.name} (adversarial)")
                continue
            if adv_text == "":
                pass  
            is_success = (adv_text == target_text_upper)
            if is_success:
                success_count += 1
            total_count += 1
            cer_orig = calculate_cer(orig_text, adv_text)
            cer_orig_to_adv.append(cer_orig)
            cer_target = calculate_cer(target_text_upper, adv_text)
            cer_target_to_adv.append(cer_target)
            cer_success = (cer_orig > 0.5)
            if cer_success:
                cer_success_count += 1
            result = {
                'audio': audio_path.name,
                'duration': wav.size(-1) / 16000,
                'original_text': orig_text,
                'adversarial_text': adv_text,
                'target_text': target_text_upper,
                'success': is_success,
                'cer_success': cer_success,  
                'cer_orig_to_adv': cer_orig,
                'cer_target_to_adv': cer_target
            }
            results.append(result)
            success_rate = success_count / total_count * 100 if total_count > 0 else 0
            cer_success_rate = cer_success_count / total_count * 100 if total_count > 0 else 0
            pbar.set_postfix({
                'exact': f'{success_count}/{total_count}',
                'cer': f'{cer_success_count}/{total_count}',
                'rate': f'{success_rate:.1f}%/{cer_success_rate:.1f}%',
                'api_err': api_error_count
            })
        finally:
            if temp_orig and os.path.exists(temp_orig):
                os.remove(temp_orig)
            if temp_adv and os.path.exists(temp_adv):
                os.remove(temp_adv)
    import datetime
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    target_clean = target_text.replace(' ', '_').lower()
    results_json = output_dir / f'alibaba_asr_{target_clean}_{timestamp}.json'
    avg_cer_orig = sum(cer_orig_to_adv) / len(cer_orig_to_adv) if cer_orig_to_adv else 0
    avg_cer_target = sum(cer_target_to_adv) / len(cer_target_to_adv) if cer_target_to_adv else 0
    with open(results_json, 'w', encoding='utf-8') as f:
        json.dump({
            'model': 'dashscope-qwen-asr',
            'target_text': target_text_upper,
            'total_audios': total_count,
            'SRoA': cer_success_count / total_count if total_count > 0 else 0,
            'avg_cer_orig_to_adv': avg_cer_orig,
            'avg_cer_target_to_adv': avg_cer_target,
            'results': results
        }, f, indent=2, ensure_ascii=False)
    print(f"SRoA: {cer_success_count}/{total_count} ({cer_success_count/total_count*100:.2f}%)")
    failed_cases = [r for r in results if not r['success']]
    success_cases = [r for r in results if r['success']]
    return success_count / total_count if total_count > 0 else 0
def main():
    parser = argparse.ArgumentParser(description='Evaluate Universal Perturbation on Qwen-ASR ')
    parser.add_argument('--api_key', type=str, required=True,
                       help='DashScope API Key')
    parser.add_argument('--delta_path', type=str, 
                       default='./outputs/ecapa_single_model/delta_universal_final.pt',
                       help='Path to universal perturbation checkpoint')
    parser.add_argument('--test_dir', type=str,
                       default='./datasets/universal_attack_testset',
                       help='Directory containing test audio files')
    parser.add_argument('--target_text', type=str,
                       default='OPEN THE DOOR',
                       help='Target text for ASR attack')
    parser.add_argument('--output_dir', type=str,
                       default='./eval_results_asr',
                       help='Directory to save evaluation results')
    parser.add_argument('--rate_limit', type=float,
                       default=0.5,
                       help='Seconds between API calls')
    args = parser.parse_args()
    success_rate = evaluate_on_testset(
        delta_path=args.delta_path,
        test_dir=args.test_dir,
        target_text=args.target_text,
        api_key=args.api_key,
        output_dir=args.output_dir,
        rate_limit=args.rate_limit
    )
    return success_rate
if __name__ == '__main__':
    main()
