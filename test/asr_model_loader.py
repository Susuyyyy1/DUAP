import torch
import sys

def setup_whisper(args):
    import whisper
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_size = getattr(args, 'model_size', 'base')
    whisper_model_path = getattr(args, 'whisper_model_path', './asr_model/whisper')
    model = whisper.load_model(
        model_size,
        device=device,
        download_root=whisper_model_path
    )
    def predict(wav_tensor):
        if isinstance(wav_tensor, torch.Tensor):
            wav_np = wav_tensor.squeeze().cpu().numpy()
        else:
            wav_np = wav_tensor
        result = model.transcribe(
            wav_np,
            language='en',
            fp16=True,
            verbose=False
        )
        return result['text'].strip().upper()
    return predict, f"whisper-{model_size}"
