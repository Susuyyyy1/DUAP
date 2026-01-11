import torch
import torch.nn.functional as F
import tempfile
from speechbrain.inference.speaker import EncoderClassifier
from transformers import WavLMForXVector

class SRModelManager:
    def __init__(self, device):
        self.device = device
        self.models = {}
        self.model_names = []

    def load_ecapa(self, model_path):
        print(f"Loading ECAPA from: {model_path}")
        tmpdir = tempfile.mkdtemp()
        model = EncoderClassifier.from_hparams(
            source=model_path,
            savedir=tmpdir,
            run_opts={"device": str(self.device)},
            use_auth_token=False
        )
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        self.models['ecapa'] = model
        if 'ecapa' not in self.model_names:
            self.model_names.append('ecapa')

    def load_wavlm(self, model_path):
        print(f"Loading WavLM from: {model_path}")
        model = WavLMForXVector.from_pretrained(model_path).eval().to(self.device)
        for param in model.parameters():
            param.requires_grad = False
        self.models['wavlm'] = model
        if 'wavlm' not in self.model_names:
            self.model_names.append('wavlm')

    def load_resnet34(self, model_path):
        print(f"Loading ResNet34 from: {model_path}")
        tmpdir = tempfile.mkdtemp()
        model = EncoderClassifier.from_hparams(
            source=model_path,
            savedir=tmpdir,
            run_opts={"device": str(self.device)},
            use_auth_token=False
        )
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        self.models['resnet'] = model
        if 'resnet' not in self.model_names:
            self.model_names.append('resnet')

    def embed(self, wav_16k, model_name):
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded.")
        if model_name == 'ecapa':
            return self._embed_ecapa(wav_16k)
        elif model_name == 'wavlm':
            return self._embed_wavlm(wav_16k)
        elif model_name == 'resnet':
            return self._embed_resnet(wav_16k)
        else:
            raise NotImplementedError(f"Embedding logic for {model_name} not implemented.")

    def _embed_ecapa(self, wav_16k):
        model = self.models['ecapa']
        if wav_16k.dim() == 1:
            wav_16k = wav_16k.unsqueeze(0)
        wav_16k = wav_16k.to(self.device)
        feats = model.mods.compute_features(wav_16k)
        lens = torch.tensor([feats.shape[1]], device=feats.device)
        feats = model.mods.mean_var_norm(feats, lens)
        emb = model.mods.embedding_model(feats)
        emb = emb.squeeze().reshape(-1)
        emb = F.normalize(emb.unsqueeze(0), p=2, dim=-1).squeeze(0)
        return emb

    def _embed_wavlm(self, wav_16k):
        model = self.models['wavlm']
        if wav_16k.dim() == 1:
            wav_16k = wav_16k.unsqueeze(0)
        wav = wav_16k.to(self.device)
        wav = wav - wav.mean(dim=1, keepdim=True)
        wav = wav / (wav.std(dim=1, keepdim=True) + 1e-7)
        inputs = {"input_values": wav}
        outputs = model(**inputs)
        emb = outputs.embeddings
        emb = emb.squeeze(0)
        emb = F.normalize(emb.unsqueeze(0), p=2, dim=-1).squeeze(0)
        return emb

    def _embed_resnet(self, wav_16k):
        model = self.models['resnet']
        if wav_16k.dim() == 1:
            wav_16k = wav_16k.unsqueeze(0)
        wav_16k = wav_16k.to(self.device)
        feats = model.mods.compute_features(wav_16k)
        lens = torch.tensor([feats.shape[1]], device=feats.device)
        feats = model.mods.mean_var_norm(feats, lens)
        emb = model.mods.embedding_model(feats)
        emb = emb.squeeze().reshape(-1)
        emb = F.normalize(emb.unsqueeze(0), p=2, dim=-1).squeeze(0)
        return emb

    def set_eval(self):
        for model in self.models.values():
            model.eval()
