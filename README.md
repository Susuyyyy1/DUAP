# DUAP: Dual Universal Adversarial Perturbation

This repository contains the implementation of **DUAP (Dual Universal Adversarial Perturbation)**, a method for generating universal adversarial perturbations that simultaneously attack both Speaker Recognition (SR) and Automatic Speech Recognition (ASR) systems.


## Environment Setup

1.  **Install Dependencies**:
    ```
    pip install -r requirements.txt
    ```


## Data & Model Preparation

### Dataset Sources
Please download the datasets from the official sources:
- **LibriSpeech**: [http://www.openslr.org/12](http://www.openslr.org/12)
- **VCTK**: [https://datashare.ed.ac.uk/handle/10283/3443](https://datashare.ed.ac.uk/handle/10283/3443)

### Pre-trained Models
Please download the pre-trained models from the following links:

**Speaker Recognition (SR)**:
- **ECAPA-TDNN**: [speechbrain/spkrec-ecapa-voxceleb](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb)
- **ResNet34**: [speechbrain/spkrec-resnet-voxceleb](https://huggingface.co/speechbrain/spkrec-resnet-voxceleb)
- **WavLM**: [microsoft/wavlm-base-plus-sv](https://huggingface.co/microsoft/wavlm-base-plus-sv)
- **X-Vector**: [speechbrain/spkrec-xvect-voxceleb](https://huggingface.co/speechbrain/spkrec-xvect-voxceleb)
- **HuBERT**: [facebook/hubert-large-ls960-ft](https://huggingface.co/facebook/hubert-large-ls960-ft)
- **i-vector**: Kaldi-based model (requires manual setup)

**Automatic Speech Recognition (ASR)**:
- **Whisper**: [openai/whisper](https://github.com/openai/whisper)
- **DeepSpeech**: [Mozilla DeepSpeech](https://github.com/SeanNaren/deepspeech.pytorch)

## Training

To train the Universal Adversarial Perturbation:

```bash
python train_uap.py \
    --audio_dir "datasets/train_set" \
    --enroll_dir "datasets/sr_eval_vctk/enroll" \
    --tgt_text "OPEN THE DOOR" \
    --ecapa_dir "sr_model/ecapa" \
    --wavlm_dir "sr_model/wavlm_base_plus_sv" \
    --resnet34_dir "sr_model/resnet34_voxceleb" \
    --batch_size 8 \
    --epochs 5 \
    --lr 0.001 \
    --delta_max 0.3 \
    --use_psy \
    --output_dir "output_result"
```

The trained perturbation will be saved in the `output_result` directory.

## Evaluation

The `test/` directory contains scripts to evaluate the attack performance against various systems.

### Speaker Recognition Attack (CSI)

**ECAPA-TDNN**
```bash
python test/evaluate_sr_attack.py --model_type ecapa --model_dir "models/ecapa" --perturbation_path "outputs/delta.pt"
```

**X-Vector**
```bash
python test/evaluate_sr_attack.py --model_type xvector --model_dir "models/spkrec-xvect-voxceleb" --perturbation_path "outputs/delta.pt"
```

### ASR Attack
Evaluate the attack on ASR systems:

**Whisper**
```bash
python test/evaluate_asr_attack.py --backend whisper --model_size base --delta_path "path/to/delta.pt"
```

### Commercial ASR Attack

**Alibaba**
```bash
python test/commercial_asr/evaluate_asr_attack_alibaba.py --api_key "YOUR_API_KEY" --delta_path "path/to/delta.pt"
```

### Perceptual Evaluation
Evaluate the perceptual quality (SNR, PESQ, NISQA) of the adversarial audio:
```bash
python test/eval_perceptual_uap.py --clean_dir "path/to/clean_wavs" --delta_path "outputs/delta.pt"
```
