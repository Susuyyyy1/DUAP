# [ICME'26]DUAP: Dual Universal Adversarial Perturbation

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
## Citation

If you use this code or its parts in your research, please cite the following paper:

```bibtex
@article{ge2023advddos,
  title={Advddos: Zero-query adversarial attacks against commercial speech recognition systems},
  author={Ge, Yunjie and Zhao, Lingchen and Wang, Qian and Duan, Yiheng and Du, Minxin},
  journal={IEEE Transactions on Information Forensics and Security},
  volume={18},
  pages={3647--3661},
  year={2023},
  publisher={IEEE}
}

@inproceedings{zong2021targeted,
  title={Targeted universal adversarial perturbations for automatic speech recognition},
  author={Zong, Wei and Chow, Yang-Wai and Susilo, Willy and Rana, Santu and Venkatesh, Svetha},
  booktitle={International Conference on Information Security},
  pages={358--373},
  year={2021},
}

@inproceedings{xie2020real,
  title={Real-time, universal, and robust adversarial attacks against speaker recognition systems},
  author={Xie, Yi and Shi, Cong and Li, Zhuohang and Liu, Jian and Chen, Yingying and Yuan, Bo},
  booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing},
  pages={1738--1742},
  year={2020},
}

@inproceedings{hanina2024universal,
  title={Universal adversarial attack against speaker recognition models},
  author={Hanina, Shoham and Zolfi, Alon and Elovici, Yuval and Shabtai, Asaf},
  booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing},
  pages={4860--4864},
  year={2024},
  organization={IEEE}
}

@inproceedings{chen2023qfa2sr,
  title={QFA2SR: Query-free adversarial transfer attacks to speaker recognition systems},
  author={Chen, Guangke and Zhang, Yedi and Zhao, Zhe and Song, Fu},
  booktitle={USENIX Security Symposium},
  pages={2437--2454},
  year={2023}
}

@inproceedings{desplanques2020ecapa,
  title={ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN based speaker verification},
  author={Desplanques, Brecht and Thienpondt, Jenthe and Demuynck, Kris},
  booktitle={21st Annual conference of the International Speech Communication Association},
  pages={3830--3834},
  year={2020}
}

@inproceedings{madry2018towards,
  title={Towards deep learning models resistant to adversarial attacks},
  author={Madry, Aleksander and Makelov, Aleksandar and Schmidt, Ludwig and Tsipras, Dimitris and Vladu, Adrian},
  booktitle={International Conference on Learning Representations},
  year={2018}
}

@article{chi2024adversarial,
  title={Adversarial attacks on autonomous driving systems in the physical world: a survey},
  author={Chi, Lijun and Msahli, Mounira and Zhang, Qingjie and Qiu, Han and Zhang, Tianwei and Memmi, Gerard and Qiu, Meikang},
  journal={IEEE Transactions on Intelligent Vehicles},
  year={2024},
  publisher={IEEE}
}

@article{bhanushali2024adversarial,
  title={Adversarial attacks on automatic speech recognition (asr): A survey},
  author={Bhanushali, Amisha Rajnikant and Mun, Hyunjun and Yun, Joobeom},
  journal={IEEE Access},
  volume={12},
  pages={88279--88302},
  year={2024},
}

@inproceedings{zhang2024laseradv,
  title={LaserAdv: Laser adversarial attacks on speech recognition systems},
  author={Zhang, Guoming and Ma, Xiaohui and Zhang, Huiting and Xiang, Zhijie and Ji, Xiaoyu and Yang, Yanni and Cheng, Xiuzhen and Hu, Pengfei},
  booktitle={USENIX Security Symposium},
  pages={3945--3961},
  year={2024}
}

@inproceedings{jin2025whispering,
  author = {Jin, Weifei and Cao, Yuxin and Su, Junjie and Wang, Derui and Zhang, Yedi and Xue, Minhui and Hao, Jie and Dong, Jin Song and Yang, Yixian},
  title = {Whispering Under the Eaves: Protecting User Privacy Against Commercial and LLM-powered Automatic Speech Recognition Systems},
  booktitle = {USENIX Security Symposium},
  year = {2025},
}

@inproceedings{yu2023smack,
  title={SMACK: Semantically meaningful adversarial audio attack},
  author={Yu, Zhiyuan and Chang, Yuanhaur and Zhang, Ning and Xiao, Chaowei},
  booktitle={USENIX security symposium},
  pages={3799--3816},
  year={2023}
}

@inproceedings{carlini2018audio,
  title={Audio adversarial examples: Targeted attacks on speech-to-text},
  author={Carlini, Nicholas and Wagner, David},
  booktitle={2018 IEEE Security and Privacy Workshops},
  pages={1--7},
  year={2018},
}

@inproceedings{zhang2017dolphinattack,
  title={Dolphinattack: Inaudible voice commands},
  author={Zhang, Guoming and Yan, Chen and Ji, Xiaoyu and Zhang, Tianchen and Zhang, Taimin and Xu, Wenyuan},
  booktitle={ACM SIGSAC Conference on Computer and Communications Security},
  pages={103--117},
  year={2017}
}

@inproceedings{fang2024zero,
  title={Zero-query adversarial attack on black-box automatic speech recognition systems},
  author={Fang, Zheng and Wang, Tao and Zhao, Lingchen and Zhang, Shenyi and Li, Bowen and Ge, Yunjie and Li, Qi and Shen, Chao and Wang, Qian},
  booktitle={ACM SIGSAC Conference on Computer and Communications Security},
  pages={630--644},
  year={2024}
}

@inproceedings{snyder2018x,
  title={X-vectors: Robust dnn embeddings for speaker recognition},
  author={Snyder, David and Garcia-Romero, Daniel and Sell, Gregory and Povey, Daniel and Khudanpur, Sanjeev},
  booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing},
  pages={5329--5333},
  year={2018},
  organization={IEEE}
}

@article{hsu2021hubert,
  title={Hubert: Self-supervised speech representation learning by masked prediction of hidden units},
  author={Hsu, Wei-Ning and Bolte, Benjamin and Tsai, Yao-Hung Hubert and Lakhotia, Kushal and Salakhutdinov, Ruslan and Mohamed, Abdelrahman},
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing},
  volume={29},
  pages={3451--3460},
  year={2021},
}

@inproceedings{radford2023robust,
  title={Robust speech recognition via large-scale weak supervision},
  author={Radford, Alec and Kim, Jong Wook and Xu, Tao and Brockman, Greg and McLeavey, Christine and Sutskever, Ilya},
  booktitle={International Conference on Machine Learning},
  pages={28492--28518},
  year={2023},
}

@article{gulati2020conformer,
  title={Conformer: Convolution-augmented transformer for speech recognition},
  author={Gulati, Anmol and Qin, James and Chiu, Chung-Cheng and Parmar, Niki and Zhang, Yu and Yu, Jiahui and Han, Wei and Wang, Shibo and Zhang, Zhengdong and Wu, Yonghui and others},
  journal={arXiv preprint arXiv:2005.08100},
  year={2020}
}

@article{neekhara2019universal,
  title={Universal adversarial perturbations for speech recognition systems},
  author={Neekhara, Paarth and Hussain, Shehzeen and Pandey, Prakhar and Dubnov, Shlomo and McAuley, Julian and Koushanfar, Farinaz},
  journal={arXiv preprint arXiv:1905.03828},
  year={2019}
}

@article{chen2022wavlm,
  title={Wavlm: Large-scale self-supervised pre-training for full stack speech processing},
  author={Chen, Sanyuan and Wang, Chengyi and Chen, Zhengyang and Wu, Yu and Liu, Shujie and Chen, Zhuo and Li, Jinyu and Kanda, Naoyuki and Yoshioka, Takuya and Xiao, Xiong and others},
  journal={IEEE Journal of Selected Topics in Signal Processing},
  volume={16},
  number={6},
  pages={1505--1518},
  year={2022},
}

@article{dehak2010front,
  title={Front-end factor analysis for speaker verification},
  author={Dehak, Najim and Kenny, Patrick J and Dehak, R{\'e}da and Dumouchel, Pierre and Ouellet, Pierre},
  journal={IEEE Transactions on Audio, Speech, and Language Processing},
  volume={19},
  number={4},
  pages={788--798},
  year={2010},
}

@inproceedings{panayotov2015librispeech,
  title={Librispeech: an asr corpus based on public domain audio books},
  author={Panayotov, Vassil and Chen, Guoguo and Povey, Daniel and Khudanpur, Sanjeev},
  booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing},
  pages={5206--5210},
  year={2015},
}

@inproceedings{amodei2016deep,
  title={Deep speech 2: End-to-end speech recognition in english and mandarin},
  author={Amodei, Dario and Ananthanarayanan, Sundaram and Anubhai, Rishita and Bai, Jingliang and Battenberg, Eric and Case, Carl and Casper, Jared and Catanzaro, Bryan and Cheng, Qiang and Chen, Guoliang and others},
  booktitle={International Conference on Machine Learning},
  pages={173--182},
  year={2016},
}

@article{yamagishi2019cstr,
  title={Cstr vctk corpus: English multi-speaker corpus for cstr voice cloning toolkit (version 0.92)},
  author={Yamagishi, Junichi and Veaux, Christophe and MacDonald, Kirsten and others},
  journal={University of Edinburgh. The Centre for Speech Technology Research (CSTR)},
  year={2019}
}


@inproceedings{he2016deep,
  title={Deep residual learning for image recognition},
  author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
  pages={770--778},
  year={2016}
}

@article{mittag2021nisqa,
  title={NISQA: A deep CNN-self-attention model for multidimensional speech quality prediction with crowdsourced datasets},
  author={Mittag, Gabriel and Naderi, Babak and Chehadi, Assmaa and M{\"o}ller, Sebastian},
  journal={arXiv preprint arXiv:2104.09494},
  year={2021}
}

@inproceedings{wu2023kenku,
  title={KENKU: Towards efficient and stealthy black-box adversarial attacks against ASR systems},
  author={Wu, Xinghui and Ma, Shiqing and Shen, Chao and Lin, Chenhao and Wang, Qian and Li, Qi and Rao, Yuan},
  booktitle={USENIX Security Symposium},
  pages={247--264},
  year={2023}
}

@misc{alibaba,
  title = {Alibaba ASR},
  howpublished = "\url{https://ai.aliyun.com/nls/asr}",
  year = {2025},
  month = {Nov},
}

@misc{iflytek,
  title = {IFLYTEK Speech to text},
  howpublished = "\url{https://global.xfyun.cn/products/speech-to-text}",
  year = {2025},
  month = {Nov},
}

@misc{tencent,
  title = {Tencent Cloud Sentence Recognition},
  howpublished = "\url{https://console.cloud.tencent.com/asr}",
  year = {2025},
  month = {Nov},
}

@inproceedings{cheng2024alif,
  title={Alif: Low-cost adversarial audio attacks on black-box speech platforms using linguistic features},
  author={Cheng, Peng and Wang, Yuwei and Huang, Peng and Ba, Zhongjie and Lin, Xiaodong and Lin, Feng and Lu, Li and Ren, Kui},
  booktitle={2024 IEEE Symposium on Security and Privacy (SP)},
  pages={1628--1645},
  year={2024},
  organization={IEEE}
}

@article{li2023inaudible,
  title={Inaudible adversarial perturbation: Manipulating the recognition of user speech in real time},
  author={Li, Xinfeng and Yan, Chen and Lu, Xuancun and Zeng, Zihan and Ji, Xiaoyu and Xu, Wenyuan},
  journal={arXiv preprint arXiv:2308.01040},
  year={2023}
}

@book{zwicker2013psychoacoustics,
  title={Psychoacoustics: Facts and models},
  author={Zwicker, Eberhard and Fastl, Hugo},
  volume={22},
  year={2013},
  publisher={Springer Science \& Business Media}
}
@inproceedings{jinalmguard,
  title={ALMGuard: Safety Shortcuts and Where to Find Them as Guardrails for Audio--Language Models},
  author={Jin, Weifei and Cao, Yuxin and Su, Junjie and Xue, Jason and Hao, Jie and Xu, Ke and Dong, Jin Song and Wang, Derui},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems}
}
