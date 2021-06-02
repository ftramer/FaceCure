# FaceCure

Oblivious and adaptive "defenses" against poisoning attacks on facial recognition systems.


## Generate cloaks with Fawkes v1.0

Download Fawkes from https://github.com/Shawn-Shan/fawkes

*WARNING* For the `--no-align` option to work properly (Fawkes won't try to detect faces), we need to patch `protection.py`
so that the option `eval_local=True` is passed to the `Faces` class.

Then, we generate protected pictures for one FaceScrub user as follows:

```bash
python3 fawkes/protection.py --gpu 0 -d facescrub/download/Adam_Sandler/face --batch-size 8 -m high --no-align
```

For each picture `filename.png`, this will create an attacked picture `filename_high_cloaked.png` in the same directory.

## Generate cloaks with LowKey

Download LowKey here: https://openreview.net/forum?id=hJmtwocEqzc

We modified the attack script slightly so that the attack does not try to align faces.
As a result, the attack can also be batched. The attack automatically resizes pictures to 112x112 pixels as LowKey does not seem to work well with larger pictures.

```bash
python3 lowkey_attack.py facescrub/download/Adam_Sandler/face
```

For each picture `filename.png`, this will create a resized picture `filename_small.png` and an attacked picture `filename_attacked.png` in the same directory.

## Evaluation

The evaluation code assumes that Fawkesv0.3 is on your PYTHONPATH.
Download it from here: https://github.com/Shawn-Shan/fawkes/releases/tag/v0.3

We assume you have:
- A directory with the original FaceScrub pictures: `facescrub/download/`
- A directory with users protected by Fawkes: `facescrub_fawkes_attack/download/`
- A directory with users protected by LowKey: `facescrub_lowkey_attack/download/`

### Baseline with NN classifier
Train a nearest neighbor classifier on top of the Fawkesv03 feature extractor, with one attacking user.

To evaluate Fawkes' attack:
```bash
python3 eval.py --facescrub-dir facescrub/download/ --attack-dir facescrub_fawkes_attack/download/ --unprotected-file-match .jpg --protected-file-match high_cloaked.png --classifier NN --names-list Adam_Sandler 
```

Result: 
```
Protection rate: 0.97
```

To evaluate LowKey's attack:
```bash
python3 eval.py --facescrub-dir facescrub/download/ --attack-dir facescrub_lowkey_attack/download/ --unprotected-file-match small.png --protected-file-match attacked.png --classifier  NN --names-list Adam_Sandler 
```

Result:
```
Protection rate: 0.94
```

### Oblivious NN classifier
We can repeat the experiment using the feature extractor from MagFace or CLIP.

- For MagFace:
    - Download MagFace: https://github.com/IrvingMeng/MagFace/
    - Download the pre-trained `iResNet100` model
    - Run ```python3 eval_oblivious.py --facescrub-dir facescrub/download/ --attack-dir facescrub_lowkey_attack/download/ --unprotected-file-match small.png --protected-file-match attacked.png --classifier  NN --names-list Adam_Sandler --model magface --resume path/to/magface_model.pth```

Result:
```
Protection rate: 1.00
```

- For CLIP:
    - Download CLIP: https://github.com/openai/CLIP
    - Run ```python3 eval_oblivious.py --facescrub-dir facescrub/download/ --attack-dir facescrub_lowkey_attack/download/ --unprotected-file-match small.png --protected-file-match    attacked.png --classifier  NN --names-list Adam_Sandler --model clip```

Result
```
Protection rate: 0.24
```

### Adaptive NN classifier
Same as for the baseline classifier above, but you can add the option `--robust-weights cp-robust-10.ckpt` to use a robustified feature extractor.
This feature extractor was trained using the `train_robust_features.py` script, which finetunes a feature extractor on known attack pictures.

Result for Fawkes:
```
Protection rate: 0.03
```

Result for LowKey:
```
Protection rate: 0.03
```

### Linear classifiers
In all the above examples, you can set `--classifier linear` to instead train a linear classifier instead of a nearest neighbor one.

### Baseline end-to-end
Train a classifier on top of the Fawkesv03 feature extractor end-to-end, with one attacking user.

To evaluate Fawkes' attack:
```bash
python3 eval_e2e.py --gpu 0 --attack-dir facescrub_fawkes_attack/download/Adam_Sandler/face --facescrub-dir facescrub/download/ --unprotected-file-match .jpg --protected-file-match high_cloaked.png
```

Result:
```
Protection rate: 0.88
```

To evaluate LowKey's attack:
```bash
python3 eval_e2e.py --gpu 0 --attack-dir facescrub_lowkey_attack/download/Adam_Sandler/face --facescrub-dir facescrub/download/ --unprotected-file-match small.png --protected-file-match attacked.png
```

Result:
```
Protection rate: 0.97
```

### Adaptive end-to-end
To evaluate robust end-to-end training, we add attacked pictures into the model's training set.
We assume here that at most half of the FaceScrub users have attacked pictures.

To evaluate Fawkes' attack:
```bash
python3 eval_e2e.py --gpu 0 --attack-dir facescrub_fawkes_attack/download/Adam_Sandler/face --facescrub-dir facescrub/download/ --unprotected-file-match .jpg --protected-file-match     high_cloaked.png --robust --public-attack-dirs facescrub_fawkes_attack/download facescrub_lowkey_attack/download
```

Result:
```
Protection rate: 0.03
```

To evaluate LowKey's attack:
```bash
python3 eval_e2e.py --gpu 0 --attack-dir facescrub_lowkey_attack/download/Adam_Sandler/face --facescrub-dir facescrub/download/ --unprotected-file-match small.png --protected-file-     match attacked.png --robust --public-attack-dirs facescrub_fawkes_attack/download facescrub_lowkey_attack/download
```

Result:
```
Protection rate: 0.03
```

In our evaluation, we also included attacked pictures computed with Fawkes' version 0.3 attack.
