# FaceCure

**Oblivious and adaptive "defenses" against poisoning attacks on facial recognition systems.**

We consider three common facial recognition approaches: 
- *NN:* 1-Nearest Neighbor on top of a feature extractor.
- *Linear:* Linear fine-tuning on top of a frozen feature extractor for 10 epochs.
- *End-to-end:* End-to-end fine-tuning of the feature extractor and linear classifier for 10 epochs.

We evaluate the following defenses against the Fawkes and Lowkey attacks on facial recognition:
- *Oblivious:* A model trainer waits for a new model to nullify the protection of previously collected pictures.
- *Adaptive:* A model trainer trains a robust model that resists the perturbations of collected pictures.

We perform all of our experiments with the *FaceScrub* dataset, which contains over 50,000 images of 530 celebrities. We use the official aligned faces from this dataset and thus disable the automatic face-detection routines in Fawkes and LowKey.

## Perturb images with Fawkes v1.0

Download Fawkesv1.0 here: https://github.com/Shawn-Shan/fawkes

*WARNING:* For the `--no-align` option to work properly, we need to patch `fawkes/protection.py`
so that the option `eval_local=True` is passed to the `Faces` class.

Then, we generate protected pictures for one FaceScrub user as follows:

```bash
python3 fawkes/protection.py --gpu 0 -d facescrub/download/Adam_Sandler/face --batch-size 8 -m high --no-align
```

For each picture `filename.png`, this will create an attacked picture `filename_high_cloaked.png` in the same directory.

Original Picture | Picture attacked with Fawkes
-----------------|-----------------
<img src="adam.jpg" alt="original picture" width="224"/> | <img src="adam_fawkes.png" alt="attacked picture" width="224"/>

## Perturb images with LowKey

Download LowKey here: https://openreview.net/forum?id=hJmtwocEqzc

We modified the attack script slightly so that the attack does not try to align faces.
As a result, the attack can also be batched. The attack automatically resizes pictures to 112x112 pixels as LowKey does not seem to work well with larger pictures.

```bash
python3 lowkey_attack.py facescrub/download/Adam_Sandler/face
```

For each picture `filename.png`, this will create a resized picture `filename_small.png` and an attacked picture `filename_attacked.png` in the same directory.

Original Picture | Picture attacked with LowKey
-----------------|-----------------
<img src="adam.jpg" alt="original picture" width="224"/> | <img src="adam_lowkey.png" alt="attacked picture" width="224"/>

## Evaluation setup

The evaluation code assumes that Fawkesv0.3 is on your PYTHONPATH.

Download Fawkesv0.3 here: https://github.com/Shawn-Shan/fawkes/releases/tag/v0.3

We assume you have:
- A directory with the original FaceScrub pictures: `facescrub/download/`
- A directory with users protected by Fawkes: `facescrub_fawkes_attack/download/`
- A directory with users protected by LowKey: `facescrub_lowkey_attack/download/`

## Baseline evaluation with NN and linear classifiers

### NN classifier
Train a nearest neighbor classifier on top of the Fawkesv0.3 feature extractor, with one attacking user.

To evaluate Fawkes' attack:
```bash
python3 eval.py --facescrub-dir facescrub/download/ --attack-dir facescrub_fawkes_attack/download/ --unprotected-file-match .jpg --protected-file-match high_cloaked.png --classifier NN --names-list Adam_Sandler 
```

To evaluate LowKey's attack:
```bash
python3 eval.py --facescrub-dir facescrub/download/ --attack-dir facescrub_lowkey_attack/download/ --unprotected-file-match small.png --protected-file-match attacked.png --classifier  NN --names-list Adam_Sandler 
```

Results:

Fawkes (baseline NN) | Lowkey (baseline NN)
---------------------|---------------------
```Protection rate: 0.97```|```Protection rate: 0.94```

### Linear classifier
You can set `--classifier linear` to instead train a linear classifier instead of a nearest neighbor one.

## Attack evaluation with NN and linear classifiers 

### Oblivious NN classifier
We can repeat the experiment using the feature extractor from Fawkes v1.0, MagFace or CLIP.

- For Fawkes v1.0:
    - Put Fawkes v1.0 (and not Fawkes v0.3) on your PYTHONPATH

- For MagFace:
    - Download MagFace here: https://github.com/IrvingMeng/MagFace/
    - Download the pre-trained `iResNet100` model

- For CLIP:
    - Download CLIP here: https://github.com/openai/CLIP


Fawkes attack & Fawkes v1.0 extractor:
```bash
python3 eval_oblivious.py --facescrub-dir facescrub/download/ --attack-dir facescrub_fawkes_attack/download/ --unprotected-file-match .jpg --protected-file-match high_cloaked.png --classifier  NN --names-list Adam_Sandler --model fawkesv10
```

Fawkes attack & MagFace extractor:
```bash
python3 eval_oblivious.py --facescrub-dir facescrub/download/ --attack-dir facescrub_fawkes_attack/download/ --unprotected-file-match .jpg --protected-file-match high_cloaked.png --classifier  NN --names-list Adam_Sandler --model magface --resume path/to/magface_model.pth
```

LowKey attack & MagFace extractor:
```bash
python3 eval_oblivious.py --facescrub-dir facescrub/download/ --attack-dir facescrub_lowkey_attack/download/ --unprotected-file-match small.png --protected-file-match attacked.png --classifier  NN --names-list Adam_Sandler --model magface --resume path/to/magface_model.pth
```

LowKey attack & CLIP extractor: 
```bash
python3 eval_oblivious.py --facescrub-dir facescrub/download/ --attack-dir facescrub_lowkey_attack/download/ --unprotected-file-match small.png --protected-file-match attacked.png --classifier  NN --names-list Adam_Sandler --model clip
```

Results:

Fawkes attack & Fawkes extractor | Fawkes attack & MagFace extractor | | LowKey attack & MagFace extractor | LowKey attack & CLIP extractor
---------------------------------|----------------------------------|-|-----------------------------------|-------------------------------
```Protection rate: 1.00```|```Protection rate: 0.00```| |```Protection rate: 1.00```|```Protection rate: 0.24```

### Adaptive NN classifier
Same as for the baseline classifier above, but you can add the option `--robust-weights cp-robust-10.ckpt` to use a robustified feature extractor.
This feature extractor was trained using the `train_robust_features.py` script, which finetunes a feature extractor on known attack pictures.

Results:

Fawkes (adaptive NN) | Lowkey (adaptive NN)
---------------------|---------------------
```Protection rate: 0.03```|```Protection rate: 0.03```

### Linear classifiers
You can set `--classifier linear` to instead train a linear classifier instead of a nearest neighbor one.

## Baseline evaluation with end-to-end training
Train a classifier on top of the Fawkesv03 feature extractor end-to-end, with one attacking user.

To evaluate Fawkes' attack:
```bash
python3 eval_e2e.py --gpu 0 --attack-dir facescrub_fawkes_attack/download/Adam_Sandler/face --facescrub-dir facescrub/download/ --unprotected-file-match .jpg --protected-file-match high_cloaked.png
```

To evaluate LowKey's attack:
```bash
python3 eval_e2e.py --gpu 0 --attack-dir facescrub_lowkey_attack/download/Adam_Sandler/face --facescrub-dir facescrub/download/ --unprotected-file-match small.png --protected-file-match attacked.png
```

Results:

Fawkes (baseline E2E) | Lowkey (baseline E2E)
----------------------|---------------------
```Protection rate: 0.88```|```Protection rate: 0.97```

## Attack evaluation with end-to-end training

### Adaptive end-to-end
To evaluate robust end-to-end training, we add attacked pictures into the model's training set.
We assume here that at most half of the FaceScrub users have attacked pictures.

To evaluate Fawkes' attack:
```bash
python3 eval_e2e.py --gpu 0 --attack-dir facescrub_fawkes_attack/download/Adam_Sandler/face --facescrub-dir facescrub/download/ --unprotected-file-match .jpg --protected-file-match     high_cloaked.png --robust --public-attack-dirs facescrub_fawkes_attack/download facescrub_lowkey_attack/download
```

To evaluate LowKey's attack:
```bash
python3 eval_e2e.py --gpu 0 --attack-dir facescrub_lowkey_attack/download/Adam_Sandler/face --facescrub-dir facescrub/download/ --unprotected-file-match small.png --protected-file-     match attacked.png --robust --public-attack-dirs facescrub_fawkes_attack/download facescrub_lowkey_attack/download
```

Results:

Fawkes (adaptive E2E) | Lowkey (adaptive E2E)
----------------------|---------------------
```Protection rate: 0.03```|```Protection rate: 0.03```
