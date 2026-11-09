# COMP3419 Option 7 Foley generation

This repository implements class-conditional diffusion on 80-band log-mel
spectrograms followed by seeded Griffin-Lim reconstruction. Class folders map
to IDs as follows: `dog_bark=0`, `footstep=1`, `gun_shot=2`, `keyboard=3`,
`moving_motor_vehicle=4`, `rain=5`, and `sneeze_cough=6`.

## Environment

Use Python 3.10+ and install only the packages in `requirements.txt`:

```bash
python -m pip install -r requirements.txt
```

The code runs on CPU for smoke tests. For the full run, use a CUDA T4 with
fp16 autocast (not bf16) and store checkpoints on a Drive-mounted path.

## File map

`src/foley/` contains audio, descriptors, data, model, diffusion, EMA,
generation, classifier, and metrics modules. `scripts/` contains preprocessing,
training, evaluation, checkpoint selection, and demo commands. `infer.py` is
the required four-argument inference interface.

## Commands

From the repository directory:

```bash
python scripts/make_splits.py --root /path/to/DCASE_2023_Challenge_Task_7_Dataset --out splits
python scripts/precompute_features.py --root /path/to/DCASE_2023_Challenge_Task_7_Dataset --splits splits --cache cache
python scripts/train_classifier.py --cache cache --out weights/classifier.pt --device cuda
python scripts/train.py --config config/core.yaml --out ckpts --device auto --resume none
python scripts/select_checkpoint.py --ckpts ckpts --out weights/core.pt
python scripts/evaluate.py --weights weights/core.pt --classifier weights/classifier.pt --norm cache/norm.json
python scripts/make_demo.py --weights weights/core.pt --out demo_wavs
python infer.py --weights weights/core.pt --class-id 1 --seed 0 --out out.wav
```

For a CPU smoke run, replace the cache path in `config/smoke.yaml` with a
synthetic cache and run:

```bash
python scripts/train.py --config config/smoke.yaml --out ckpts --device cpu --max-steps 20
python scripts/train.py --config config/smoke.yaml --out ckpts --device cpu --max-steps 40 --resume auto
```

The optional control fine-tune uses `config/ctrl.yaml`, initialized from
`weights/core_raw.pt`, and the bonus demo is:

```bash
python scripts/make_bonus_demo.py --weights weights/ctrl.pt --class-id 1 --control density --values 2 6 9
python infer.py --weights weights/ctrl.pt --class-id 1 --seed 7 --density 6 --out bonus.wav
```

The evaluator must never use `eval/` for training, checkpoint selection, or
guidance selection. It is read only by `precompute_features.py` and the final
real-data reference stage of `evaluate.py`.
