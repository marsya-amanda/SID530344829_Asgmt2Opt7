## 1. Files created/changed

- Implemented audio validation, normalisation, Griffin-Lim, and WAV output in `audio.py`.
- Added robust class mapping and validation in `constants.py`.
- Implemented split listing, stratified splitting, cache dataset loading, and control handling in `data.py` and `src/data.py`.
- Fixed U-Net skip validation, GroupNorm constraints, control conditioning, and CFG batching in `model.py` and `diffusion.py`.
- Implemented metrics and classifier helpers in `metrics.py` and compatibility modules under `src/foley/`.
- Implemented CPU/CUDA training, fp16 CUDA support, resume, EMA checkpoints, `last.pt`, and `--max-steps` in `train.py`.
- Added `smoke.yaml`.
- Implemented preprocessing, classifier training, checkpoint selection, generation, evaluation, nearest-neighbour, and demo scripts.
- Made `infer.py` location-independent.
- Added the audit and deviation record in `logs/deviations.md`.
- Replaced the empty scaffold checkpoint with a tiny deterministic smoke checkpoint in `weights/core.pt`.
- Updated `README.md`.

## 2. Test results

- `pytest -q`: **8 passed**, 90.58 seconds.
- `python -m compileall src scripts infer.py`: passed.
- All script `--help` commands: passed.
- Default U-Net parameter count: **16,431,937**, below the 30M limit.
- Tiny U-Net and default U-Net output shape: `(B, 1, 80, 352)`.
- CPU training smoke test and resume test: passed.
- Location-independent inference from `/tmp`: passed.
- Generated WAV validation: 88,200 samples, 22,050 Hz, mono, PCM16, finite, non-silent.
- Same seed reproduced identical audio; different seeds produced different audio.

## 3. Deviations and bugs fixed

Recorded in `logs/deviations.md`, including:

- Incorrect unconditional CFG control duplication.
- CPU-incompatible validation generator.
- Missing U-Net skip-shape validation.
- CWD-dependent `sys.path` handling.
- Missing training resume and atomic `last.pt`.
- Missing split/cache/preprocessing implementations.
- Missing metrics and generation implementations.
- Scaffold path mismatch between top-level modules and the planned `src/foley/` layout.

## 4. Not verified here

These require real data or a Colab T4:

1. Full preprocessing:
   ```bash
   python scripts/make_splits.py --root /path/to/DCASE_2023_Challenge_Task_7_Dataset --out splits
   python scripts/precompute_features.py --root /path/to/DCASE_2023_Challenge_Task_7_Dataset --splits splits --cache cache
   ```
   Verify cache shapes, folder-name mapping, train-only normalisation, and descriptor quantiles.

2. Full CUDA training:
   ```bash
   python scripts/train.py --config config/core.yaml --out ckpts --device cuda --resume none
   ```
   Verify fp16 stability, throughput, loss decrease, periodic checkpoints, and Drive persistence.

3. Resume after interruption:
   ```bash
   python scripts/train.py --config config/core.yaml --out ckpts --device cuda --resume auto
   ```
   Verify the step counter and LR schedule continue correctly.

4. Classifier quality and checkpoint selection on real validation audio.

5. Final evaluation protocol and generated audio quality.

6. Bonus control fine-tuning and descriptor correlation.

## 5. Exact Colab commands to run next

```bash
cd /content/SID530344829_Asgmt2Opt7
pip install -r requirements.txt

python scripts/make_splits.py \
  --root /content/DCASE_2023_Challenge_Task_7_Dataset \
  --out splits

python scripts/precompute_features.py \
  --root /content/DCASE_2023_Challenge_Task_7_Dataset \
  --splits splits \
  --cache cache

python scripts/train_classifier.py \
  --cache cache \
  --out weights/classifier.pt \
  --device cuda

python scripts/train.py \
  --config config/core.yaml \
  --out /content/drive/MyDrive/foley_ckpts \
  --device cuda \
  --resume none

python scripts/select_checkpoint.py \
  --ckpts /content/drive/MyDrive/foley_ckpts \
  --out weights/core.pt

python scripts/evaluate.py \
  --weights weights/core.pt \
  --classifier weights/classifier.pt \
  --norm cache/norm.json \
  --device cuda

python scripts/make_demo.py \
  --weights weights/core.pt \
  --out demo_wavs \
  --device cuda

python infer.py \
  --weights weights/core.pt \
  --class-id 1 \
  --seed 0 \
  --out demo.wav \
  --device cuda
```

## 6. Ranked risks and weak spots

1. **High:** `weights/core.pt` is a random tiny smoke checkpoint, not a trained model. It only validates the inference contract.
2. **High:** Full 50k-step CUDA training and final audio quality were not verified.
3. **Medium:** The evaluation scripts are functional but more compact than the complete PLAN.md sweep/figure protocol; the final real-data evaluation should be checked and expanded if required by the report.
4. **Medium:** Real dataset folder variants and cache size were not validated against the complete archive.
5. **Low:** Pylance reports unresolved imports for scripts because they add `src/` dynamically at runtime; execution and compilation succeed.