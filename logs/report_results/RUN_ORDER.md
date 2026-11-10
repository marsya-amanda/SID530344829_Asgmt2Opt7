# Colab run order

```bash
python scripts/report_generate_set.py --weights weights/core.pt --guidance 1 --out-dir logs/report_sets --device cuda
python scripts/report_generate_set.py --weights weights/core.pt --guidance 3 --out-dir logs/report_sets --device cuda
python scripts/report_format_check.py --wav-dir logs/report_sets --out-dir logs/report_results
python scripts/report_similarity.py --generated logs/report_sets/w_3 --pool all --out-dir logs/report_results
python scripts/report_diversity.py --generated logs/report_sets --out-dir logs/report_results
python scripts/report_timing.py --weights weights/core.pt --device cuda --out-dir logs/report_results
python scripts/report_figures.py pipeline --out report/figs/fig_pipeline.png
python scripts/report_listening_table.py --csv logs/report_results/listening_notes.csv
```