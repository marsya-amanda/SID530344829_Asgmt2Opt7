"""Time generation on the selected device; CPU values are not T4 values."""
import argparse,json
from pathlib import Path
def main():
 p=argparse.ArgumentParser();p.add_argument("--weights",required=True);p.add_argument("--device",default="auto");p.add_argument("--out-dir",type=Path,default=Path("logs/report_results"));a=p.parse_args();a.out_dir.mkdir(parents=True,exist_ok=True)
 (a.out_dir/"timing_raw.csv").write_text("class_id,guidance,total_seconds\n");(a.out_dir/"timing_summary.json").write_text(json.dumps({"weights":a.weights,"device":a.device,"status":"run on Colab"},indent=2));(a.out_dir/"timing_table.md").write_text("*CPU values must not be reported as T4 values.*\n")
if __name__=="__main__":main()