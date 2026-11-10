"""Create the diversity result placeholders for a generated set."""
import argparse,json
from pathlib import Path
def main():
 p=argparse.ArgumentParser();p.add_argument("--generated",type=Path,required=True);p.add_argument("--out-dir",type=Path,default=Path("logs/report_results"));a=p.parse_args();a.out_dir.mkdir(parents=True,exist_ok=True)
 (a.out_dir/"diversity.json").write_text(json.dumps({"status":"run on Colab generated sets","generated_dir":str(a.generated)},indent=2));(a.out_dir/"diversity_table.md").write_text("| setting | class | generated/real |\n|---|---:|---:|\n")
if __name__=="__main__":main()