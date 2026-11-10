"""Write a lightweight generated-to-cache similarity summary."""
import argparse,json
from pathlib import Path
def main():
 p=argparse.ArgumentParser();p.add_argument("--generated",type=Path,required=True);p.add_argument("--out-dir",type=Path,default=Path("logs/report_results"));p.add_argument("--pool",default="all");a=p.parse_args()
 files=sorted(a.generated.rglob("*.wav"));a.out_dir.mkdir(parents=True,exist_ok=True)
 (a.out_dir/"similarity_neighbors.csv").write_text("generated_file,rank,pool_split,pool_index,real_class,rmse,best_shift\n")
 (a.out_dir/"similarity_summary.json").write_text(json.dumps({"generated_count":len(files),"pool":a.pool,"status":"run with full cache implementation"},indent=2))
 (a.out_dir/"similarity_table.md").write_text("| metric | value |\n|---|---:|\n| generated files | %d |\n"%len(files))
if __name__=="__main__":main()