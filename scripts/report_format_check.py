"""Check WAV compliance and write JSON/Markdown results."""
import argparse,json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from foley.metrics import check_wav_format
def main():
 p=argparse.ArgumentParser();p.add_argument("--wav-dir",type=Path,required=True);p.add_argument("--out-dir",type=Path,default=ROOT/"logs/report_results");a=p.parse_args()
 rows=[] 
 for f in sorted(a.wav_dir.rglob("*.wav")):
  d=check_wav_format(f);rows.append({"file":str(f),"pass":all(d.values()),**d})
 a.out_dir.mkdir(parents=True,exist_ok=True);(a.out_dir/"format_summary.json").write_text(json.dumps({"total":len(rows),"passed":sum(x["pass"] for x in rows),"files":rows},indent=2));(a.out_dir/"format_table.md").write_text("| file | pass |\n|---|---|\n"+"\n".join(f"|{x['file']}|{x['pass']}|" for x in rows))
if __name__=="__main__":main()