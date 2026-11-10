"""Summarise completed human listening ratings."""
import argparse,csv
from pathlib import Path
def main():
 p=argparse.ArgumentParser();p.add_argument("--csv",type=Path,required=True);p.add_argument("--out",type=Path,default=Path("logs/report_results/listening_table.md"));a=p.parse_args();rows=list(csv.DictReader(a.csv.open()))
 if len(rows)!=14 or any(not r["quality_1to5"] for r in rows): raise ValueError("complete all 14 ratings")
 a.out.write_text("*Single unblinded listener; means require completed ratings.*\n")
if __name__=="__main__":main()