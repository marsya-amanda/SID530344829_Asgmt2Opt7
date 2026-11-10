"""Create the report pipeline diagram."""
import argparse
from pathlib import Path
import matplotlib;matplotlib.use("Agg");import matplotlib.pyplot as plt
def main():
 p=argparse.ArgumentParser();p.add_argument("command",choices=["pipeline"]);p.add_argument("--out",type=Path,default=Path("report/figs/fig_pipeline.png"));a=p.parse_args()
 a.out.parent.mkdir(parents=True,exist_ok=True);fig,ax=plt.subplots(figsize=(8,2));ax.axis("off");labels=["dev WAV","log-mel\n80x345","normalise\n+ pad 352","FiLM U-Net\nε prediction","DDIM 50\n+ CFG","Griffin-Lim","4 s WAV"]
 for i,label in enumerate(labels):
  x=i/(len(labels)-1);ax.text(x,.5,label,ha="center",va="center",bbox={"boxstyle":"round","fc":"#d9edf7"})
  if i<len(labels)-1: ax.annotate("",(x+.11,.5),(x+.035,.5),arrowprops={"arrowstyle":"->"})
 fig.savefig(a.out,dpi=300,bbox_inches="tight");plt.close(fig)
if __name__=="__main__":main()