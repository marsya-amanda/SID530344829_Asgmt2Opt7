"""Generate deterministic class/seed WAV sets through ``foley.generate``."""
import argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from foley.generate import load_ckpt,generate
from foley.audio import save_wav
def main():
 p=argparse.ArgumentParser();p.add_argument("--weights",type=Path,required=True);p.add_argument("--guidance",type=float,required=True);p.add_argument("--steps",type=int);p.add_argument("--classes",nargs="*",type=int,default=list(range(7)));p.add_argument("--n-seeds",type=int,default=20);p.add_argument("--batch",type=int,default=1);p.add_argument("--out-dir",type=Path,required=True);p.add_argument("--device",default="auto");a=p.parse_args()
 import torch;dev=("cuda" if torch.cuda.is_available() else "cpu") if a.device=="auto" else a.device;m,d,ck=load_ckpt(a.weights,dev);out=a.out_dir/f"w_{a.guidance:g}";out.mkdir(parents=True,exist_ok=True)
 for c in a.classes:
  for s in range(a.n_seeds):
   w,_=generate(m,d,ck,[c],[s],a.guidance,a.steps or ck["gen_defaults"]["steps"],device=dev);save_wav(out/f"class_{c}_seed_{s}.wav",w[0])
if __name__=="__main__":main()