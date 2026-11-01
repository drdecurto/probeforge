"""Fetch pinned public source snapshots for diagnostics; no downloaded code is executed."""
import argparse,subprocess,json
from pathlib import Path

REPOS={
 "crossplatform-llm":("https://github.com/pablogarciaamolina/crossplatform-llm","a88246f3ead49885fcdb5b90ae589e1b3dace90d"),
 "comparison_llm_decurto":("https://github.com/drdecurto/comparison_llm_decurto","181fd20294d0d86c14014693e53b89b8ad5d5b84"),
 "energy_multilingual_llm_evaluation":("https://github.com/drdezarza/energy_multilingual_llm_evaluation","971e09e6eec48e1b25697f5977132c7bac90d5a7"),
 "vlm_energy_multilingual":("https://github.com/drdezarza/vlm_energy_multilingual","04d4ab0e9d42855f280e0cb77ae49399061eba36"),
 "lic":("https://github.com/drdezarza/lic","59ba06d7fc051acf2aaed8e263e61c1d50ba3440")}

def main():
 p=argparse.ArgumentParser();p.add_argument("--out",default="sources");a=p.parse_args();root=Path(a.out);root.mkdir(parents=True,exist_ok=True)
 for name,(url,sha) in REPOS.items():
  dest=root/name
  if not dest.exists():
   subprocess.run(["git","init",str(dest)],check=True,stdout=subprocess.DEVNULL)
   subprocess.run(["git","-C",str(dest),"remote","add","origin",url],check=True)
   subprocess.run(["git","-C",str(dest),"fetch","--depth","1","origin",sha],check=True)
   subprocess.run(["git","-C",str(dest),"checkout","--detach","FETCH_HEAD"],check=True,stdout=subprocess.DEVNULL)
  actual=subprocess.check_output(["git","-C",str(dest),"rev-parse","HEAD"],text=True).strip()
  if actual!=sha:raise ValueError(f"Existing checkout has a different revision: {dest}")
 (root/"source_manifest.json").write_text(json.dumps(REPOS,indent=2)+"\n")
 print("Pinned sources ready. Import response records with contrastforge import-legacy; they remain diagnostics.")

if __name__=="__main__":main()
