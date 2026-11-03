import sys
import os,json
S="/gpfs/scratch/bsc21/bsc021885/cf"; src=open(f"{S}/sig/sig.py").read()
B=f"{S}/energybench"; L=["en","es","ca","ar","lb","eu","fr","pt","it","de","ru","zh-cn","zh-tw"]
ns={}; exec(open(f"{B}/answer_acc.py").read().split("rows = []")[0], ns)
B=f"{S}/kolibri/energy"
seg=src.split("def items(od, s):")[1].split("\ndef ")[0]; exec("def items(od, s):"+seg, globals())
for od in ("out","out_chat"):
    it=items(od,sys.argv[1]); print(od, "missing" if it is None else "S=%.3f K=%.3f"%(sum(it[0].values())/len(it[0]), sum(it[1].values())/len(it[1])))
