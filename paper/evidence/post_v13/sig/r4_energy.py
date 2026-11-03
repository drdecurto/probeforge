import json,os,random,statistics as st
S="/gpfs/scratch/bsc21/bsc021885/cf"; src=open(f"{S}/sig/sig.py").read()
g={}; exec(src.split("# ---------- energy benchmark")[1].split("def boot")[0].replace("",""), {**g,"S":S,"os":os,"json":json}, g) if False else None
B=f"{S}/energybench"; L=["en","es","ca","ar","lb","eu","fr","pt","it","de","ru","zh-cn","zh-tw"]
ns={}; exec(open(f"{B}/answer_acc.py").read().split("rows = []")[0], ns)
seg=src.split("def items(od, s):")[1].split("\ndef ")[0]; exec("def items(od, s):"+seg, {**globals()}, globals())
def mean(d): return sum(d.values())/len(d)
def boot(a,b,n=2000):
    k=sorted(set(a)&set(b)); P=sorted({i for _,i in k}); by={p:[x for x in k if x[1]==p] for p in P}; r=random.Random(1); ds=[]
    for _ in range(n):
        s=[x for p in (r.choice(P) for _ in P) for x in by[p]]; ds.append(sum(a[x]-b[x] for x in s)/len(s))
    ds.sort(); return sum(a[x]-b[x] for x in k)/len(k), ds[50], ds[1949]
for fam,base,groups in [("Phi-4","phi-4-base",{"mined":["phi-4-mined","phi-4-mined_s1","phi-4-mined_s2"],"v4c":["v4c-phi4_s1","v4c-phi4_s2","v4c-phi4_s3"]}),
                        ("Aya","aya-expanse-8b",{"mined":["aya-mined","aya-mined_s2","aya-mined_s3"],"v4c":["v4c-aya_own_s1","v4c-aya_own_s2","v4c-aya_own_s3"]})]:
  for od in ("out","out_chat"):
    bb=items(od,base)
    if bb is None: print(fam,od,"base missing"); continue
    print(f"{fam} {od} base S={mean(bb[0]):.3f} K={mean(bb[1]):.3f}")
    for arm,slots in groups.items():
      Ss,Ks,sig=[],[],[]
      for s in slots:
        it=items(od,s)
        if it is None: print("  ",s,"missing"); continue
        Ss.append(mean(it[0])); Ks.append(mean(it[1])); d,lo,hi=boot(it[0],bb[0]); dk,lok,hik=boot(it[1],bb[1]); sig.append(f"{s}: dS {d:+.3f}[{lo:+.3f},{hi:+.3f}] dK {dk:+.3f}[{lok:+.3f},{hik:+.3f}]")
      if Ss: print(f"  {arm:6s} n={len(Ss)} S={st.mean(Ss):.3f}±{(st.stdev(Ss) if len(Ss)>1 else 0):.3f} K={st.mean(Ks):.3f}±{(st.stdev(Ks) if len(Ks)>1 else 0):.3f}"); [print("     ",x) for x in sig]
