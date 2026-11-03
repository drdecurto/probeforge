import json,glob,os,csv,statistics as st
D="/gpfs/scratch/bsc21/bsc021885/cf/stdbench"
BL=["eng_Latn","spa_Latn","cat_Latn","arb_Arab","eus_Latn","fra_Latn","por_Latn","ita_Latn","deu_Latn","rus_Cyrl","zho_Hans","zho_Hant"]
GM=["ar","de","en","es","fr","it","pt","zh"]; MG=["de","en","es","fr","ru","zh"]
rows=[]
for name in [l.split("|")[0] for l in open(__import__("os").environ.get("MODELS",f"{D}/models.txt")) if l.strip()]:
    r={"model":name}
    for part in ("mc","mgsm"):
        fs=sorted(glob.glob(f"{D}/out/{name}/{part}/**/results_*.json",recursive=True))
        if not fs: continue
        res=json.load(open(fs[-1]))["results"]
        if part=="mc":
            for l in BL:
                x=res.get(f"belebele_{l}"); 
                if x: r[f"bele_{l}"]=x.get("acc,none")
            for l in GM:
                x=res.get(f"global_mmlu_{l}")
                if x: r[f"gmmlu_{l}"]=x.get("acc,none")
        else:
            for l in MG:
                x=res.get(f"mgsm_native_cot_{l}")
                if x: r[f"mgsm_{l}"]=x.get("exact_match,flexible-extract"); r[f"mgsmstrict_{l}"]=x.get("exact_match,strict-match")
    for pre,ls in (("bele",BL),("gmmlu",GM),("mgsm",MG),("mgsmstrict",MG)):
        v=[r[f"{pre}_{l}"] for l in ls if r.get(f"{pre}_{l}") is not None]
        r[f"{pre}_mean"]=round(st.mean(v),4) if len(v)==len(ls) else None
    rows.append(r)
keys=["model","bele_mean","gmmlu_mean","mgsm_mean","mgsmstrict_mean"]+sorted({k for r in rows for k in r}-{"model","bele_mean","gmmlu_mean","mgsm_mean","mgsmstrict_mean"})
with open(__import__("os").environ.get("OUTCSV",f"{D}/stdbench_summary.csv"),"w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows)
print(f"{'model':24s} belebele12 gmmluLite8 mgsm6(flex) mgsm6(strict)")
for r in rows: print(f"{r['model']:24s} {r['bele_mean']}  {r['gmmlu_mean']}  {r['mgsm_mean']}  {r['mgsmstrict_mean']}")
