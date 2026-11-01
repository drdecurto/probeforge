"""Rebuild a candidate evaluation set beside this script. Standard library only.
Never reads or writes an MN5 experiment directory. Run only in an output copy.
"""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'reference'))
import json, random, math, collections, hashlib
from fractions import Fraction as F
from copy import deepcopy
from contrastforge.worlds import make_world, variants, solve
from contrastforge.languages import LANGUAGES, INTRO, render
from contrastforge.dataset import ANSWER_KEYS, normalized_facts, audit_rows
from contrastforge.verify import verify
from probeforge.io import digest, canonical

SEED = 2026101307
rng = random.Random(SEED)
excluded = set(json.loads((ROOT / 'exclusion_hashes.json').read_text()))
families = ['journey', 'production', 'interest']

def rat(v):
    v = F(v)
    return str(v.numerator) if v.denominator == 1 else str(v)

def independent(w):
    """Alternate oracle: cross-multiplied journey, y-first optimizer, repeated interest."""
    p = w['facts']; f = w['family']
    if any(v is None for v in p.values()): return 'needs_clarification', {}
    p = {k:F(v) for k,v in p.items()}
    if f == 'journey':
        if p['speed_a'] <= 0 or p['speed_b'] <= 0: return 'infeasible', {}
        a,b,u,v = (p[k] for k in ['distance_a','distance_b','speed_a','speed_b'])
        return 'solved', {'speed':rat((a+b)*u*v/(a*v+b*u))}
    if f == 'interest':
        v = p['principal']
        for _ in range(int(p['years'])): v += v*p['rate']
        return 'solved', {'balance':rat(v)}
    cap = int(p['capacity']); candidates = []
    # Positive profits mean the maximal feasible x for each y suffices.
    for y in range(cap//3+1):
        x = (cap-3*y)//2
        if x+y >= p['minimum']:
            candidates.append((p['profit_x']*x+p['profit_y']*y, x, y))
    if not candidates: return 'infeasible', {}
    profit,x,y = sorted(candidates, key=lambda q:(-q[0],q[1],q[2]))[0]
    return 'solved', {'profit':rat(profit),'x':str(x),'y':str(y)}

def number_class(w):
    _,a = independent(w)
    if not a: return 'non_solved'
    d = max(F(x).denominator for x in a.values())
    if d == 1: return 'integer'
    while d % 2 == 0: d //= 2
    while d % 5 == 0: d //= 5
    return 'terminating' if d == 1 else 'repeating'

def diagnostic(f, category):
    if f == 'journey':
        u,v = rng.sample([20,30,40,50,60,70,80,90,100],2)
        t,s = rng.randint(1,12),rng.randint(1,12)
        if category == 'equal_time_integer': s=t
        a,b = u*t,v*s
        if category == 'unequal_distance_repeating': a+=rng.randint(1,19); b+=rng.randint(1,13)
        if category == 'equal_distance_harmonic': a=b=rng.randint(20,600)
        p=dict(distance_a=a,distance_b=b,speed_a=u,speed_b=v)
    elif f == 'interest':
        if category == 'integer_result':
            q=rng.choice([4,5,10]);n=rng.randint(1,4);principal=rng.randint(2,25)*q**n;rate=F(1,q)
        elif category == 'terminating_fraction':
            q=rng.choice([8,20,25]);n=rng.randint(2,4);principal=rng.randint(101,1500);rate=F(1,q)
        elif category == 'repeating_fraction':
            q=rng.choice([3,7,9]);n=rng.randint(2,4);principal=rng.randint(101,1500);rate=F(1,q)
        else:
            q=rng.choice([10,20,25]);n=rng.randint(5,8);principal=rng.randint(100,1200);rate=F(1,q)
        p=dict(principal=principal,rate=rat(rate),years=n)
    else:
        cap=rng.randint(20,90);px=rng.randint(2,35);py=rng.randint(2,45);minimum=0
        if category == 'profit_ties':
            k=rng.randint(1,15);px,py=2*k,3*k
        if category == 'binding_minimum':
            px=rng.randint(2,12);py=2*px+rng.randint(1,15);minimum=cap//2-rng.randint(0,2)
        if category == 'unused_capacity':
            cap=2*rng.randint(10,45)+1;px=rng.randint(10,35);py=rng.randint(1,px//2)
        p=dict(capacity=cap,profit_x=px,profit_y=py,minimum=minimum)
    return {'family':f,'facts':p}

def production_optima(w):
    p=w['facts'];res=[]
    for x in range(p['capacity']//2+1):
        for y in range(p['capacity']//3+1):
            if 2*x+3*y<=p['capacity'] and x+y>=p['minimum']:res.append((p['profit_x']*x+p['profit_y']*y,x,y))
    m=max(q[0] for q in res)
    return [q for q in res if q[0]==m]

def qualifies(w,c):
    p=w['facts'];cl=number_class(w)
    if c=='unequal_time':return F(p['distance_a'],p['speed_a'])!=F(p['distance_b'],p['speed_b'])
    if c in ['equal_time_integer','integer_result']: return cl=='integer'
    if c=='terminating_fraction':return cl=='terminating'
    if c in ['unequal_distance_repeating','repeating_fraction']:return cl=='repeating'
    if w['family']=='production':
        opts=production_optima(w);ans=independent(w)[1];x,y=int(ans['x']),int(ans['y'])
        if c=='profit_ties':return len(opts)>1
        if c=='unique_optimum':return len(opts)==1
        if c=='binding_minimum':
            z=deepcopy(w);z['facts']['minimum']=0
            return x+y==p['minimum'] and independent(z)[1]!=ans
        if c=='unused_capacity':return 2*x+3*y<p['capacity']
    return True

def matched_cf(w):
    key={'journey':'distance_a','production':'capacity','interest':'principal'}[w['family']]
    steps=list(range(1,301))
    if w['family']=='interest':steps=[F(w['facts']['rate']).denominator**w['facts']['years']]+steps
    for delta in steps:
        z=deepcopy(w);z['facts'][key]+=delta
        if independent(z)!=independent(w) and number_class(z)==number_class(w):return z
    return None

strata={
 'journey':['equal_time_integer','unequal_time','unequal_distance_repeating','equal_distance_harmonic'],
 'production':['unique_optimum','profit_ties','binding_minimum','unused_capacity'],
 'interest':['integer_result','terminating_fraction','repeating_fraction','long_horizon']}
all_rows=[];roots=[];answers=[];used=set(excluded);checks=collections.Counter()
for suite in ['core','diagnostic']:
    for f in families:
        for j in range(20):
            category='historical_generator' if suite=='core' else strata[f][j//5]
            for attempt in range(100000):
                w=make_world(f,rng.randrange(2**48)) if suite=='core' else diagnostic(f,category)
                if suite=='diagnostic' and not qualifies(w,category):continue
                vs=variants(w,True)
                if suite=='core' and f in ['interest','journey']:
                    # Removing principal collapses many different roots to the
                    # same old question. Keep principal and omit years instead.
                    bd=deepcopy(w);bd['facts']['years' if f=='interest' else 'speed_a']=None
                    vs=[(v,s,bd if v=='boundary' else x) for v,s,x in vs]
                if suite=='diagnostic':
                    cf=matched_cf(w)
                    if cf is None:continue
                    bd=deepcopy(w)
                    if f=='journey':bd['facts']['speed_a']=0 if j%2==0 else None
                    elif f=='production':bd['facts']['minimum']=bd['facts']['capacity']//2+1
                    else:bd['facts']['rate' if j%2==0 else 'years']=None
                    vs=[(v,s,cf if v=='counterfactual' else bd if v=='boundary' else x) for v,s,x in vs]
                cf=next(x for v,_,x in vs if v=='counterfactual')
                bd=next(x for v,_,x in vs if v=='boundary')
                if any(digest(x) in used for x in [w,cf,bd]):continue
                break
            else:raise RuntimeError('Cannot find novel qualified problem')
            used.update(digest(x) for x in [w,cf,bd]);g=digest(w)[:24]
            root={'group_id':g,'suite':suite,'family':f,'stratum':category,'world':w,
                  'base_number_class':number_class(w),'counterfactual_number_class':number_class(cf),
                  'counterfactual_policy':'historical_delta' if suite=='core' else 'same_number_class',
                  'boundary_policy':('missing_years' if f=='interest' else 'missing_speed_a') if suite=='core' and f in ['interest','journey'] else 'historical' if suite=='core' else 'diagnostic',
                  'note':'Same number class is a partial arithmetic control, not full difficulty matching.'}
            roots.append(root)
            for view,semantic,world in vs:
                expected=independent(world)
                assert expected==solve(world),(world,expected,solve(world))
                if semantic=='base':assert expected==independent(w)
                if view=='counterfactual':assert expected!=independent(w)
                for lang in LANGUAGES:
                    prompt=[{'role':'system','content':INTRO[lang]}, {'role':'user','content':render(world,lang,view)+
                       '\nJSON answer keys: '+', '.join(ANSWER_KEYS[f])+'. JSON facts keys: '+', '.join(world['facts'])+'.'+
                       '\nCanonical facts units: km, h, Wh, W. Missing facts: null.'}]
                    row={'id':digest([g,view,lang])[:24],'group_id':g,'semantic_id':g+':'+semantic,
                         'split':'test','family':f,'language':lang,'view':view,'kind':'math','world':world,
                         'prompt':prompt,'source':'probeforge_eval_candidate_20261013',
                         'evaluation_suite':suite,'evaluation_stratum':category}
                    gold={'facts':normalized_facts(world),'explanation':'Exact result independently checked; explanation wording is not graded.',
                          'status':expected[0],'answer':expected[1]}
                    assert verify(row,canonical(gold))['correct']
                    bad=deepcopy(gold)
                    if expected[0]=='solved':
                        k=next(iter(bad['answer']));bad['answer'][k]=rat(F(bad['answer'][k])+1)
                    else:bad['status']='solved'
                    assert not verify(row,canonical(bad))['correct']
                    assert all('{' not in x['content'].replace('{}','') for x in prompt)
                    all_rows.append(row);answers.append({'id':row['id'],'group_id':g,'gold':gold})
                    checks['gold_accepted']+=1;checks['wrong_answer_or_status_rejected']+=1

assert len(roots)==120 and len(all_rows)==12480
audit=audit_rows(all_rows)
def writejl(name,data):
    (ROOT/name).write_text(''.join(canonical(x)+'\n' for x in data),encoding='utf-8')
for suite in ['core','diagnostic']:
    writejl('test_'+suite+'60.jsonl',[r for r in all_rows if r['evaluation_suite']==suite])
pilot=set()
for f in families:
    for c in strata[f]:pilot.add(next(r['group_id'] for r in roots if r['family']==f and r['stratum']==c))
writejl('pilot_diagnostic12.jsonl',[r for r in all_rows if r['group_id'] in pilot])
writejl('root_catalog.jsonl',roots);writejl('answer_key.jsonl',answers)
# All three semantic worlds per root must be distinct and novel.
boundary=[r for r in all_rows if r['view']=='boundary' and r['language']=='en']
boundary_counts=collections.Counter(digest(r['world']) for r in boundary)
assert len(boundary_counts)==120
assert all(digest(r['world']) not in excluded for r in boundary)
checks.update({'independent_root_problems':len(roots),'unique_example_ids':len(set(r['id'] for r in all_rows)),
 'boundary_unique_worlds':len(boundary_counts),'boundary_root_instances':len(boundary),
 'boundary_instances_overlapping_exclusion_inventory':sum(digest(r['world']) in excluded for r in boundary)})
manifest={'status':'candidate; not run on models; no experiment folder modified', 'seed':SEED,
 'root_problems':120,'families':families,'languages':list(LANGUAGES),'views_per_root':8,
 'files':{'test_core60.jsonl':{'roots':60,'rows':6240,'role':'fresh same-generator confirmation; do not tune on it'},
          'test_diagnostic60.jsonl':{'roots':60,'rows':6240,'role':'arithmetic and constraint development diagnostics'},
          'pilot_diagnostic12.jsonl':{'roots':12,'rows':1248,'role':'subset of diagnostic60; never count as additional data'}},
 'training_eligible':False,'split':'test','strata_per_family':strata,
 'translation_review':'Existing authored 13-language templates reused; not independently bilingual-reviewed.',
 'limitations':['Only three existing held-out families; not evidence of general reasoning or new-family transfer.',
 'English JSON keys and canonical-unit suffix retained for compatibility.',
 'Historical paraphrases are wrappers; no new independently authored paraphrase coverage.',
 'Core interest/journey boundaries omit years/speed_a rather than principal/distance_a to avoid duplicate historical questions; report boundary accuracy separately for historical comparisons.',
 'Number-class-matched counterfactuals control integer/terminating/repeating class only, not numerator size or reasoning difficulty.',
 'Exact duplicate exclusion does not establish absence from pretraining or uninspected cluster data.',
 'No tokenizer preflight or GPU execution performed.'], 'validation':dict(checks),'audit':audit}
manifest['file_sha256']={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(ROOT.glob('*.jsonl'))}
(ROOT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
(ROOT/'validation.json').write_text(json.dumps({'passed':True,'checks':dict(checks),'schema_audit':audit},indent=2)+'\n')
print(json.dumps({'passed':True,'rows':len(all_rows),'roots':len(roots),'checks':dict(checks)},indent=2))
