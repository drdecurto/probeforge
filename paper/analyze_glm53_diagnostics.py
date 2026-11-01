"""Check GLM-5.3 completion-budget and exact-answer diagnostics from archived responses."""
from pathlib import Path
import json,hashlib
P=Path(__file__).resolve().parent;E=P/'evidence/round20/evalglm53'
read=lambda p:[json.loads(l) for l in p.read_text().splitlines()]
by={r['id']:r for r in read(P/'evidence/paired/test_eval12.jsonl')};out={}
for name,n,correct,length in [('main4k',1248,521,73),('think26',26,12,13),('think60',52,34,4)]:
 rs=read(E/f'all_{name}.jsonl');assert len(rs)==len({r['id'] for r in rs})==n
 assert all(r['budget']==4096 and not r['synthetic'] for r in rs)
 assert sum(bool(r['verification']['correct']) for r in rs)==correct
 assert sum(r['finish_reason']=='length' for r in rs)==length
 out[name]={'n':n,'correct':correct,'length_stops':length}
 if name=='main4k':
  for r in rs:
   expected=hashlib.sha256(json.dumps(by[r['id']],sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
   assert r['example_hash']==expected
  failures=[r['verification'] for r in rs if not r['verification']['correct']]
  answer_only=sum(v.get('status_correct',False) and v.get('facts_correct',False) and not v.get('answer_correct',True) for v in failures)
  assert len(failures)==727 and answer_only==424
  out[name].update(failures=len(failures),incorrect_answer_with_correct_status_and_facts=answer_only)
(P/'VALIDATION_GLM53.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
