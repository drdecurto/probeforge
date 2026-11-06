"""Check archived original ProbeForge splits; not a pretraining contamination audit."""
from pathlib import Path
import json
# The audited files are the repository's data/contrast13m_200 splits (identical sha256; see evidence/split_audit/SPLIT_AUDIT_V14_2.json).
P=Path(__file__).resolve().parent.parent/'data/contrast13m_200'
rows={n:[json.loads(l) for l in (P/(n+'.jsonl')).read_text().splitlines()] for n in ['train','validation','test_eval12']}
for a,b in [('train','validation'),('train','test_eval12'),('validation','test_eval12')]:
 for k in ['id','group_id','semantic_id','prompt']:
  values=lambda n:{json.dumps(r[k],sort_keys=True,ensure_ascii=False) for r in rows[n]}
  overlap=values(a)&values(b)
  assert not overlap,(a,b,k,len(overlap))
 print(a,b,'zero overlap in IDs, roots, semantic IDs, and exact prompts')
