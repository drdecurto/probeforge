"""Deterministic, standard-library-only candidate benchmark builder.
Writes only beside this script; never reads/writes an MN5 experiment folder.
"""
import sys
sys.dont_write_bytecode=True
import json, hashlib, random, math, collections
from fractions import Fraction as F
from copy import deepcopy
from pathlib import Path
ROOT=Path(__file__).resolve().parent
SEED=2026101319
R=random.Random(SEED)
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False)
def digest(x):return hashlib.sha256(canonical(x).encode()).hexdigest()
def rat(x):
    x=F(x);return str(x.numerator) if x.denominator==1 else str(x)
def jl(name,rs): (ROOT/name).write_text(''.join(canonical(r)+'\n' for r in rs),encoding='utf-8')

FAMILIES=['reverse_percentage','conditional_probability','modular_schedule',
          'inventory_balance','dependency_schedule','unit_conversion','linear_optimization']
KEY={'reverse_percentage':'original_price','conditional_probability':'probability',
     'modular_schedule':'first_time','inventory_balance':'closing_stock',
     'dependency_schedule':'finish_time','unit_conversion':'converted_mass',
     'linear_optimization':'maximum'}
REQ={'reverse_percentage':['sale','discount'], 'conditional_probability':['red_small','blue_small'],
     'modular_schedule':['period_a','period_b','offset_a','offset_b'],
     'inventory_balance':['opening','incoming','outgoing'],
     'dependency_schedule':['duration_a','duration_b','duration_c'],
     'unit_conversion':['quantity','source_unit','target_unit'],
     'linear_optimization':['coefficient','lower','upper']}
OPTIONAL={'conditional_probability':'red_large'}
SCALE={'kg':F(1000),'g':F(1),'mg':F(1,1000)}

def solve(f,p):
    need=REQ[f]
    if f=='linear_optimization' and p.get('no_upper'):need=['coefficient','lower']
    if any(p[k] is None for k in need):return 'needs_clarification',{}
    if f=='reverse_percentage':
        s,d=F(p['sale']),F(p['discount'])
        if s<0 or not 0<=d<100:return 'infeasible',{}
        v=s/(1-d/100)
    elif f=='conditional_probability':
        a,b=p['red_small'],p['blue_small']
        if a<0 or b<0:return 'infeasible',{}
        if a+b==0:return 'needs_clarification',{}
        v=F(a,a+b)
    elif f=='modular_schedule':
        m,n,a,b=(p[k] for k in REQ[f]);g=math.gcd(m,n)
        if (b-a)%g:return 'infeasible',{}
        v=(a+m*((b-a)//g*pow(m//g,-1,n//g)%(n//g)))%math.lcm(m,n)
    elif f=='inventory_balance':
        v=p['opening']+p['incoming']-p['outgoing']
        if min(p['opening'],p['incoming'],p['outgoing'],v)<0:return 'infeasible',{}
    elif f=='dependency_schedule':
        if p['cyclic']:return 'infeasible',{}
        v=max(p['duration_a'],p['duration_b'])+p['duration_c']
    elif f=='unit_conversion':
        if F(p['quantity'])<0:return 'infeasible',{}
        v=F(p['quantity'])*SCALE[p['source_unit']]/SCALE[p['target_unit']]
    else:
        if p.get('no_upper'):return 'unbounded',{}
        if p['lower']>p['upper']:return 'infeasible',{}
        v=p['coefficient']*p['upper']
    return 'solved',{KEY[f]:rat(v)}

def second_oracle(f,p):
    """Independently compute the same answers; no call to solve()."""
    req=REQ[f]
    if f=='linear_optimization' and p['no_upper']:req=['coefficient','lower']
    if any(p[k] is None for k in req):return 'needs_clarification',{}
    if f=='reverse_percentage':
        if F(p['sale'])<0 or F(p['discount'])<0 or F(p['discount'])>=100:return 'infeasible',{}
        # Solve 100*s = original*(100-discount).
        v=100*F(p['sale'])/(100-F(p['discount']))
    elif f=='conditional_probability':
        a,b=p['red_small'],p['blue_small']
        if min(a,b)<0:return 'infeasible',{}
        if a+b==0:return 'needs_clarification',{}
        # Explicit equally likely outcomes in the conditioning event.
        outcomes=[True]*a+[False]*b
        v=F(sum(outcomes),len(outcomes))
    elif f=='modular_schedule':
        hits=[t for t in range(math.lcm(p['period_a'],p['period_b']))
              if t%p['period_a']==p['offset_a'] and t%p['period_b']==p['offset_b']]
        if not hits:return 'infeasible',{}
        v=min(hits)
    elif f=='inventory_balance':
        if min(p['opening'],p['incoming'],p['outgoing'])<0:return 'infeasible',{}
        stock=p['opening']
        for _ in range(p['incoming']):stock+=1
        for _ in range(p['outgoing']):stock-=1
        if stock<0:return 'infeasible',{}
        v=stock
    elif f=='dependency_schedule':
        durations={k:p['duration_'+k.lower()] for k in ['A','B','C']}
        edges={'A':['C'] if p['cyclic'] else [],'B':[],'C':['A','B']}
        finish={}
        while len(finish)<3:
            ready=[k for k in edges if k not in finish and all(q in finish for q in edges[k])]
            if not ready:return 'infeasible',{}
            for k in ready:finish[k]=max([finish[q] for q in edges[k]] or [0])+durations[k]
        v=finish['C']
    elif f=='unit_conversion':
        if F(p['quantity'])<0:return 'infeasible',{}
        # Explicit pairwise conversion table, separate from the gram scale map.
        pairs={('kg','g'):F(1000),('kg','mg'):F(1000000),('g','kg'):F(1,1000),
               ('g','mg'):F(1000),('mg','kg'):F(1,1000000),('mg','g'):F(1,1000)}
        v=F(p['quantity'])*pairs[p['source_unit'],p['target_unit']]
    else:
        if p['no_upper']:return 'unbounded',{}
        if p['lower']>p['upper']:return 'infeasible',{}
        # For a positive linear objective on a closed interval, compare endpoints.
        v=max(p['coefficient']*p['lower'],p['coefficient']*p['upper'])
    return 'solved',{KEY[f]:rat(v)}

def make(f,level):
    size=[20,100,1000][level]
    p={'archive_ref':R.randint(10000,99999)}
    if f=='reverse_percentage':p.update(sale=R.randint(1,size)*R.choice([1,2,5]),discount=R.randint(1,90))
    elif f=='conditional_probability':p.update(red_small=R.randint(1,size),blue_small=R.randint(1,size),red_large=R.randint(1,size),blue_large=R.randint(1,size))
    elif f=='modular_schedule':
        a,b=R.sample([3,5,7,11,13,17,19,23],2);m,n=2*a,2*b;t=R.randint(0,math.lcm(m,n)-1)
        p.update(period_a=m,period_b=n,offset_a=t%m,offset_b=t%n)
    elif f=='inventory_balance':
        a,b=R.randint(1,size),R.randint(1,size);p.update(opening=a,incoming=b,outgoing=R.randint(0,a+b))
    elif f=='dependency_schedule':p.update(duration_a=R.randint(1,size),duration_b=R.randint(1,size),duration_c=R.randint(1,size),cyclic=False)
    elif f=='unit_conversion':
        a,b=R.sample(['kg','g','mg'],2);p.update(quantity=rat(F(R.randint(1,size),R.choice([1,2,4,5,8]))),source_unit=a,target_unit=b)
    else:
        lo=R.randint(0,size);p.update(coefficient=R.randint(1,30),lower=lo,upper=lo+R.randint(1,size),no_upper=False)
    return p

def variant_facts(f,p):
    cf=deepcopy(p)
    k={'reverse_percentage':'sale','conditional_probability':'red_small','modular_schedule':'offset_a',
       'inventory_balance':'incoming','dependency_schedule':'duration_c','unit_conversion':'quantity','linear_optimization':'upper'}[f]
    cf[k]=rat(F(cf[k])+1) if f=='unit_conversion' else (cf[k]+2)%cf['period_a'] if f=='modular_schedule' else cf[k]+1
    missing=deepcopy(p)
    missing[{'unit_conversion':'source_unit','modular_schedule':'offset_a'}.get(f,REQ[f][0])]=None
    optional=deepcopy(p);optional[OPTIONAL.get(f,'archive_ref')]=None
    invalid=deepcopy(p)
    if f=='reverse_percentage':invalid['discount']=100
    elif f=='conditional_probability':invalid['red_small']=-1
    elif f=='modular_schedule':invalid['offset_b']=(invalid['offset_b']+1)%invalid['period_b']
    elif f=='inventory_balance':invalid['outgoing']=p['opening']+p['incoming']+1
    elif f=='dependency_schedule':invalid['cyclic']=True
    elif f=='unit_conversion':invalid['quantity']=rat(-F(p['quantity']))
    else:invalid['lower']=p['upper']+1
    out=[('base','base',p),('paraphrase','base',p),('fact_reorder','base',p),('distractor','base',p),
         ('counterfactual','changed',cf),('missing_required','missing_required',missing),
         ('missing_optional','missing_optional',optional),('infeasible','infeasible',invalid)]
    if f=='linear_optimization':
        z=deepcopy(p);z['no_upper']=True;z['upper']=None;out.append(('unbounded','unbounded',z))
    return out

def wording(f,p,lang,view):
    es=lang=='es';para=view=='paraphrase';v=lambda k:'?' if p.get(k) is None else str(p[k])
    if f=='reverse_percentage':
        normal=(f'Tras un descuento del {v("discount")}% sobre el precio original, el precio de venta es {v("sale")}. Calcula el precio original.' if es else f'After a {v("discount")}% discount on the original price, the sale price is {v("sale")}. Find the original price.')
        alt=(f'El comprador paga {v("sale")} después de que el vendedor reduzca el precio inicial en un {v("discount")}%. ¿Cuál era el precio antes de la rebaja?' if es else f'A buyer pays {v("sale")} after the seller reduces the starting price by {v("discount")}%. What was the price before the reduction?')
        facts=[('Precio de venta' if es else 'Sale price',v('sale')),('Descuento porcentual' if es else 'Discount percent',v('discount'))]
        question='Recupera el precio original.' if es else 'Recover the original price.'
        rule='Los precios no pueden ser negativos. Los datos deben admitir un precio original finito; no hay impuestos ni redondeo.' if es else 'Prices must be nonnegative. The data must admit a finite original price; there are no taxes or rounding.'
    elif f=='conditional_probability':
        normal=(f'Una urna contiene {v("red_small")} bolas rojas pequeñas, {v("blue_small")} azules pequeñas, {v("red_large")} rojas grandes y {v("blue_large")} azules grandes. Se extrae una bola al azar, con igual probabilidad para cada bola. Sabiendo que es pequeña, calcula la probabilidad de que sea roja.' if es else f'An urn contains {v("red_small")} small red balls, {v("blue_small")} small blue balls, {v("red_large")} large red balls, and {v("blue_large")} large blue balls. Draw one ball uniformly at random. Given that it is small, find the probability it is red.')
        alt=(f'Solo se consideran las bolas pequeñas: hay {v("red_small")} rojas y {v("blue_small")} azules. Las grandes son {v("red_large")} rojas y {v("blue_large")} azules. Si una bola elegida uniformemente resulta ser pequeña, ¿qué probabilidad tiene de ser roja?' if es else f'The small balls consist of {v("red_small")} red and {v("blue_small")} blue balls; the large balls consist of {v("red_large")} red and {v("blue_large")} blue balls. Condition a uniform draw on a small ball being selected. What is its probability of being red?')
        facts=list(zip(['Rojas pequeñas','Azules pequeñas','Rojas grandes','Azules grandes'] if es else ['Small red','Small blue','Large red','Large blue'],[v(k) for k in ['red_small','blue_small','red_large','blue_large']]))
        question='En una extracción uniforme, calcula P(roja | pequeña).' if es else 'For a uniform draw, find P(red | small).'
        rule='Las cantidades de bolas deben ser enteros no negativos.' if es else 'Ball counts must be nonnegative integers.'
    elif f=='modular_schedule':
        normal=(f'La señal A ocurre en los minutos enteros no negativos t con t mod {v("period_a")} = {v("offset_a")}. La señal B ocurre cuando t mod {v("period_b")} = {v("offset_b")}. Encuentra el primer minuto en que coinciden.' if es else f'Signal A occurs at nonnegative integer minutes t satisfying t mod {v("period_a")} = {v("offset_a")}. Signal B occurs when t mod {v("period_b")} = {v("offset_b")}. Find their first simultaneous minute.')
        alt=(f'Busca el menor entero t ≥ 0 cuyo resto al dividir entre {v("period_a")} sea {v("offset_a")} y cuyo resto al dividir entre {v("period_b")} sea {v("offset_b")}.' if es else f'Find the least integer t ≥ 0 that leaves remainder {v("offset_a")} on division by {v("period_a")} and remainder {v("offset_b")} on division by {v("period_b")}.')
        facts=list(zip(['Periodo A','Periodo B','Resto A','Resto B'] if es else ['Period A','Period B','Remainder A','Remainder B'],[v(k) for k in REQ[f]]))
        question='Calcula el menor minuto entero no negativo que cumple ambas congruencias.' if es else 'Find the least nonnegative integer minute satisfying both congruences.'
        rule='Si las congruencias son incompatibles, no existe una coincidencia factible.' if es else 'Incompatible congruences mean there is no feasible simultaneous time.'
    elif f=='inventory_balance':
        normal=(f'Un almacén comienza con {v("opening")} unidades, recibe {v("incoming")} y después debe enviar {v("outgoing")}. ¿Cuántas unidades quedan tras completar todo el envío?' if es else f'A warehouse starts with {v("opening")} units, receives {v("incoming")}, and must then ship {v("outgoing")}. How many units remain after completing the entire shipment?')
        alt=(f'Primero hay {v("opening")} artículos. Antes de atender un pedido de {v("outgoing")}, llegan {v("incoming")} más. Calcula las existencias después de entregar el pedido completo.' if es else f'Initially there are {v("opening")} items. Before fulfilling an order for {v("outgoing")}, another {v("incoming")} arrive. Calculate the stock after fulfilling the order in full.')
        facts=list(zip(['Existencias iniciales','Llegada previa al envío','Envío obligatorio'] if es else ['Opening stock','Arrival before shipment','Required shipment'],[v(k) for k in REQ[f]]))
        question='Calcula las existencias finales después del envío completo.' if es else 'Find closing stock after the full shipment.'
        rule='No se permite pedir prestado, dejar pedidos pendientes ni tener existencias negativas.' if es else 'Borrowing, backorders, and negative stock are not allowed.'
    elif f=='dependency_schedule':
        dep=('A solo puede comenzar cuando termine C; C solo puede comenzar cuando terminen A y B.' if es else 'A can start only after C finishes; C can start only after both A and B finish.') if p['cyclic'] else ('A y B pueden comenzar en el minuto 0; C solo comienza cuando ambas terminan.' if es else 'A and B may start at minute 0; C starts only after both finish.')
        normal=(f'Las tareas A, B y C duran {v("duration_a")}, {v("duration_b")} y {v("duration_c")} minutos, respectivamente. {dep} Calcula el instante más temprano de finalización de C.' if es else f'Tasks A, B, and C last {v("duration_a")}, {v("duration_b")}, and {v("duration_c")} minutes, respectively. {dep} Find the earliest finish time of C.')
        alt=(f'C necesita {v("duration_c")} minutos de trabajo, A necesita {v("duration_a")} y B necesita {v("duration_b")}. {dep} ¿En qué minuto puede terminar C como muy pronto?' if es else f'C needs {v("duration_c")} minutes of work, A needs {v("duration_a")}, and B needs {v("duration_b")}. {dep} At what minute can C finish at the earliest?')
        facts=list(zip(['Duración A','Duración B','Duración C'] if es else ['Duration A','Duration B','Duration C'],[v(k) for k in REQ[f]]))+[('Dependencias' if es else 'Dependencies',dep)]
        question='Encuentra la finalización más temprana de C.' if es else 'Find the earliest finish time of C.'
        rule='El tiempo comienza en 0. Hay trabajadores suficientes para todas las tareas listas. Todas las tareas deben terminar y sus duraciones son positivas.' if es else 'Time starts at 0. There are enough workers for all ready tasks. Every task must finish and all durations are positive.'
    elif f=='unit_conversion':
        normal=(f'Una muestra tiene una masa de {v("quantity")} {v("source_unit")}. Expresa su masa en {v("target_unit")}.' if es else f'A sample has mass {v("quantity")} {v("source_unit")}. Express its mass in {v("target_unit")}.')
        alt=(f'Cambia la unidad de la medida {v("quantity")} {v("source_unit")} a {v("target_unit")}, sin cambiar la masa.' if es else f'Change the unit of the measurement {v("quantity")} {v("source_unit")} to {v("target_unit")} without changing the mass.')
        facts=list(zip(['Valor','Unidad inicial','Unidad final'] if es else ['Value','Source unit','Target unit'],[v(k) for k in REQ[f]]))
        question='Convierte la masa a la unidad final.' if es else 'Convert the mass to the target unit.'
        rule='Una masa debe ser no negativa. Usa exactamente 1 kg = 1000 g y 1 g = 1000 mg, sin redondear.' if es else 'Mass must be nonnegative. Use exactly 1 kg = 1000 g and 1 g = 1000 mg, without rounding.'
    else:
        bounds=(f'x ≥ {v("lower")}, sin cota superior' if es else f'x ≥ {v("lower")}, with no upper bound') if p['no_upper'] else f'{v("lower")} ≤ x ≤ {v("upper")}'
        normal=(f'Para una variable real x que cumple {bounds}, maximiza {v("coefficient")}*x. Devuelve el valor máximo de la función objetivo.' if es else f'For real x satisfying {bounds}, maximize {v("coefficient")}*x. Return the maximum objective value.')
        alt=(f'El beneficio es {v("coefficient")} veces una cantidad real x. Sus límites son {bounds}. ¿Qué beneficio máximo se puede alcanzar?' if es else f'The payoff is {v("coefficient")} times a real quantity x. Its limits are {bounds}. What is the greatest attainable payoff?')
        facts=[('Coeficiente' if es else 'Coefficient',v('coefficient')),('Restricciones' if es else 'Constraints',bounds)]
        question='Calcula el máximo de la función objetivo, no el valor de x.' if es else 'Find the maximum objective value, not the value of x.'
        rule='No existen otras restricciones. Una región vacía es infeasible; un objetivo sin máximo finito es unbounded.' if es else 'There are no other constraints. An empty feasible region is infeasible; an objective with no finite maximum is unbounded.'
    text=alt if para else normal
    if view=='fact_reorder':text='\n'.join(k+': '+str(vv) for k,vv in reversed(facts))+'\n'+question
    text+='\n'+rule
    text+=('\nReferencia de archivo (no afecta al cálculo): ' if es else '\nArchive reference (does not affect the calculation): ')+v('archive_ref')+'.'
    if view=='distractor':text+=('\nLa portada tiene 17 líneas y la carpeta lleva el número 203. Son datos administrativos que no afectan al problema.' if es else '\nThe cover has 17 lines and the folder is numbered 203. These are administrative details that do not affect the problem.')
    return text

def system(lang,key):
    if lang=='es':return ('Resuelve exactamente con los datos y restricciones indicados. Devuelve solo un objeto JSON con las claves status y answer. '
      'status debe ser solved, needs_clarification, infeasible o unbounded. ? significa un dato desconocido; pide aclaración solo si impide determinar la respuesta. '
      'Las restricciones explícitas imposibles dan infeasible. Si status es solved, answer debe contener únicamente la clave '+key+
      ' con un número como cadena racional reducida: "3" o "3/2", nunca decimal. En otro caso answer debe ser {}. No incluyas explicación ni otras claves.')
    return ('Solve exactly using the stated data and constraints. Return only a JSON object with keys status and answer. '
      'status must be solved, needs_clarification, infeasible, or unbounded. ? means unknown data; request clarification only if it prevents determining the answer. '
      'Explicitly impossible constraints yield infeasible. If status is solved, answer must contain only key '+key+
      ' with a number as a reduced rational string: "3" or "3/2", never a decimal. Otherwise answer must be {}. Do not include an explanation or other keys.')

def verify_expected(row,text):
    try:
        def unique(items):
            d={}
            for k,v in items:
                if k in d:raise ValueError('duplicate key')
                d[k]=v
            return d
        actual=json.loads(text,object_pairs_hook=unique)
        return canonical(actual)==canonical(row['expected_response'])
    except (ValueError,TypeError):return False

def main():
    rows=[];catalog=[];truth=[];used=set();stats=collections.Counter();preview=[]
    for family in FAMILIES:
        for j in range(24):
            level=j%3
            for attempt in range(100000):
                p=make(family,level)
                if family=='unit_conversion' and p['quantity']=='1':continue
                variations=variant_facts(family,p)
                # Ignore the irrelevant reference to catch superficial duplicates.
                hashes={digest([family,{k:v for k,v in q.items() if k!='archive_ref'}]) for _,_,q in variations}
                if not hashes.isdisjoint(used):continue
                break
            else:raise RuntimeError('Novelty exhaustion '+family)
            used.update(hashes)
            group=digest([family,p])[:24];cohort='development' if j<16 else 'confirmation'
            base=solve(family,p)
            root={'group_id':group,'family':family,'cohort':cohort,'draw_scale':['small','medium','large'][level],
                  'facts':p,'note':'draw_scale affects numeric draws in most families; not a measured difficulty rating.'}
            catalog.append(root)
            for view,semantic,q in variations:
                status,ans=solve(family,q)
                assert (status,ans)==second_oracle(family,q),(family,q)
                if semantic=='base' or view=='missing_optional':assert (status,ans)==base
                if view=='counterfactual':assert status=='solved' and (status,ans)!=base
                if view=='missing_required':assert status=='needs_clarification'
                if view=='infeasible':assert status=='infeasible'
                if view=='unbounded':assert status=='unbounded'
                for lang in ['en','es']:
                    expected={'status':status,'answer':ans}
                    row={'id':digest([group,view,lang])[:24],'group_id':group,'semantic_id':group+':'+semantic,
                         'split':'test','family':'retention_'+family,'language':lang,'view':view,'kind':'retention',
                         'prompt':[{'role':'system','content':system(lang,KEY[family])},{'role':'user','content':wording(family,q,lang,view)}],
                         'expected_response':expected,'source':'coverage_extension_20261013',
                         'evaluation_suite':cohort,'task_facts':q,'training_eligible':False}
                    assert verify_expected(row,canonical(expected))
                    bad=deepcopy(expected)
                    if status=='solved':bad['answer'][KEY[family]]=rat(F(ans[KEY[family]])+1)
                    else:bad['status']='solved'
                    assert not verify_expected(row,canonical(bad))
                    if view=='paraphrase':assert wording(family,q,lang,view)!=wording(family,q,lang,'base')
                    rows.append(row);truth.append({'id':row['id'],'group_id':group,'expected_response':expected})
                    stats['oracle_and_mutation_checks']+=1
            if j==0:
                preview += [family.upper(),'BASE (English)',wording(family,p,'en','base'),canonical({'status':base[0],'answer':base[1]}),'PARAPHRASE (Spanish)',wording(family,p,'es','paraphrase'),'']
    pilot={r['group_id'] for f in FAMILIES for r in [q for q in catalog if q['family']==f and q['cohort']=='development'][:2]}
    dev=[r for r in rows if r['evaluation_suite']=='development'];confirmation=[r for r in rows if r['evaluation_suite']=='confirmation']
    assert len(catalog)==168 and len(rows)==2736 and len(dev)==1824 and len(confirmation)==912
    assert len({r['id'] for r in rows})==len(rows)
    assert {r['group_id'] for r in dev}.isdisjoint({r['group_id'] for r in confirmation})
    jl('development112.jsonl',dev);jl('confirmation56.jsonl',confirmation)
    jl('pilot14.jsonl',[r for r in dev if r['group_id'] in pilot])
    jl('root_catalog.jsonl',catalog);jl('answer_key.jsonl',truth)
    # Prompt-only exports reduce accidental oracle leakage when using other runners.
    for name,rs in [('development112',dev),('confirmation56',confirmation),('pilot14',[r for r in dev if r['group_id'] in pilot])]:
        jl(name+'_prompts_only.jsonl',[{'id':r['id'],'messages':r['prompt']} for r in rs])
    (ROOT/'preview.txt').write_text('\n'.join(preview)+'\n',encoding='utf-8')
    manifest={'candidate':True,'seed':SEED,'roots':168,'rows':2736,'families':FAMILIES,'languages':['en','es'],
      'views':['base','paraphrase','fact_reorder','distractor','counterfactual','missing_required','missing_optional','infeasible'],
      'extra_view':'unbounded for linear_optimization only','cohorts':{'development':{'roots':112,'rows':1824},'confirmation':{'roots':56,'rows':912},'pilot':{'roots':14,'rows':228,'subset_of':'development'}},
      'status_counts':dict(collections.Counter(r['expected_response']['status'] for r in rows)),
      'family_counts':dict(collections.Counter(r['family'] for r in rows)),
      'validation':dict(stats),'split':'test','training_eligible':False,
      'transport':'existing retention exact-JSON branch; synthetic transfer diagnostic, not external retention benchmark',
      'format':'Only status and answer. Canonical reduced rational strings. Different from old four-key math contract; compare models within this suite.',
      'limitations':['Only English/Spanish; authored translations need human bilingual review.',
       'Two authored task formulations per language, plus facts-list rendering; not unconstrained linguistic diversity.',
       'Shared generators/templates: unique roots are not independent draws from all real-world tasks.',
       'Negative counts/masses are intentionally contradictory diagnostic inputs, not naturalistic task prevalence.',
       'Counterfactual difficulty is not matched. Number magnitude is not a measured difficulty rating.',
       'No unrelated open-ended retention, tool use, visual reasoning, adversarial safety, or pretraining-contamination claim.',
       'No tokenizer/model/GPU execution performed.'],
      'sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(ROOT.glob('*.jsonl'))}}
    (ROOT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'roots':len(catalog),'rows':len(rows),'checks':dict(stats),'statuses':manifest['status_counts']},indent=2))
if __name__=='__main__':main()
