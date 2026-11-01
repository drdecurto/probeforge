"""Generate LaTeX tables from the frozen CSV snapshot; does not alter prose."""
import csv
from pathlib import Path
ROOT = Path(__file__).resolve().parent

def rows(name):
    with (ROOT/'source_tables'/name).open() as f:
        return list(csv.DictReader(f))
def esc(s):
    return str(s).replace('_', r'\_').replace('%', r'\%').replace('&', r'\&')
def table(name, caption, label, heads, data, spec=None):
    n=len(heads)
    text='\\begin{table}['+('H' if name in {'languages','views'} else 'htbp')+']\n\\caption{'+caption+'}\\label{tab:'+label+'}\n\\centering\\small\n'
    text+='\\setlength{\\tabcolsep}{4pt}\n\\begin{tabular}{'+(spec or 'l'+'r'*(n-1))+'}\n\\toprule\n'
    text+=' & '.join(heads)+r'\\\midrule'+'\n'
    text+='\n'.join(' & '.join(esc(c) for c in row)+r'\\' for row in data)
    text+='\n\\bottomrule\n\\end{tabular}\n\\end{table}\n'
    (ROOT/'tables'/f'{name}.tex').write_text(text)
r=rows('table1_main_benchmark.csv')
table('main','Main benchmark (1248 prompts). SD is between-seed sample standard deviation; a dash denotes one run. Worst is the mean of per-seed worst-language accuracy. Bnd. and CF denote boundary and counterfactual views.','main',['Model','Arm','$n$','Accuracy','SD','Worst','Bnd.','CF','Tokens'],[[x['model'],x['arm'],x['seeds'],x['pass_at_1'],x['sd'] if int(x['seeds'])>1 else '--',x['worst_language'],x['boundary'],x['counterfactual'],x['tokens']] for x in r], 'llrrrrrrr')
r=rows('table4_packs.csv')
table('packs','Additional evaluation packs (6240 prompts each). Intervals bootstrap roots for selected checkpoints.','packs',['Pack','Checkpoint','Accuracy',r'95\% CI','Worst','Bnd.','CF'],[[x['pack'],x['model'],x['pass_at_1'],f"[{x['ci_lo']}, {x['ci_hi']}]",x['worst_language'],x['boundary'],x['counterfactual']] for x in r],'llrlrrr')
r=rows('table6_dev112_transfer.csv')
table('transfer','Development112 (1824 prompts): lenient status-and-answer accuracy under two response protocols. Dashes are unreported conditions.','transfer',['Checkpoint','Two-key protocol','Four-key protocol'],[[x['model'],x['pack_protocol_2key_lenient'] or '--',x['native_protocol_4key_lenient'] or '--'] for x in r])
r=rows('table5_regimes.csv')
def score(x, mean, sd, seeds):
    return f"{float(x[mean]):.3f}" + (r" $\pm$ " + f"{float(x[sd]):.3f}" if int(x[seeds]) > 1 else '')
table('regimes',r'Qwen3-14B accuracy by inference regime. Replicated entries report mean $\pm$ sample SD. $n_{\mathrm{NT}}/n_{\mathrm{T}}$ gives the numbers of evaluated runs in the non-thinking/thinking regimes. Thinking-token counts are seed-averaged on the 26-prompt subset and exclude the final answer tokens.','regimes',['Adapter',r'$n_{\mathrm{NT}}/n_{\mathrm{T}}$','No-think (1248)','Think (26)','Think (52)','Think tokens'],[[x['adapter'],x['seeds_no_think']+'/'+x['seeds_thinking'],score(x,'no_think_test_eval12','no_think_sd','seeds_no_think'),score(x,'think26_main','think26_sd','seeds_thinking'),score(x,'think60_diag60','think60_sd','seeds_thinking'),x['think_tokens_think26']] for x in r], 'lcrrrr')
r=rows('table2_per_language.csv'); langs=list(r[0])[1:]
table('languages','Language-level accuracy for the selected series. Qwen contrast columns average three seeds.','langs',['Language','8B base','8B mined','14B base','14B random','Phi-4 mined'],[[l]+[x[l] for x in r] for l in langs])
r=rows('table3_per_view.csv'); views=list(r[0])[1:]
table('views','View-level accuracy. Last two columns refer to diag60; other columns refer to the main benchmark.','views-ext',['View','8B base','8B mined','14B base','14B random','8B diag','14B diag'],[[v.replace('context_','ctx. ')]+[x[v] for x in r] for v in views])

# v6 external benchmarks. Existing experiment tables retain the v5 snapshot.
import json
fr=[]
for label,fn,budget in [('Main','report12_muse.json','4096'),('Core60','report_core60_muse.json','4096'),('Diag60','report_diag60_muse.json','4096'),('Main (512-token diagnostic)','report12_muse_strict512.json','512')]:
 j=json.loads((ROOT/'evidence/round14'/fn).read_text());q=j['budgets'][budget];ci=q['group_mean_accuracy_ci95']
 fr.append([label,str(j['expected_examples']),f"{q['pass_at_1']:.3f}",f'[{ci[0]:.3f}, {ci[1]:.3f}]',f"{q['worst_language']:.3f}",str(round(q['mean_completion_tokens']))])
table('frontier',r'Muse-Spark-Open reference on ProbeForge. Primary runs use 4096 tokens and low reasoning effort; the 512-token run is a separate protocol diagnostic. CIs bootstrap roots. Tokens include generated reasoning and answer content. These runs use 16 GPUs and are not compute-matched to the adapted models.','frontier',['Set','Prompts','Accuracy',r'95\% CI','Worst','Tokens'],fr,'lrrlrr')
label={'phi-4-mini-instruct':'Phi-4-mini anchor','phi-4-base':'Phi-4 base','phi-4-mined':'Phi-4 mined','qwen3-8b-base':'Qwen3-8B base','qwen3-8b-mined':'Qwen3-8B mined','qwen3-8b-v3random':'Qwen3-8B v3 random','qwen3-14b-base':'Qwen3-14B base','qwen3-14b-random':'Qwen3-14B random','qwen3-14b-mined':'Qwen3-14B mined','qwen3-14b-v3mined':'Qwen3-14B v3 mined','qwen3-14b-distill':'Qwen3-14B distill','qwen3-14b-mixed50':'Qwen3-14B mixed50'}
r=rows('table7_energy_benchmark.csv');out=[]
for x in r:
 if x['protocol']=='raw' and x['slot']=='phi-4-base':continue
 out.append([x['protocol'],label[x['slot']],f"{float(x['mean']):.3f}",f"{float(x['step_accuracy']):.3f}",f"{float(x['wh_per_1000_queries']):.1f}"])
table('energybench',r'Previously published reasoning benchmark: local 13-language means under two protocols. Score is whole-response reference similarity; Step is the first-response step-alignment subscore, not verified step accuracy. Energy is Wh per 1000 problems, with three generated responses per problem, including scoring intervals. Raw Phi-4 runs are excluded owing to empty completions and incomplete adapter coverage.','energybench',['Protocol','Checkpoint','Score','Step','Energy (Wh)'],out,'llrrr')
r=rows('table8_standard_benchmarks.csv')
table('standard',r'Standard benchmark scores for 18 configurations. Columns give language means for Belebele (12 configurations), Global-MMLU-Lite (8), and MGSM (6). MGSM strict extraction is primary; last-number extraction is shown separately. Each row is a single selected checkpoint evaluation, not a training-seed average.','standard',['Checkpoint','Belebele','Global-MMLU','MGSM strict','MGSM last'],[[x['model']]+[f"{float(x[c]):.3f}" for c in ['belebele12','global_mmlu_lite8','mgsm6_strict','mgsm6_flexible']] for x in r],'lrrrr')
