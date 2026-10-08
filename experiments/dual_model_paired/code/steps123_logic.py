"""Deterministic diagnostic selection and statistics; no simulator imports."""
import numpy as np

def candidate_subset(rows,label):
 c=sorted([r for r in rows if r['static_candidate']],key=lambda r:r['control_step'])
 if label=='A':
  if len(c)>15:raise RuntimeError('A candidate authorization exceeded')
  return c
 return c if len(c)<=5 else [c[j*(len(c)-1)//4] for j in range(5)]

def controls(rows):
 rows=sorted(rows,key=lambda r:r['control_step'])
 c=[r for r in rows if r['static_candidate']]
 eligible=[r for r in rows if not r['over_threshold'] and all(v for k,v in r['static_filters'].items() if k!='threshold')]
 if not eligible:return []
 if not c:
  return list({r['control_step']:r for r in [eligible[(len(eligible)-1)//3],eligible[2*(len(eligible)-1)//3]]}.values())
 chosen=[]
 for anchor in [c[0],c[-1]]:
  pool=[r for r in eligible if r['control_step'] not in [x['control_step'] for x in chosen]]
  if not pool:continue
  predicates=anchor.get('truth',{}).get('predicates')
  same=[r for r in pool if predicates is not None and r.get('truth',{}).get('predicates')==predicates]
  pool=same or pool
  r=min(pool,key=lambda r:(abs(r['control_step']-anchor['control_step']),r['control_step']))
  chosen.append(dict(r,control_anchor_step=anchor['control_step'],control_same_predicates=bool(same)))
 return chosen

def distance(a,b,k=5):
 a,b=np.asarray(a,dtype=np.float64)[:k],np.asarray(b,dtype=np.float64)[:k]
 if k<=0 or a.shape!=(k,7) or b.shape!=(k,7) or not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('invalid comparable prefix')
 d=a-b
 return float((np.sqrt(np.mean(d[:,:3]**2))+np.sqrt(np.mean(d[:,3:6]**2))+np.mean(np.abs(d[:,6])))/3)

def pair_stats(props,dmin):
 if any(len(props.get(r,[]))!=5 for r in ['main','aux']):return {'noise_separated':'UNDETERMINED','pairs':[],'summary':{}}
 pairs=[];stats={}
 for category,ra,rb in [('main_within','main','main'),('aux_within','aux','aux'),('cross_model','main','aux')]:
  vals=[]
  for i in range(5):
   for j in range(5):
    if ra==rb and i>=j:continue
    d=distance(props[ra][i]['canonical_actions'],props[rb][j]['canonical_actions']);vals.append(d)
    pairs.append(dict(category=category,i=i,j=j,D=d,over_threshold=d>=dmin))
  stats[category]=dict(pairs=len(vals),median=float(np.median(vals)),p90=float(np.percentile(vals,90)),maximum=max(vals),over_fraction=sum(d>=dmin for d in vals)/len(vals))
 separated=stats['cross_model']['median']>max(stats['main_within']['p90'],stats['aux_within']['p90']) and stats['cross_model']['over_fraction']>=.8
 return dict(noise_separated=bool(separated),pairs=pairs,summary=stats)

def select_fork(rows,repeat_results,restorable,horizon=520):
 eligible=[];excluded=[]
 for row in rows:
  if not row['static_candidate']:continue
  step=row['control_step'];why=[];r=repeat_results.get(step)
  if not row['over_threshold']:why.append('original_threshold')
  if not r or r.get('noise_separated') is not True:why.append('noise_separation_missing_or_false')
  if not restorable(row):why.append('complete_restore_material_missing')
  if horizon-step<10:why.append('insufficient_shared_horizon')
  if row.get('failure_filter_pass',True) is not True:why.append('failure_filter')
  if why:excluded.append(dict(control_step=step,reasons=why))
  else:eligible.append(row)
 eligible.sort(key=lambda x:(-x['D_proposed'],x['control_step']))
 return (eligible[0] if eligible else None),excluded
