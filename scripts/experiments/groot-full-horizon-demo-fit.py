#!/usr/bin/env python3
"""Bounded full-horizon GR00T predictions on the existing exp11 anchors."""
import argparse,json
from pathlib import Path
import numpy as np
FSQ_MIN,FSQ_MAX,FSQ_STEP=-.625,.625,.0625
WINDOWS=(("h0",0,1),("h1_5",1,6),("h6_15",6,16),("h16_39",16,40))
def fsq(x): return np.clip(np.round(np.clip(x,FSQ_MIN,FSQ_MAX)/FSQ_STEP)*FSQ_STEP,FSQ_MIN,FSQ_MAX)
def err(a,b): d=np.asarray(a)-np.asarray(b); return {'mae':float(np.abs(d).mean()),'rmse':float(np.sqrt((d*d).mean()))}
def summary(rows):
 out={}
 for name,lo,hi in WINDOWS:
  valid=[r for r in rows if r['valid_horizon']>=hi]; out[name]={'samples':len(valid)}
  for group,sl in [('motion_token',slice(0,64)),('left_hand',slice(64,71)),('right_hand',slice(71,78))]:
   target=np.concatenate([np.asarray(r['target'])[lo:hi,sl] for r in valid]); out[name][group]={}
   for m in ['model','hold','mean']: out[name][group][m]=err(np.concatenate([np.asarray(r[m])[lo:hi,sl] for r in valid]),target)
   if group=='motion_token': out[name][group]['model_fsq']=err(np.concatenate([fsq(np.asarray(r['model'])[lo:hi,sl]) for r in valid]),target)
 return out
def main():
 p=argparse.ArgumentParser(); p.add_argument('--checkpoint',required=True); p.add_argument('--dataset',type=Path,required=True); p.add_argument('--anchors',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--seeds',type=int,nargs='+',default=[292285,292286]); a=p.parse_args()
 import torch
 from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
 from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
 from gr00t.data.embodiment_tags import EmbodimentTag
 from gr00t.data.utils import parse_observation_gr00t
 from gr00t.policy.gr00t_policy import Gr00tPolicy
 tag=EmbodimentTag.resolve('unitree_g1_sonic'); policy=Gr00tPolicy(embodiment_tag=tag,model_path=a.checkpoint,device='cuda'); policy.model.action_head.num_inference_timesteps=4
 mod=policy.get_modality_config(); loader=LeRobotEpisodeLoader(dataset_path=str(a.dataset),modality_configs=mod); keys=mod['action'].modality_keys; obsmod=dict(mod); obsmod.pop('action')
 anchors=json.loads(a.anchors.read_text())['rows']; episodes=sorted({r['episode'] for r in anchors})
 sums=np.zeros(78); count=0
 meta=[json.loads(x) for x in (a.dataset/'meta/episodes.jsonl').read_text().splitlines()]
 prompt=anchors[0].get('prompt','Stack the three cubic blocks on the black tape in the order red, yellow, blue.')
 for e in [x['episode_index'] for x in meta if x['tasks'][0].startswith('Stack the three')]:
  tr=loader[e]; ar=np.concatenate([np.vstack(tr[f'action.{k}']) for k in keys],1); sums+=ar.sum(0); count+=len(ar)
 mean=sums/count; rows=[]
 for spec in anchors:
  ep,frame=spec['episode'],spec['frame']; tr=loader[ep]; target_all=np.concatenate([np.vstack(tr[f'action.{k}']) for k in keys],1); n=min(40,len(tr)-frame); target=target_all[frame:frame+n]; hold=np.repeat(target_all[max(0,frame-1)][None],n,0); meanchunk=np.repeat(mean[None],n,0)
  point=extract_step_data(tr,frame,obsmod,tag); obs={**{f'state.{k}':v for k,v in point.states.items()},**{f'video.{k}':np.asarray(v) for k,v in point.images.items()}};
  for k in mod['language'].modality_keys: obs[k]=point.text
  parsed=parse_observation_gr00t(obs,mod)
  for seed in a.seeds:
   torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); pred,_=policy.get_action(parsed); chunk=np.concatenate([pred[k][0] for k in keys],1)[:n]
   rows.append({'episode':ep,'frame':frame,'phase':spec['phase'],'seed':seed,'valid_horizon':n,'target':target.tolist(),'model':chunk.tolist(),'hold':hold.tolist(),'mean':meanchunk.tolist()})
 result={'mode':'direct_independent','checkpoint':a.checkpoint,'episodes':episodes,'seeds':a.seeds,'rows':rows,'summary':summary(rows),'semantics':{'motion_token':'action.motion_token, FSQ body-command token','left_hand':'teleop.left_hand_joints recorded command labels, not measured observation.state hand joints','right_hand':'teleop.right_hand_joints recorded command labels, not measured observation.state hand joints','alignment':'prediction k versus recorded action anchor+k; episode end masked'}}
 a.output.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__': main()
