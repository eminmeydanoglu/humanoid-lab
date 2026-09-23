#!/usr/bin/env python3
"""Independent direct psi0 predictions on recorded BlockStacking observations."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np

PROMPT="Stack the three cubic blocks on the black tape in the order red, yellow, blue."
WINDOWS=(("h0",0,1),("h1_5",1,6),("h6_15",6,16),("h16_29",16,30))
FSQ_STEP=0.0625; FSQ_MIN=-0.625; FSQ_MAX=0.625

def phase_offsets(n:int): return [(p,min(n-1,max(0,round(c*(n-1))))) for p,c in (("early",.15),("middle",.5),("late",.85))]
def fsq(x): return np.clip(np.round(np.clip(x,FSQ_MIN,FSQ_MAX)/FSQ_STEP)*FSQ_STEP,FSQ_MIN,FSQ_MAX)
def error(a,b):
 d=np.asarray(a,np.float64)-np.asarray(b,np.float64); return {"mae":float(np.abs(d).mean()),"rmse":float(np.sqrt((d*d).mean()))}
def aligned_chunk(actions,start,horizon):
 end=min(len(actions),start+horizon); return actions[start:end], end-start
def repeated_hold(actions,start,n):
 prev=actions[max(0,start-1)]; return np.repeat(prev[None],n,axis=0)

def no_aug_state_row(window, temporal_jitter, num_past_frames=0):
 """Current state selected by SonicRepackTransform when no_aug disables jitter."""
 end=num_past_frames+temporal_jitter+1
 return np.asarray(window)[end-(num_past_frames+1):end]
def summarize(rows):
 out={}
 for name,lo,hi in WINDOWS:
  valid=[r for r in rows if r["valid_horizon"]>=hi]
  out[name]={"samples":len(valid)}
  for group,sl in (("token",slice(0,64)),("hands",slice(64,78))):
   out[name][group]={}
   target=np.concatenate([np.asarray(r["target"])[lo:hi,sl] for r in valid])
   for method in ("model","hold","mean"):
    pred=np.concatenate([np.asarray(r[method])[lo:hi,sl] for r in valid])
    out[name][group][method]=error(pred,target)
   if group=="token":
    pred=np.concatenate([fsq(np.asarray(r["model"])[lo:hi,sl]) for r in valid])
    out[name][group]["model_fsq"]=error(pred,target)
 return out

def main():
 p=argparse.ArgumentParser(); p.add_argument('--run-dir',type=Path,required=True); p.add_argument('--dataset-root',type=Path,required=True); p.add_argument('--split',choices=['train','val'],required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--episodes',type=int,default=4); p.add_argument('--seeds',type=int,nargs='+',default=[292285,292286]); p.add_argument('--anchors-json',type=Path); a=p.parse_args()
 import torch, pyarrow.parquet as pq
 from PIL import Image
 from torchvision.transforms import v2
 from transformers import AutoProcessor
 from psi.config.config import LaunchConfig
 from psi.config.data_lerobot import LerobotDataConfig
 from psi.models.psi0 import Psi0Model,QWEN3VL_VARIANT
 from psi.deploy.serve_psi0_sonic import PooledTextEncoderCache
 from psi.utils import parse_args_to_tyro_config,apply_legacy_model_config_defaults,seed_everything
 cfg0:LaunchConfig=parse_args_to_tyro_config(a.run_dir/'argv.txt') # type:ignore
 cfg=cfg0.model_validate(apply_legacy_model_config_defaults(json.loads((a.run_dir/'run_config.json').read_text())))
 processor=AutoProcessor.from_pretrained(QWEN3VL_VARIANT,local_files_only=True)
 data_cfg:LerobotDataConfig=cfg.data # type:ignore
 ds=data_cfg(split=a.split,transform_kwargs={'vlm_processor':processor,'no_aug':True})
 idx=ds.raw_dataset.base_dataset.episode_data_index
 meta=[json.loads(x) for x in (a.dataset_root/a.split/'meta/episodes.jsonl').read_text().splitlines()]
 candidates=[e['episode_index'] for e in meta if e['tasks']==[PROMPT]]
 anchor_map=None
 if a.anchors_json:
  old=json.loads(a.anchors_json.read_text()); anchor_map={}
  for row in old['rows']: anchor_map[(int(row['episode']),int(row['frame']))]=str(row['phase'])
  selected=sorted({e for e,_ in anchor_map})
 else:
  selected=[candidates[i] for i in np.linspace(0,len(candidates)-1,a.episodes,dtype=int)]
 sums=np.zeros(78); count=0
 for e in meta:
  if e['tasks']!=[PROMPT]: continue
  ep=e['episode_index']; path=a.dataset_root/a.split/'data'/f'chunk-{ep//1000:03d}'/f'episode_{ep:06d}.parquet'
  t=pq.read_table(path,columns=['action.body_token_v1_1','action']); arr=np.c_[np.stack(t.column(0).to_pylist()),np.stack(t.column(1).to_pylist())]; sums+=arr.sum(0); count+=len(arr)
 mean=sums/count
 seed_everything(a.seeds[0]); model=Psi0Model.from_pretrained(a.run_dir,40000,cfg,device='cuda:0').to('cuda:0').eval()
 mc=cfg.model; cache=mc.pooled_cache_path if Path(str(mc.pooled_cache_path)).is_absolute() else a.run_dir/str(mc.pooled_cache_path)
 pooled=PooledTextEncoderCache(cache_path=str(cache),encoder=mc.pooled_text_encoder,encoder_path=mc.pooled_text_encoder_path,projection_dim=mc.pooled_projection_dim,device=torch.device('cuda:0'))
 transform=v2.Compose([cfg.data.transform.model.resize(),cfg.data.transform.model.center_crop()]) # type:ignore
 rows=[]; input_min=np.full(45,np.inf); input_max=np.full(45,-np.inf)
 for ep in selected:
  start,end=int(idx['from'][ep]),int(idx['to'][ep]); episode_actions=[]
  path=a.dataset_root/a.split/'data'/f'chunk-{ep//1000:03d}'/f'episode_{ep:06d}.parquet'; t=pq.read_table(path,columns=['action.body_token_v1_1','action']); episode_actions=np.c_[np.stack(t.column(0).to_pylist()),np.stack(t.column(1).to_pylist())].astype(np.float32)
  anchors=(phase_offsets(end-start) if anchor_map is None else [(phase,off) for (episode,off),phase in anchor_map.items() if episode==ep])
  for phase,off in anchors:
   item=ds[start+off]; state=np.asarray(item['states'],np.float32); state=state[-1:] if state.ndim==2 else state[None]; input_min=np.minimum(input_min,state.min(0)); input_max=np.maximum(input_max,state.max(0))
   images=[[transform(Image.fromarray(np.asarray(im,dtype=np.uint8))) for im in item['observations']]]
   target,n=aligned_chunk(episode_actions,off,30); hold=repeated_hold(episode_actions,off,n); mean_chunk=np.repeat(mean[None],n,0)
   for seed in a.seeds:
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    instruction=str(item['instruction']).lower()
    with torch.inference_mode(): norm=model.predict_action(observations=images,states=torch.from_numpy(state[None]).cuda(),instructions=[instruction],num_inference_steps=8,traj2ds=None,pooled_projections=pooled(instruction))
    pred=cfg.data.transform.field.denormalize(norm[0].float().cpu().numpy())[:n] # type:ignore
    rows.append({'split':a.split,'episode':ep,'frame':off,'phase':phase,'seed':seed,'valid_horizon':n,'target':target.tolist(),'model':pred.tolist(),'hold':hold.tolist(),'mean':mean_chunk.tolist()})
 result={'schema_version':1,'mode':'direct_independent_unguided','checkpoint':str(a.run_dir/'checkpoints/ckpt_40000'),'split':a.split,'episodes':selected,'seeds':a.seeds,'rows':rows,'summary':summarize(rows),'normalized_state_input_min':input_min.tolist(),'normalized_state_input_max':input_max.tolist(),'contract':{'camera':'observation.images.egocentric; canonical no_aug resize+center_crop','prediction':'Psi0Model.predict_action, 8 diffusion steps, no RTC/previous_action/controller/server','alignment':'prediction row k against recorded action at anchor+k; rows beyond episode end masked','action_layout':'64 body tokens + 14 hands; model padding dims 78:80 excluded','fsq':'token predictions rounded/clipped to [-0.625,0.625] step 0.0625'}}
 a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result['summary'],indent=2))
if __name__=='__main__': main()
