from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser(); p.add_argument('--calibration-dir',required=True); p.add_argument('--steps',type=int,default=10000); p.add_argument('--device',default='cuda'); p.add_argument('--dry-run',action='store_true'); a=p.parse_args()
    root=Path(__file__).resolve().parents[1]; cal=Path(a.calibration_dir)
    for candidate in ('safety_dominant','balanced','event_dominant'):
        for seed in (0,1,2):
            out=root/'outputs'/f'v4_calibrated_{candidate}_10k_probe_s{seed}'
            cmd=[sys.executable,'-u',str(root/'scripts/train_happo_reference.py'),'--config',str(cal/f'v4_{candidate}.yaml'),'--reward-mode','brma_tam_paper_calibrated_v4','--output-dir',str(out),'--total-env-steps',str(a.steps),'--rollout-length','256','--num-envs','1','--max-steps','1000','--device',a.device,'--policy-arch','pure_happo','--opponent-policy','tam_greedy_rule','--enable-rich-logging','--rich-log-dir',str(out/'rich_logs'),'--seed',str(seed)]
            print(' '.join(cmd),flush=True)
            if not a.dry_run: subprocess.run(cmd,cwd=root,check=True)
if __name__=='__main__': main()
