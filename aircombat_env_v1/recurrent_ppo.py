"""Minimal sequence-correct recurrent PPO with continuous and fire heads."""

from __future__ import annotations
import math
import numpy as np
import torch
from torch import nn
from torch.distributions import Bernoulli
from .ppo import SquashedNormal, _orthogonal, LOG_STD_MIN, LOG_STD_MAX, compute_gae


class RecurrentActor(nn.Module):
    def __init__(self, observation_dim=26, hidden_size=128, action_dim=3):
        super().__init__(); self.hidden_size = hidden_size
        self.encoder = nn.Sequential(_orthogonal(nn.Linear(observation_dim,128),math.sqrt(2)),nn.Tanh())
        self.gru = nn.GRU(128,hidden_size,batch_first=True)
        self.mean = _orthogonal(nn.Linear(hidden_size,action_dim),.01)
        self.log_std = _orthogonal(nn.Linear(hidden_size,action_dim),.01)
        self.fire = _orthogonal(nn.Linear(hidden_size,1),.01)
        # Engineering initialization: nominal deterministic evaluation must
        # exercise the weapon head instead of sitting exactly on p(fire)=0.5.
        nn.init.constant_(self.fire.bias, 1.0)

    def initial_hidden(self, batch, device=None):
        return torch.zeros(1,batch,self.hidden_size,device=device)

    def features(self, observations, hidden, episode_starts=None, active=None):
        if observations.ndim == 2: observations = observations[:,None,:]
        batch, steps = observations.shape[:2]; outputs=[]; state=hidden
        starts = torch.zeros(batch,steps,device=observations.device) if episode_starts is None else episode_starts
        for t in range(steps):
            state = state * (1.0-starts[:,t]).view(1,batch,1)
            out,state = self.gru(self.encoder(observations[:,t])[:,None,:],state)
            if active is not None: state = state*active[:,t].view(1,batch,1)
            outputs.append(out[:,0])
        return torch.stack(outputs,1),state

    def distributions(self, observations, hidden, episode_starts=None, active=None):
        features,state=self.features(observations,hidden,episode_starts,active)
        frac=torch.sigmoid(self.log_std(features)); log_std=LOG_STD_MIN+frac*(LOG_STD_MAX-LOG_STD_MIN)
        return SquashedNormal(self.mean(features),log_std.exp()), Bernoulli(logits=self.fire(features).squeeze(-1)),state

    def act(self, observations, hidden, episode_starts=None, deterministic=False):
        maneuver_dist,fire_dist,state=self.distributions(observations,hidden,episode_starts)
        maneuver=maneuver_dist.mean if deterministic else maneuver_dist.sample()
        fire=(fire_dist.probs>=.5).float() if deterministic else fire_dist.sample()
        log_prob=maneuver_dist.log_prob(maneuver)+fire_dist.log_prob(fire)
        entropy=maneuver_dist.entropy()+fire_dist.entropy()
        return maneuver[:,0],fire[:,0],log_prob[:,0],entropy[:,0],state


class RecurrentCritic(nn.Module):
    def __init__(self, observation_dim=26, hidden_size=128):
        super().__init__(); self.hidden_size=hidden_size
        self.encoder=nn.Sequential(_orthogonal(nn.Linear(observation_dim,128),math.sqrt(2)),nn.Tanh())
        self.gru=nn.GRU(128,hidden_size,batch_first=True)
        self.value=_orthogonal(nn.Linear(hidden_size,1),1.0)
    def initial_hidden(self,batch,device=None): return torch.zeros(1,batch,self.hidden_size,device=device)
    def forward(self,observations,hidden,episode_starts=None,active=None):
        if observations.ndim==2: observations=observations[:,None,:]
        batch,steps=observations.shape[:2]; starts=torch.zeros(batch,steps,device=observations.device) if episode_starts is None else episode_starts
        outputs=[]; state=hidden
        for t in range(steps):
            state=state*(1-starts[:,t]).view(1,batch,1)
            out,state=self.gru(self.encoder(observations[:,t])[:,None,:],state)
            if active is not None: state=state*active[:,t].view(1,batch,1)
            outputs.append(self.value(out[:,0]).squeeze(-1))
        return torch.stack(outputs,1),state


class RecurrentRolloutBuffer:
    def __init__(self,steps,envs,obs_dim=26,hidden=128):
        s=(steps,envs); self.steps=steps; self.envs=envs; self.pos=0
        self.observations=np.zeros(s+(obs_dim,),np.float32); self.maneuvers=np.zeros(s+(3,),np.float32)
        self.fire=np.zeros(s,np.float32); self.log_probs=np.zeros(s,np.float32); self.values=np.zeros(s,np.float32)
        self.next_values=np.zeros(s,np.float32); self.rewards=np.zeros(s,np.float32)
        self.terminated=np.zeros(s,bool); self.truncated=np.zeros(s,bool); self.episode_starts=np.zeros(s,np.float32)
        self.actor_hidden=np.zeros(s+(hidden,),np.float32); self.critic_hidden=np.zeros(s+(hidden,),np.float32)

    def add(self,**values):
        i=self.pos
        for key,value in values.items(): getattr(self,key)[i]=value
        self.pos+=1

    def sequences(self,length=32,gamma=.99,gae_lambda=.95):
        if self.pos!=self.steps: raise RuntimeError("rollout buffer is not full")
        adv,ret=compute_gae(self.rewards,self.values,self.next_values,self.terminated,self.truncated,gamma,gae_lambda)
        chunks=[]
        for env in range(self.envs):
            for start in range(0,self.steps,length):
                end=min(start+length,self.steps); n=end-start
                chunk={"observations":np.zeros((length,self.observations.shape[-1]),np.float32),
                       "maneuvers":np.zeros((length,3),np.float32),"fire":np.zeros(length,np.float32),
                       "old_log_probs":np.zeros(length,np.float32),"advantages":np.zeros(length,np.float32),
                       "returns":np.zeros(length,np.float32),"episode_starts":np.ones(length,np.float32),
                       "active":np.zeros(length,np.float32),"actor_hidden":self.actor_hidden[start,env],
                       "critic_hidden":self.critic_hidden[start,env]}
                for key,src in (("observations",self.observations),("maneuvers",self.maneuvers),("fire",self.fire),
                    ("old_log_probs",self.log_probs),("episode_starts",self.episode_starts)):
                    chunk[key][:n]=src[start:end,env]
                chunk["advantages"][:n]=adv[start:end,env]; chunk["returns"][:n]=ret[start:end,env]; chunk["active"][:n]=1
                chunks.append(chunk)
        valid=np.concatenate([c["advantages"][c["active"]>0] for c in chunks]); mean,std=valid.mean(),valid.std()+1e-8
        for c in chunks: c["advantages"]=(c["advantages"]-mean)/std*c["active"]
        return chunks


def recurrent_ppo_update(actor,critic,optimizer,chunks,device,clip_epsilon=.2,
                         update_epochs=10,minibatch_sequences=8,entropy_coef=.01,
                         value_coef=.5,max_grad_norm=.5,target_kl=.03):
    metrics=[]
    for _ in range(update_epochs):
        order=np.random.permutation(len(chunks)); epoch_kl=[]
        for start in range(0,len(chunks),minibatch_sequences):
            batch=[chunks[i] for i in order[start:start+minibatch_sequences]]
            t=lambda key:torch.as_tensor(np.stack([x[key] for x in batch]),dtype=torch.float32,device=device)
            obs,starts,active=t("observations"),t("episode_starts"),t("active")
            ah=t("actor_hidden").unsqueeze(0); ch=t("critic_hidden").unsqueeze(0)
            md,fd,_=actor.distributions(obs,ah,starts,active)
            new_lp=md.log_prob(t("maneuvers"))+fd.log_prob(t("fire")); old=t("old_log_probs")
            ratio=(new_lp-old).exp(); advantages=t("advantages"); mask=active>0
            policy=-torch.min(ratio*advantages,ratio.clamp(1-clip_epsilon,1+clip_epsilon)*advantages)[mask].mean()
            values,_=critic(obs,ch,starts,active); value_loss=.5*(values-t("returns")).square()[mask].mean()
            entropy=(md.entropy()+fd.entropy())[mask].mean(); loss=policy+value_coef*value_loss-entropy_coef*entropy
            if not torch.isfinite(loss): raise FloatingPointError("non-finite recurrent PPO loss")
            optimizer.zero_grad(); loss.backward(); grad=nn.utils.clip_grad_norm_(list(actor.parameters())+list(critic.parameters()),max_grad_norm); optimizer.step()
            kl=((ratio-1)-(new_lp-old))[mask].mean(); epoch_kl.append(float(kl.detach()))
            metrics.append(dict(policy_loss=float(policy.detach()),value_loss=float(value_loss.detach()),entropy=float(entropy.detach()),approximate_kl=float(kl.detach()),gradient_norm=float(grad)))
        if np.mean(epoch_kl)>target_kl: break
    return {k:float(np.mean([m[k] for m in metrics])) for k in metrics[0]}
