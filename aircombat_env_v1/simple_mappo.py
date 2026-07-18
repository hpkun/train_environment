"""Minimal feed-forward MAPPO baseline for SimpleTAMCombatEnv."""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleMAPPOAdapter:
    def __init__(self,env):
        self.env=env;self.agent_ids=list(env.controlled_ids);self.num_agents=len(self.agent_ids)
        self.obs_dim=int(env.observation_space[self.agent_ids[0]].shape[0]);self.state_dim=self.obs_dim*self.num_agents;self.action_dim=3
    def _adapt(self,observations):
        actor_obs=np.stack([np.asarray(observations[aid],np.float32) for aid in self.agent_ids])
        return actor_obs,np.concatenate(actor_obs).astype(np.float32)
    def _active_mask(self):return np.asarray([float(self.env.by_id[aid].alive) for aid in self.agent_ids],np.float32)
    def reset(self,**kwargs):
        observations,info=self.env.reset(**kwargs);actor_obs,state=self._adapt(observations);return actor_obs,state,info
    def step(self,actions):
        actions=np.asarray(actions,np.float32)
        if actions.shape!=(self.num_agents,self.action_dim):raise ValueError(f"actions must have shape {(self.num_agents,self.action_dim)}")
        obs,rewards,terminated,truncated,info=self.env.step({aid:actions[i] for i,aid in enumerate(self.agent_ids)})
        actor_obs,state=self._adapt(obs);reward_array=np.asarray([rewards[aid] for aid in self.agent_ids],np.float32)
        return actor_obs,state,reward_array,bool(terminated or truncated),self._active_mask(),info

class SharedMAPPOActorCritic(nn.Module):
    def __init__(self,obs_dim,state_dim,action_dim=3):
        super().__init__();self.obs_dim=int(obs_dim);self.state_dim=int(state_dim);self.action_dim=int(action_dim)
        self.actor=nn.Sequential(nn.Linear(self.obs_dim,128),nn.Tanh(),nn.Linear(128,128),nn.Tanh(),nn.Linear(128,self.action_dim))
        self.log_std=nn.Parameter(torch.full((self.action_dim,),float(np.log(.3))))
        self.critic=nn.Sequential(nn.Linear(self.state_dim,128),nn.Tanh(),nn.Linear(128,128),nn.Tanh(),nn.Linear(128,1))
    def _distribution(self,actor_obs):
        mean=self.actor(actor_obs);std=self.log_std.exp().clamp(1e-4,5.).expand_as(mean);return torch.distributions.Normal(mean,std)
    @staticmethod
    def _log_prob(dist,z,action):return (dist.log_prob(z)-torch.log(1-action.square()+1e-6)).sum(-1)
    def act(self,actor_obs,critic_state,deterministic=False):
        dist=self._distribution(actor_obs);z=dist.mean if deterministic else dist.rsample();action=torch.tanh(z)
        log_prob=self._log_prob(dist,z,action);entropy=dist.entropy().sum(-1);value=self.value(critic_state)
        return action,log_prob,value,entropy
    def evaluate_actions(self,actor_obs,critic_state,actions):
        action=actions.clamp(-1+1e-6,1-1e-6);z=torch.atanh(action);dist=self._distribution(actor_obs)
        return self._log_prob(dist,z,action),dist.entropy().sum(-1),self.value(critic_state)
    def value(self,critic_state):
        if critic_state.ndim==1:critic_state=critic_state.unsqueeze(0)
        return self.critic(critic_state).squeeze(-1)

class RolloutBuffer:
    def __init__(self,max_len,num_agents,obs_dim,state_dim,action_dim=3):
        self.max_len=int(max_len);self.num_agents=int(num_agents);self.obs_dim=int(obs_dim);self.state_dim=int(state_dim);self.action_dim=int(action_dim);self.clear()
    def clear(self):
        self.actor_obs=[];self.critic_state=[];self.actions=[];self.log_probs=[];self.rewards=[];self.team_dones=[];self.values=[];self.active_masks=[]
    def __len__(self):return len(self.rewards)
    @property
    def length(self):return len(self)
    def store(self,actor_obs,critic_state,actions,log_probs,rewards,team_done,value,active_mask):
        if len(self)>=self.max_len:raise IndexError("rollout buffer is full")
        self.actor_obs.append(np.asarray(actor_obs,np.float32));self.critic_state.append(np.asarray(critic_state,np.float32))
        self.actions.append(np.asarray(actions,np.float32));self.log_probs.append(np.asarray(log_probs,np.float32));self.rewards.append(np.asarray(rewards,np.float32))
        self.team_dones.append(float(team_done));self.values.append(float(value));self.active_masks.append(np.asarray(active_mask,np.float32))
    def to_tensors(self,device="cpu"):
        arrays={"actor_obs":np.asarray(self.actor_obs,np.float32),"critic_state":np.asarray(self.critic_state,np.float32),
          "actions":np.asarray(self.actions,np.float32),"log_probs":np.asarray(self.log_probs,np.float32),"rewards":np.asarray(self.rewards,np.float32),
          "team_dones":np.asarray(self.team_dones,np.float32),"values":np.asarray(self.values,np.float32),"active_masks":np.asarray(self.active_masks,np.float32)}
        return {key:torch.as_tensor(value,device=device) for key,value in arrays.items()}

class MAPPOTrainer:
    def __init__(self,model,actor_lr=3e-4,critic_lr=3e-4,gamma=.99,gae_lambda=.95,clip_param=.2,
                 value_coef=.5,entropy_coef=.01,max_grad_norm=10.,ppo_epochs=4):
        self.model=model;self.actor_optimizer=torch.optim.Adam(list(model.actor.parameters())+[model.log_std],lr=actor_lr)
        self.critic_optimizer=torch.optim.Adam(model.critic.parameters(),lr=critic_lr)
        self.gamma=float(gamma);self.gae_lambda=float(gae_lambda);self.clip_param=float(clip_param);self.value_coef=float(value_coef)
        self.entropy_coef=float(entropy_coef);self.max_grad_norm=float(max_grad_norm);self.ppo_epochs=int(ppo_epochs)
    def compute_gae(self,data,next_value):
        rewards=data["rewards"];masks=data["active_masks"];valid=masks.sum(-1).clamp(min=1.)
        team_rewards=(rewards*masks).sum(-1)/valid;values=data["values"];dones=data["team_dones"]
        advantages=torch.zeros_like(values);gae=torch.zeros((),device=values.device,dtype=values.dtype);next_v=torch.as_tensor(next_value,device=values.device,dtype=values.dtype).reshape(())
        for t in reversed(range(values.shape[0])):
            continuation=1.-dones[t];delta=team_rewards[t]+self.gamma*continuation*next_v-values[t]
            gae=delta+self.gamma*self.gae_lambda*continuation*gae;advantages[t]=gae;next_v=values[t]
        return advantages,advantages+values
    def update(self,buffer,next_value):
        device=next(self.model.parameters()).device;data=buffer.to_tensors(device);advantages,returns=self.compute_gae(data,next_value)
        if advantages.numel()>1:advantages=(advantages-advantages.mean())/(advantages.std(unbiased=False)+1e-8)
        T,N=data["rewards"].shape;valid=data["active_masks"];valid_count=valid.sum().clamp(min=1.)
        actor_losses=[];critic_losses=[];entropies=[];kls=[];clips=[];gradients=[]
        for _ in range(self.ppo_epochs):
            log_prob,entropy,new_values=self.model.evaluate_actions(data["actor_obs"].reshape(T*N,-1),data["critic_state"],data["actions"].reshape(T*N,-1))
            log_prob=log_prob.reshape(T,N);entropy=entropy.reshape(T,N);ratio=torch.exp(log_prob-data["log_probs"])
            surr1=ratio*advantages[:,None];surr2=ratio.clamp(1-self.clip_param,1+self.clip_param)*advantages[:,None]
            actor_loss=-(torch.minimum(surr1,surr2)*valid).sum()/valid_count;entropy_mean=(entropy*valid).sum()/valid_count
            critic_loss=F.mse_loss(new_values,returns)
            self.actor_optimizer.zero_grad();(actor_loss-self.entropy_coef*entropy_mean).backward()
            actor_grad=torch.nn.utils.clip_grad_norm_(list(self.model.actor.parameters())+[self.model.log_std],self.max_grad_norm);self.actor_optimizer.step()
            self.critic_optimizer.zero_grad();(self.value_coef*critic_loss).backward()
            critic_grad=torch.nn.utils.clip_grad_norm_(self.model.critic.parameters(),self.max_grad_norm);self.critic_optimizer.step()
            with torch.no_grad():
                approx_kl=((data["log_probs"]-log_prob)*valid).sum()/valid_count
                clip_fraction=(((ratio-1.).abs()>self.clip_param).float()*valid).sum()/valid_count
            actor_losses.append(actor_loss.item());critic_losses.append(critic_loss.item());entropies.append(entropy_mean.item())
            kls.append(approx_kl.item());clips.append(clip_fraction.item());gradients.append(float(max(actor_grad,critic_grad)))
        result={"actor_loss":np.mean(actor_losses),"critic_loss":np.mean(critic_losses),"entropy":np.mean(entropies),
          "approx_kl":np.mean(kls),"clip_fraction":np.mean(clips),"gradient_norm":np.mean(gradients)}
        if not np.isfinite(list(result.values())).all():raise FloatingPointError("non-finite MAPPO update")
        return {key:float(value) for key,value in result.items()}
