---
toc: true
layout: post
description: A summary of "Understanding Deep Reinforcement Learning"
categories: [Reinforcement Learning]
title: Monte Carlo Policy Evaluation
image: 
---

# Monte Carlo Methods and Temporal Difference Learning in Policy Evaluation

## Monte Carlo Policy Evaluation
At first, generate and store episode information under policy $\pi$: \
in $i$-th episode:

$$ s_{i, 1}, a_{i, 1}, r_{i,1}, s_{i, 2}, a_{i, 2}, r_{i, 2}, \dots, s_{i, T} $$

Based on this information, Calculate total expected return $G_t$ for each episode: 
$$ G_t = r_t + \gamma * r_{t+1} + \gamma^2 * r_{t+2} + \cdots $$

Monte Carlo (MC) Policy Evaluation estimates expectation ($V^{\pi}(s) = E_{\pi}[G_t \vert s_t = s]$) by iteration using
  - Empirical average: given episodes sampled from policy $\pi$
  - Importance Sampling: reweighted empirical average \
  (for example, apply more weights on latest episode information, or apply more weights on important episode information, etc...)

MC Policy Evaluation does not require transition dynamics ($T$) and reward function ($r$) in MDP. And total expected return can be calculated based on collected return, so Bootstrapping method (calculate future reward with current information) is not required. Also, It does not assume that state follows Markov Property.
But one constraint is that it can only be applied to episodic MDPs
  - Averaging over returns from a complete episode
  - Requires each episode to terminate \
  $\rarr$ Cannot fully online learning

Detailed process:
- Generate episodes under policy $\pi$
- Calculate $G$ from each episode
- Calculate MC empirical average return from episodes which visit the state $s$ to approximate value function: 
  $$V^{\pi}(s) = E_{\pi}[G_t \vert s_t = s]$$
- If we have more episodes, update estimate of $V^{\pi}$ with these episodes.

## First-Visit Monte Carlo (MC) Policy Evaluation
When we update value function $V^{\pi}(s)$, we select the **first visit** state value.

1. Initialize $N(s) = 0 = G(s),\forall_s \in S$ \
($N(s)$: number of visitation, $G(s)$: total expected return in state $s$)
2. Loop
   1. Generate Episode $i = s_{i,1}, a_{i,1}, r_{i,1}, s_{i,2}, a_{i,2}, r_{i,2}, \dots, s_{i,T_i}$
   2. Define return of $i$-th episode from time step $t$:\
   $G_{i,t} = r_{i,t} + \gamma r_{i, t+1}, \gamma^2r_{i,t+2} + \cdots + \gamma^{T_i-1} r_{i, T_i}$
   3. For each state $s$ visited in $i$-th episode \
      -For **first** time $t$ that state $s$ is visited in $i$-th episode,
         - Increment counter of total first visits: $N(s) = N(s) + 1$
         - Increment total return: $G(s) = G(s) + G_{i, t}$
         - Update estimate: $V^{\pi}(s) = \dfrac{G(s)}{N(s)}$ 


