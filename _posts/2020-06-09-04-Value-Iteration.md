---
toc: true
layout: post
description: A summary of "Understanding deep reinforcement learning"
categories: [Reinforcement_Learning]
title: Value Iteration
image: 
---

# Policy Improvement by Iterative Methods

## Value Iteration
- Initialize state-value for all states
- Iterate for $k$ until convergence
  - Update the state-value for all states

$V_{k+1} (s) = \max_{a \in A}(R(s, a) + \gamma \sum_{s' \in S} P(s' \vert s, a) V_k(s'))$

We can re-define this term with expectation,

$V_{k+1} (s) = \max_{a \in A}(R(s, a) + \gamma E_k[Q_k(s_{t+1}, a) \vert s_t = s])$

- Iterative application of Bellman  optimality backup
- Using tensor notation ($E_k[Q_k(s_{t+1}, a) \vert s_t = s] \rarr \Tau^a V_k$):

$$ V_{k+1} = \max_{a \in A}(R(a) + \gamma \Tau^a V_k) $$

- This is called synchronous backup. The reason we called "backup" is that, we always hold state value in current time step and next time step. And when the state value in current time step is prepared, then we can calculate the next time step state value. So it is synchronous.

- Value iteration finds no explicit policy at each iteration, even the optimal policy.
- To find the optimal policy, we should save and update the best action for each iteration.
- The complexity is $O(\vert A \vert \vert S \vert^2)$ per iteration.
($\vert S \vert$ for current state, and $\vert S \vert$ for next state)