---
toc: true
layout: post
description: A summary of "Understanding deep reinforcement learning"
categories: [Reinforcement Learning]
title: Policy Iteration
image: 
---

# Policy Improvement by Iterative Methods

## Greedy Policy Improvement
- Starting from a initial policy $\pi_0$
- Iterate for $k$ until convergence
  - Evaluate the policy $\pi_k$ 
  
  $V^{\pi_k}(s) = E_{\pi_k}[r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots \vert s_t = s]$

  - Update the policy $\pi_k$ by choosing better action for each state:
  
  $\pi_{k+1}(s) = \argmax_{a \in A} Q^{\pi_k}(s, a)$

This process actually improves policy:

  - Iteration:
  
$Q^{\pi_k}(s, \pi_{k+1}(s)) = \max_{a \in A} Q^{\pi_k}(s, a) \\ \ge Q^{\pi_k}(s, \pi_k(a)) = V^{\pi_k}(s)$

  - By this iteration, we have:

$\begin{aligned} V^{\pi_k}(s) &\le Q^{\pi_k}(s, \pi_{k+1}(s)) \\
&= E_{\pi_{k+1}}[r_t + \gamma V^{\pi_k}(s_{t+1}) \vert s_t = s] \\
&\le E_{\pi_{k+1}}[r_t + \gamma Q^{\pi_k}(s_{t+1}, \pi_{k+1}(s_{t+1})) \vert s_t = s] \\ &\le E_{\pi_{k+1}}[r_t + \gamma r_{t+1} + \gamma^2 Q^{\pi_k}(s_{t+2}, \pi_{k+1}(s_{t+2})) \vert s_t = s] \\ &\le E_{\pi_{k+1}}[r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots \vert s_t = s] \\ &= V^{\pi_{k+1}}(s) \end{aligned}$

- When we have a convergent policy $\pi_c$, it satisfies the Bellman optimality condition:

$Q^{\pi_c}(s, \pi_c(s)) = \max_{a \in A} Q^{\pi_c}(s, a) = V^{\pi_c}(s)$

- Thus, $\pi_c$ is the optimal policy $\pi^{\ast}$.