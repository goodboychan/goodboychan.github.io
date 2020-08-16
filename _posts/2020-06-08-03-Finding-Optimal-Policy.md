---
toc: true
layout: post
description: A summary of "Understanding deep reinforcement learning"
categories: [Reinforcement_Learning]
title: Finding Optimal Policy
image: 
---

## Optimal Value Function
The optimal state-value function $V^{\ast}(s)$ is the maximum state-value function over all policies:


$$ V^{\ast}(s) = \max_{\pi}V^{\pi}(s) $$


And optimal state action-value function $Q^{\ast}(s, a)$ is the maximum action-value function over all policies:


$$ Q^{\ast}(s, a) = \max_{\pi} Q^{\pi}(s,a)$$


The difference between them is that $Q^{\ast}(s, a)$ takes the inital action $a$ based on policy $\pi$, but $V^{\ast}(s)$ is not.

## Optimal Policy

Find the optimal policy $\pi^{\ast}$ which maximize the state-value at each state:
$$ \pi^{\ast}(s) = \argmax_{\pi}V^{\pi}(s) $$

For the optimal policy $\pi^*$, we have
- $V^{\pi^*} \gt V^{\pi}(s)$ for any policy $\pi$ and state $s$
- $Q^{\pi^*} \gt Q^{\pi}(s, a)$ for any policy $\pi$, any state $s$ and any action $a$

## Relation of Two Optimality Value Function

Find actions which maximize state-value function
$$V^*(s) = \max_{\pi}V{\pi}(s) = \max_{a}Q^*(s, a) $$
We can derive the bellman equations for optimal policy:
$$ \begin{aligned} Q^*(s,a) &= R(s, a) + \gamma E_{\pi^*}[ V^*(s_{t+1}) \vert s_t = s, a_t=a] \\ &= R(s,a) + \gamma E_{\pi^*}[\max_{a'}Q^*(s_{t+1}, a') \vert s_t=s, a_t=a] \\ V^*(s) &= \max_aR(s,a) + \gamma E_{\pi^*}[V^*(s_{t+1}) \vert S_t = s, a_t = a^*] \\ & \text{where } a^* = \argmax_a R(s, a)  \end{aligned}$$

## Bellman Optimality Equation for finite MDP

In discrete state space, we can replace the expectation term with summation:
$$ \begin{aligned} Q^*(s,a) &= R(s,a) + \gamma E_{\pi^*}[V^*(s_{t+1}) \vert s_t=s, a_t=a] \\ &= R(s,a) + \gamma \sum_{s' \in S} T_{ss'}^a V^*(s') \end{aligned} $$

$$ \begin{aligned} Q^*(s,a) &= R(s,a) + \gamma E_{\pi^*}[\max_{a'} Q^*(s_{t+1}, a') \vert s_t=s, a_t=a] \\ &= R(s,a) + \gamma \sum_{s' \in S} T_{ss'}^a \max_{a'}Q^*(s', a') \end{aligned} $$

$$ V^*(s) = \max_a R(s, a) + \gamma \sum_{s' \in S} T_{ss'}^a V^*(s') $$

## Finding Optimal Policy

Searching the every policy satisfying:
$$\pi^*(s) = \argmax_{\pi} V^{\pi}(s)$$

If we have finite action space and finite state space, then we should compare the number of $\vert A \vert^{\vert s \vert}$ (huge number). So we need Iterative approach to solve it:
- Value iteration
- Policy iteration
- Q-learning
- SARSA
