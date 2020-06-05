---
toc: true
layout: post
description: A summary of "Understanding deep reinforcement learning"
categories: [Reinforcement Learning]
title: Dynamic Programming for solving Bellman equation
---
# Dynamic Programming for solving Bellman equation

## Dynamic Programming for solving MRP

- Dynamic Programming
  - Initialize values $V_0(s)$ for all state $s$
  - For $k=1$ until convergence

  $$ V_k(s) = R(s) + \gamma \sum_{s_2 \in S} T_{s s_2} V_{k-1}(s_2) $$
    for all states $s \in S$

> Note: if $\vert V_{k+1} - V_{k} \vert \lt \epsilon$, then value function is converged.

Computational complexity is $O(n^2)$ where n is the number of states.

## Markov Decision Process
Markov Decision Process is MRP with actions, and it is a tuple of $(S, A, T, R, \gamma)$:
- $S$ is a set of Markov states ($s \in S$)
- $A$ is a set of actions ($a \in A$)
- $T$ is transition model for each action that specifies: