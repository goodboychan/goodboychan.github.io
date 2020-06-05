---
toc: true
layout: post
description: A summary of "Understanding deep reinforcement learning"
categories: [Reinforcement Learning]
title: Bellman equations and Markov decision process
---
# Bellman equations and Markov decision process

## Recall: Markov Reward Process

Markov Reward Process is Markov Chain with rewards. it is a tuple of $(S, T, R, \gamma)$

- $S$ is a set of Markov states $(s \in S)$
- $T$ is transition model that specifies $T(t, s_1, s_2) = T_{s_1s_2} \doteq P(s_{t+1} = s_2 | s_t = s_1)$
- $R$ is a reward function $R(s_t = s) = E[r_t | s_t = s]$
- $\gamma$ is Discount factor $\gamma \in [0, 1]$

> Note: 
> - There is no actions
> - If $S$ is finite, then $R$ can be represented by a vector

