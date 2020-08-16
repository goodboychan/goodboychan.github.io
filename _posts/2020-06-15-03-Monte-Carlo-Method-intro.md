---
toc: true
layout: post
description: A summary of "Understanding deep reinforcement learning"
categories: [Reinforcement_Learning]
title: Monte Carlo method - Intro
image: 
---

# Monte Carlo Methods and Temporal Difference Learning in Policy Evaluation

## Recall Policy Evaluation
Given a MDP and a policy $\pi$, how can we measure the goodness of this policy? To do this, we defined state value function ($V^{\pi}(s)$) and action value function ($Q^{\pi}(s, a)$), and calculated with Bellman equation.

$$ \begin{aligned} Q^{\pi}(s, a) &= R(s, a) + \gamma E_{\pi}[Q^{\pi}(s_{t+1}, a_{t+1}) \vert s_t =s, a_t = a] \\ &= R(s, a) + \gamma \sum_{s' \in S} P(s' \vert s, a) V^{\pi}(s') \end{aligned}$$ 

$$ \begin{aligned} V^{\pi}(s) &= R^{\pi}(s) + \gamma E[V^{\pi}(s_{t+1}) \vert s_t =s, a_t = \pi(a \vert s)] \\ &= R^{\pi}(s) + \gamma \sum_{s' \in S} P(s' \vert s, a) V^{\pi}(s') \end{aligned}$$ 

## Recall Dynamic Programming for Policy Evaluation

- Iterative Policy Evaluation
  - Initialize values $V_0(s) = 0$ for all states $s$
  - Repeat until convergence, \
  For each state $s \in S$,
  $$ V_{k}^{\pi}(s) = R(s) + \gamma \sum_{s' \in S} P(s' \vert s, \pi(a)) V_{k-1}^{\pi}(s') $$

- This is called a Bellman (synchronous) backup for policy $\pi$
- This recursive algorithm converges when $\gamma < 1$ or $T_{ss'}^{\pi} < 1$ for all $s, s' \in S$

## Dynamic Programming for Policy Evaluation

DP for policy Evaluation requires MDP model ($S, A, T, R, \gamma$) which contains transition dynamics $T$ and reward model $R$.

$$ V^{\pi}(s) \approx E_{\pi}[R_t + \gamma V_{k-1}^{\pi} \vert s_t = s]$$

- Bootstraps future return using value estimate.\
(meaning that to estimate future return, current return is used)
- Requires Markov assumption: bootstrapping regardless of history

But what if we don't know transition dynamics ($T$) and/or reward model ($R$). For example in robotic control or self-driving cars, we cannot solve the exact dynamics of that environment. In that case, we can approximate the dynamics (or models) from **Experience**. But how?

## Policy Evaluation without a model ($\approx \text{model-free}$)

Experience is sort of given data acquired by interacting in the environment. And by using this, we can efficiently compute a good estimate of a policy $\pi$.

