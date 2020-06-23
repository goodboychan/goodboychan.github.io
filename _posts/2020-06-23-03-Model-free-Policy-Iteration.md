---
toc: true
layout: post
description: A summary of "Understanding Deep Reinforcement Learning"
categories: [Reinforcement Learning]
title: Model-free Policy Iteration with Monte Carlo Methods
image: 
---

# Model-free control

## Monte-Carlo Policy Iteration
- Starting from a policy $\pi$

- Iterate until convergence:
  - Policy evaluation using Monte-Carlo policy evaluation of $V^{\pi}$ for each policy $\pi$
  - Policy improvement using $\epsilon$-greedy policy improvement

## Monte Carlo Online Control

1. Initialize $Q(s, a) = 0, N(s, a) = 0, \forall (s, a)$ \
Set $\epsilon=1, k=1, \pi=\epsilon-\text{greedy}(Q)$
2. Repeat until convergence,
   1. Sample episode ($s_{k, 1}, a_{k, 1}, r_{k,1}, s_{k, 2}, \dots, s_{k, T}$) given $\pi_k$ \
   $G_{k, t} =r_{k, t} + \gamma r_{k, t+1} + \gamma^2 r_{k, t+2} + \cdots + \gamma^{T_i - 1} r_{k, T_i}$ for $t=1, \dots,T$
   2. If we update the value with first-visit, when we first visit to $(s, a)$ in episode $k$, then,\
   $N(s,a) \mathrel{+{=}} 1$\
   $Q(s_t, a_t) = Q(s_t, a_t) + \frac{1}{N(s, a)}(G_{k, t} - Q(s_t, a_t))$ \
   $k = k + 1$ \
   $\pi_{k+1} = \epsilon-\text{greedy}(Q) (\approx \pi_k)$

## GLIE Monte-Carlo Control
- Theorem: (GLIE) Monte-Carlo control converges to the optimal state-actino value function $Q^{*}(s, a)$

## Importance Sampling for Off-policy Monte-Carlo Control
Recall that, importance sampline is
$$ \mathbb{E}_{X \sim}