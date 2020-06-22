---
toc: true
layout: post
description: A summary of "Understanding Deep Reinforcement Learning"
categories: [Reinforcement Learning]
title: Model-free Control
image: 
---

# Model-free Control

## Recall Optimal Policy
- Find the optimal policy $\pi^{*}$ which maximize the state-value at each state:
$$ \pi^{*}(s) = \arg \max_{\pi} V^{\pi}(s)$$

- For the optimal policy $\pi^{*}$, we have,
  - $V^{\pi^{*}} \geq V^{\pi}(s)$ for any policy $\pi$ and any state $s$
  - $Q^{\pi^{*}}(s, a) \geq Q^{\pi}(s, a)$ for any policy $\pi$, any state $s$ and any action $a$.

- Iterative approach:
  - Value iteration
  - Policy iteration

## Model-Free Control
**Control** is to find the optimal policy for MDP model. For most of control problems, it has some condtion:
  - MDP model is unknown, but experience can be sampled
  - MDP model is known, but it is too big to use, except by samples.

Examples are
  - Autonomous Robot
  - Game Play
  - Portfolio Management
  - Protein Folding

## On and Off-Policy Learning
**On-Policy** learning is sort of learning approach that learns from direct experiences following behavior policy. And it learns to evaluate a policy $\pi$ from experience sampled from $\pi$. 

On the other hand, **Off-Policy** learning is that learn from indirect experiences such as human experts or other agents. And it learns to evaluate a policy $\pi$ from experience sampled from other policies. Usually, agent learns to follow optimal policy using exploratory policy, or learns multiple policies while following one policy

## Importance Sampling
Usually, Off-policy learning samples the experience from different distribution. In that case, importance Sampling can estimate the expection from a different distribution.
$$ \begin{aligned} \mathbb{E}_{X \sim P}[f(X)] &= \sum P(X) f(X) \\ &= \sum Q(X) \frac{P(X)}{Q(X)} f(X) \\ &= \mathbb{E}_{X \sim Q} \big[ \frac{P(X)}{Q(X)} f(X) \big] \end{aligned} $$ 

## Model-free Generalized Policy Improvement (GPI)
Given a Policy $\pi$, estimate the state action value function $Q^{\pi}(S, a)$. Using this information, update the $\pi$ with $\pi'$:

$$ \pi'(s) = \arg \max_{a \in A} Q^{\pi}(s, a)$$