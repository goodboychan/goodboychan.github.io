---
toc: true
layout: post
description: A summary of "Understanding deep reinforcement learning"
categories: [Reinforcement_Learning]
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
Markov Decision Process is MRP with **actions**, and it is a tuple of $(S, A, T, R, \gamma)$:
- $S$ is a set of Markov states ($s \in S$)
- $A$ is a set of actions ($a \in A$)
- $T$ is transition model for each action that specifies:

$$ T(s_1, s_2, a) = T_{s_1s_2}^a \doteq P(s_{t+1} = s_2 \vert s_t = s_1, a_t = a) $$

- $R$ is a reward function ($R(s, a) = E[r_t \vert s_t = s, a_t = a]$)
- $\gamma$ is Discount factor ($\gamma \in [0, 1]$)

## Policy
Policy $\pi$ determines how the agent chooses actions. It is a function from states to actions ($\pi : S \rarr A$)
- Deterministic policy : $\pi(s) = a$
- Stochastic policy : $\pi(a \vert s) = P(a_t = a \vert s_t = s)$

> Note: If a policy $\pi$ is given for MDP (i.e. action is fixed for each state), MDP is just MRP with specific model & reward function:

$$ T^{\pi}(s_1, s_2) = P(s_{t+1} = s_2 \vert s_t = s_1, a_t = \pi(a \vert s)) \\ R^{\pi}(s) = E[r_t \vert s_t = s, a_t = \pi(a \vert s)] $$

## State-value function for MDP
State-value function $V^{\pi}$ is the expected discounted sum of future rewards under a particular policy $\pi$ from initial state s

$$ V^{\pi}(s) = E_{\pi}[r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \gamma^3 r_{t+3} + \cdots \vert s_t = s, a_t = \pi(a \vert s)] $$

Bellman equation for $V^{\pi}$:

$$ V^{\pi}(s) = R^{\pi}(s) + \gamma E[V{\pi}(s_{t+1}) | s_t = s, a_t = \pi(a \vert s)]$$

## Action-value function for MDP
Action-value function $Q^{\pi}$ is the expected discounted sum of future rewards tarting from state $s$, taking action $a$, and then following a particular policy $\pi$,

$$ \begin{aligned}Q^{\pi}(s, a) &= E_{\pi}[r_t + \gamma r_{t+1}  + \gamma^2 r_{t+2} + \gamma^3 r_{t+3} + \cdots \vert s_t = s, a_t = a] \\ &= E[r_t \vert s_t = s, a_t = a] + \gamma E_{\pi}[G_{t+1} \vert s_t = s, a_t = a] \end{aligned}$$

Bellman equation for $Q^{\pi}$:

$$Q^{\pi}(s, a) = R(s, a) + \gamma E_{\pi}[Q^{\pi}(s_{t+1}, a_{t+1}) \vert s_t = s, a_t = a] $$

## Relation of Two value Functions
Bellman equations for policy $\pi$:

$$ \begin{aligned}Q^{\pi}(s, a) &=  R(s, a) + \gamma E_{\pi}[V^{\pi}(s_{t+1} \vert s_t = s, a_t = a)] \\ &= R(s, a) + \gamma E_{\pi}[Q^{\pi}(s_{t+1}, \pi(a \vert s_{t+1})) \vert s_t = s, a_t = a] \\ \\ V^{\pi}(s) &= R(s, \pi(a \vert s)) + \gamma E_{\pi}[V{\pi}(s_{t+1}) | s_t = s] \end{aligned}$$

In discrete MDP, Bellman equations for policy $\pi$ becomes,

$$ \begin{aligned} Q^{\pi}(s, a) &= R(s, a) + \gamma \sum_{s' \in S} P(s' \vert s, a)V^{\pi}(s') \\ V^{\pi}(s) &= R^{\pi}(s) + \gamma \sum_{s' \in S} P(s' \vert s, a) V^{\pi}(s') \\ &= \sum_{a \in A} \pi(a \vert s) Q^{\pi}(s, a) \end{aligned} $$ 