---
toc: true
layout: post
description: A summary of "Understanding deep reinforcement learning"
categories: [Reinforcement Learning]
title: Variants of Dynamic Programming
image: 
---

# Policy Improvement by Iterative Methods

## Asynchronous Dynamic Programming
Dynamic Programming mentioned before used synchronous backups which updates all staes at each iteration in parallel. (that means, next state value function can be calculated when the current state value function is ready.) **Asynchronous** DP updates each state in any order. This can significantly reduce computation, and it is convergent if all states continue to update.
- Variants
  - In-place Dynamic Programming
  - Prioritized sweeping
  - Real-time Dynamic Programming

## In-Place Dynamic Programming
- Synchronous value iteration stores two copies of value function
 1.  For all states $s$:
   
   $V_{new}(s) = \max_{a \in A} (R(s, a) + \gamma \sum_{s' \in S} P(s' \vert s, a) V_{old}(s'))$

  2. $V_{old}(s) = V_{new}(s)$

- In-place value iteration only use **one memory space** of value function

1. For all states $s$

$$ V(s) = \max_{a \in A}(R(s, a) + \gamma \sum_{s' \in S}P(s' \vert s, a) V(s')) $$

> Note: when the $V(s)$ is calculated, it will replace the $V(s')$ in next iteration

## Prioritized Sweeping

Prioritized Sweeping is the approach to update state with priority. To select states for update, find the state with the largest Bellman error:

$$ \argmax_{s \in S} \vert \max_{a \in A} \big(R(s, a) + \gamma \sum_{s' \in S} P(s' \vert s, a) V(s')\big) - V(s) \vert $$

- Update the state and Bellman error of affected states ($V_{new}(s) - V_{old}(s)$)
- This can be implemented by using a priority queue

## Sample Backups
In the previous post, Full-width backups are introduced. It is consider all states and actions. Compared to this, **Sample backups** use the samples of rewards and transitions instead of reward function and transition function. By sampling reward and transition, it can reduce the computation.

## Approximate DP

Value function can be approximated by using
- Function approximator $\hat{V}(s, w)$ with parameter $w$ (usually wa used neural network for training approximator)
- Applying DP to calculate $\hat{V}(\cdot, w)$

It is called **Fitted Value Iteration**, and follow the process
- Repeat until convergence,
  - Sample states $\tilde{S} \subset S$
  - For each state $s \in \tilde{S}$, estimate target value using Bellman optimality equation,
  
  $\tilde{V_k}(s) = \max_{a \in A} \big(R(s, a) + \gamma \sum_{s' \in S} P(s' \vert s, a) \hat{V}(s', w_k) \big)$

  - Train next value function $\hat{V}(\cdot, w_{k+1})$ with targets $\{ \langle s, \tilde{V_k}(s) \rangle \}$

