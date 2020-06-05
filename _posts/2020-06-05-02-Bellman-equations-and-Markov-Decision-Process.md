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
- $T$ is transition model (or dynamics) that specifies $T(t, s_1, s_2) = T_{s_1s_2} \doteq P(s_{t+1} = s_2 \vert s_t = s_1)$
- $R$ is a reward function $R(s_t = s) = E[r_t \vert s_t = s]$
- $\gamma$ is Discount factor $\gamma \in [0, 1]$

> Note: 
> - There is no actions
> - If $S$ is finite, then $R$ can be represented by a vector
> - By Markov property, the transition model is independent of $t$ so that it can be expresed by $T(t, s_1, s_2) = T(s_1, s_2) = P(s_{t+1} = s_2 \vert s_t = s_1)$
> - Also same as $R(s_t = s) = R(s)$

## Markov Reward Process for Finite State
If $S$ is finite, $S$ can be represented by a vector: 
$$S=\{s_1, s_2, \dots, s_N \}$$
And the transition model $T$ can be represented by a matrix:
$$ T = \begin{bmatrix} P(s_1 \vert s_1) & P(s_2 \vert s_1) & \cdots & P(s_N \vert s_1) \\ P(s_1 \vert s_2) & P(s_2 \vert s_2) & \cdots & P(s_N \vert s_2) \\ \vdots & \vdots & \ddots & \vdots \\  P(s_1 \vert s_N) & P(s_2 \vert s_N) & \cdots & P(s_N \vert s_N)\end{bmatrix}  $$
Also, the reward function $R$ can be represented by a vector:
$$ R = \big( R(s_1), R(s_2, \cdots, R(s_N) \big)^T $$

## Computing return from rewards
$R$ is a reward function
$$ R(s) = E[r_t | s_t = s]$$
We can calculate total discounted rewards from $t$ be the return $G_t$:

$$\begin{aligned} G_t &= r_t + \gamma * r_{t+1} + \gamma^2 * r_{t+2} + \cdots \\
&= \sum_{t=0}^{\infty} \gamma^t r_t \\
&= r_t + \gamma G_{t+1}   \end{aligned}$$

- Time of horizon: number of time steps in each episode
  - Finite ($t \leq T$) or infinite ($t \rarr \infty$)
  - If finite, MRP is called **finite** MRP

## State-value function for MRP
- State-value function $V$: expected discounted sum of future rewards from initial state $s$
$$ \begin{aligned} V(s_t = s) &= E[G_t \vert s_t = s] \\
&= E[r_t + \gamma r_{t+1} + \gamma^2r_{t+2} + \gamma^3 r_{t+3} + \cdots \vert s_t = s] \end{aligned} $$
- By Markov property, $V(s_t = s)$ is independent of the initial step $t$ so that we can use simpler notation $V(s)$ instead of $V(s_t = s)$.

## Bellman Equation for MRP
The value function can be decomposed by:
- Immediate reward ($r_t$)
- Discounterd value of subsequent states ($\gamma G_{t+1}$)

The Bellman equation for MRP is:
$$\begin{aligned} V(s) &= E[G_t \vert s_t = s] \\
&= E[r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \gamma^3 r_{t+3} + \cdots \vert s_t = s] \\
&= E[r_t + \gamma G_{t+1} \vert s_t = s] \\
&= E[r_t | s_t = s] + \gamma E[ E[G_{t+1} \vert s_{t+1}] \vert s_t = s] \\
&= R(s) + \gamma E[V(s_{t+1}) \vert s_t = s]  \end{aligned}$$