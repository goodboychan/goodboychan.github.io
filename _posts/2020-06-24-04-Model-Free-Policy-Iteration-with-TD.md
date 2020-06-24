---
toc: true
layout: post
description: A summary of "Understanding Deep Reinforcement Learning"
categories: [Reinforcement Learning]
title: Model-free Policy Iteration with TD Methods
image: 
---

# Model-free control

## Recall Updating Action-Value Functions with TD(0)

$$ Q(s_t, a_t) \larr Q(s_t, a_t) + \alpha (r_t + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t)) $$

To utilize this formula (TD(0)), it is required to gather some informations ($s_t, a_t, r_t, s_{t+1}, a_{t+1}$). So TD(0) is called **SARSA** 

## Model-free Policy Iteration with TD method
- Starting from a policy $\pi$
- Iterate until convergence
  - **Policy evaluation** using TD policy evaluation of $Q^{\pi}$ for $\epsilon$-greedy policy
  - **Policy improvement** using $\epsilon$-greedy policy improvement

## On-Policy Control with SARSA
> Note: On-Policy means that target policy is same as behavior policy

1. Starting from $\epsilon$-greedy policy $\pi$ randomly at $t=0$ with initial state $s_0$. Then sample the initial action from policy ($a_0 \sim \pi(s_0)$), observe the next information $r_0, s_1$
2.  Repeat until convergence 
    1.  Take the action $a_{t+1} \sim \pi(s_{t+1})$
    2.  And observe the next state ($r_{t+1}, s_{t+2}$)
    3.  Update the value function 

    $$ Q(s_t, a_t) \larr Q(s_t, a_t) + \alpha_t (r_t + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t))$$

    4.  Improve the policy

    $$ \pi(s_t) = \begin{cases} \argmax_{a} Q(s_t, a_t) & \text{with probability of } 1 - \epsilon \\ \text{random} & else \end{cases}$$

## Convergence Theorem of SARSA
- SARSA for finite-state and finite-action MDPS