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

## Model-Free Policy Iteration
- Initialize policy $\pi$
- Repeat until convergence
  - Policy evaluation: estimate $Q^{\pi}$
  - Policy improvement: Generate $\pi' \geq \pi$

## Policy Evaluation with Exploration

What if $\pi$ is deterministic, how can we compute $Q^{\pi}(s, a)$ for $a \neq \pi(s)$? To do this, it requires the data of $(s, a)$ pairs for $a \neq \pi(s)$, and it is called **exploration**. And it can get through:
- Get all $(s, a)$ pairs for $a \neq \pi(s)$
- Or get some $(s, a)$ pairs for $a \neq \pi(s)$ enough to ensure that resulting estimate $Q^{\pi}$ improve current policy

So how can we sure that we collect enough $(s, a)$ pairs?

## $\epsilon$-greedy Exploration
$\epsilon$-greedy exploration is the simple idea for ensuring continual exploration. Using this strategy, All actions are tried with non-zero probability
At first, Let $m = \vert A \vert$ be the number of actions. Then an $\epsilon$-greedy policy w.r.t $Q^{\pi}(s, a)$ is

$$\pi(a \vert s)= \begin{cases} \frac{\epsilon}{m} + 1 - \epsilon & \text{if } a^{*} = \arg \max_{a \in A} Q(s, a) \\ \dfrac{\epsilon}{m} & otherwise \end{cases} $$

## $\epsilon$-greedy Policy Improvement
- Theorem : Given policy $\pi$, the $\epsilon$-greedy policy $\pi'$ with respect to $Q_{\pi}$ is an improved policy of $\pi$

$$ V_{\pi'}(s) \geq V_{\pi}(s) $$

- Proof:

$$ \begin{aligned} Q_{\pi}(s, \pi'(s)) &= \sum_{a \in A} \pi'(a \vert s) Q_{\pi}(s, a) \\ &= \frac{\epsilon}{m} \sum_{a \in A} Q_{\pi}(s, a) + (1 - \epsilon) \max_{a \in A} Q_{\pi}(s, a) \\ & \geq \frac{\epsilon}{m} \sum_{a \in A} Q_{\pi}(s, a) +  (1 - \epsilon) \sum_{a \in A} \frac{\pi(a \vert s) - \frac{\epsilon}{m}}{1 - \epsilon} Q_{\pi}(s, a) \\ &= \sum_{a \in A} \pi(a \vert s) Q_{\pi}(s, a) \\ &= V_{\pi}(s)  \end{aligned} $$

As a result, we can get $Q_{\pi}(s, \pi'(s)) \geq V_{\pi}(s)$. It means that following policy $\pi$, if we choose an action from $\pi'$, our value function will be improved, and derive it like this,
$$ V_{\pi'}(s) \geq V_{\pi}(s) $$

## $\epsilon$-greedy Policy Improvement
In views of expectation, we can also prove it like this,

$$ \begin{aligned} V_{\pi}(s) &\leq Q_{\pi}(s, \pi'(s)) \\
&= E[R_t + \gamma V_{\pi}(s_{t+1}) \vert s_t = s, a_t = \pi'(s)] \\ &= E_{\pi'}[R_t + \gamma V_{\pi}(s_{t+1}) \vert s_t = s] \\ &\leq E_{\pi'}[R_t + \gamma Q_{\pi}(s_{t+1}, \pi'(s_{t+1})) \vert s_t = s] \\ &= E_{\pi'}[R_t + \gamma R_{\pi'}[R_{t+1} + \gamma V_{\pi}(s_{t+2})] \vert s_t = s] \\ &= E_{\pi'}[R_t + \gamma R_{t+1} + \gamma^2 V_{\pi}(s_{t+2}) \vert s_t = s] \\ &\leq E_{\pi'}[R_t + \gamma R_{t+1} + \gamma^2 Q_{\pi}(s_{t+2}, \pi'(s_{t+2})) \vert s_t = s] \\ &\qquad \qquad \qquad \qquad \vdots  \\ & \leq E_{\pi'}[R_t + \gamma R_{t+1} + \gamma^2 R_{t+2} + \gamma^3 R_{t+3} + \dots \vert s_t = s] \\ &= V_{\pi'}(s) \end{aligned} $$

> Note: $E[E[X]] = E[X]$

## Greedy in the Limit of Infinite Exploration (GLIE)

If learning policy $\pi$ satisfy these conditions:
- If a state is visited infinitely often, then every action in that state is chosen infinitely often (with probability 1)

$$ \lim_{i \rarr \infty} N_i (s, a) \rarr \infty $$

- As $t \rarr \infty$, the learning policy is greedy with respect to the learned $Q^{\pi}$ function with probability 1:
  
$$ \lim_{i \rarr \infty} \pi (a \vert s) \rarr \arg \max_{a} Q (s, a) \quad\text{with probability 1} $$

We can call this policy, **GLIE** (Greedy in the Limits of Infinite Exploration)

Bring this idea in $\epsilon$-greedy exploration, if $\epsilon_i$ is reduced to 0, this strategy is GLIE:
$$ \epsilon_i = \frac{1}{i} $$