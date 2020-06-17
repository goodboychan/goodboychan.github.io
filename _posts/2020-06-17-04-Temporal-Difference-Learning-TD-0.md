---
toc: true
layout: post
description: A summary of "Understanding Deep Reinforcement Learning"
categories: [Reinforcement Learning]
title: Temporal Difference Learning TD(0)
classes: wide
image: 
---

# Temporal Difference Learning

## Temporal Difference Learning for Estimating Value function

We want to estimate $V^{\pi}(s) = E_{\pi}[G_t \vert s_t = s]$ given episodes generated under policy $\pi$. If we know MDP models of environment (that is, we know the transition probability of eash state), we can repeat the Bellman operator $B^{\pi}$ to estimate:
$$ B^{\pi}V(s) = R(s, \pi(s)) + \gamma \sum_{s' \in S}p(s' \vert s, \pi(s)) V(s') $$

In incremental every-visit MC, update the estimate using one sample of return for the current $i$-th episode:

$$ V^{\pi}(s) = V^{\pi}(s) + \alpha \big(G_{i, t} - V^{\pi}(s)\big)$$

What if we use the following recursive formula:\
(replaced $G_{i, t} \rarr r_t + \gamma V^{\pi}(s_{t+1})$)
$$ V^{\pi}(s) = V^{\pi}(s) + \alpha \big(r_t + \gamma V^{\pi}(s_{t+1}) - V^{\pi}(s)\big)$$

## Temporal Difference TD(0) Learning

- Generate episodes $s_1, a_1, r_1, s_2, a_2, r_2, \cdots$ under policy $\pi$ where the actions are sampled from $\pi$ to make tuples ($s, a, r, s'$)
- TD(0) learning: update value by estimated value

$$ V^{\pi}(s_t) = V^{\pi}(s_t) + \alpha ( \underbrace{[r_t + \gamma V^{\pi}(S_{t+1})]}_{\text{TD target}} - V^{\pi}(s_t)) $$

In this formula, we can define the error of TD(0) as

$$ \delta_t = r_t + \gamma V^{\pi}(S_{t+1}) - V^{\pi}(s_t) $$

As you can see, value function can be calculated from current value and next value. So TD(0) can immediately update value estimate using ($s, a, r, s'$) tuple without waiting for the termination of each episode.

Detailed process as follows:
- Initialize $V^{\pi}(s), \forall s \in S$
- Repeat until convergence
  - Sample tuple ($s_t, a_t, r_t, s_{t+1}$)
  - Update value with a learning rate $\alpha$ \
  $V^{\pi}(s_t) = V^{\pi}(s_t) + \alpha (\underbrace{[r_t + \gamma V^{\pi}(s_{t+1})]}_{\text{TD target}} - V^{\pi}(s_t))$

In MC, $G_t$ is itself estimated by sampling episodes. Bt in TD(0), $r_t + \gamma V^{\pi}(S_{t+1})$ is an estimator of $G_t$. That is, $G_t$ is estimated by other estimator which uses bootstrapping.

## MC vs. TD
- TD can learn online after every step
  - But MC must wait until end of episode before return is known
- TD can learn with incomplete sequences
  - MC can only learn from complete sequences
- TD works in continuing (non-episodic) environments
  - MC only works for episodic (terminating) environments

- MC has high variance and unbiased
  - Simple to use
  - Good convergence properties (even with function approximation)
  - Not very sensitive to initial value
  - Usually more efficient in non-Markov environments

- TD has low variance and biased
  - Usually more efficient than MC
  - TD(0) converges to $V_{\pi}(s)$ (but not always with function approximation)
  - More sensitive to initial value
  - Usually more effective in Markov environments