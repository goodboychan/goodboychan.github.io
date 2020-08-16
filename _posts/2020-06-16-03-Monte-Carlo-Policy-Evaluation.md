---
toc: true
layout: post
description: A summary of "Understanding Deep Reinforcement Learning"
categories: [Reinforcement_Learning]
title: Monte Carlo Policy Evaluation
classes: wide
image: 
---

# Monte Carlo Methods and Temporal Difference Learning in Policy Evaluation

## Monte Carlo Policy Evaluation
At first, generate and store episode information under policy $\pi$: \
in $i$-th episode:

$$ s_{i, 1}, a_{i, 1}, r_{i,1}, s_{i, 2}, a_{i, 2}, r_{i, 2}, \dots, s_{i, T} $$

Based on this information, Calculate total expected return $G_t$ for each episode: 
$$ G_t = r_t + \gamma * r_{t+1} + \gamma^2 * r_{t+2} + \cdots $$

Monte Carlo (MC) Policy Evaluation estimates expectation ($V^{\pi}(s) = E_{\pi}[G_t \vert s_t = s]$) by iteration using
  - Empirical average: given episodes sampled from policy $\pi$
  - Importance Sampling: reweighted empirical average \
  (for example, apply more weights on latest episode information, or apply more weights on important episode information, etc...)

MC Policy Evaluation does not require transition dynamics ($T$) and reward function ($r$) in MDP. And total expected return can be calculated based on collected return, so Bootstrapping method (calculate future reward with current information) is not required. Also, It does not assume that state follows Markov Property.
But one constraint is that it can only be applied to episodic MDPs
  - Averaging over returns from a complete episode
  - Requires each episode to terminate \
  $\rarr$ Cannot fully online learning

Detailed process:
- Generate episodes under policy $\pi$
- Calculate $G$ from each episode
- Calculate MC empirical average return from episodes which visit the state $s$ to approximate value function: 
  $$V^{\pi}(s) = E_{\pi}[G_t \vert s_t = s]$$
- If we have more episodes, update estimate of $V^{\pi}$ with these episodes.

## First-Visit Monte Carlo (MC) Policy Evaluation
When we update value function $V^{\pi}(s)$, we select the **first visit** state value.

1. Initialize $N(s) = 0 = G(s),\forall_s \in S$ \
($N(s)$: number of visitation, $G(s)$: total expected return in state $s$)
2. Loop
   1. Generate Episode $i = s_{i,1}, a_{i,1}, r_{i,1}, s_{i,2}, a_{i,2}, r_{i,2}, \dots, s_{i,T_i}$
   2. Define return of $i$-th episode from time step $t$:\
   $G_{i,t} = r_{i,t} + \gamma r_{i, t+1}, \gamma^2r_{i,t+2} + \cdots + \gamma^{T_i-1} r_{i, T_i}$
   3. For each state $s$ visited in $i$-th episode \
      -For **first** time $t$ that state $s$ is visited in $i$-th episode,
         - Increment counter of total first visits: $N(s) = N(s) + 1$
         - Increment total return: $G(s) = G(s) + G_{i, t}$
         - Update estimate: $V^{\pi}(s) = \dfrac{G(s)}{N(s)}$ 

## Every-Visit Monte Carlo (MC) Policy Evaluation
When we update value function $V^{\pi}(s)$, we select the **all visit** state value.

1. Initialize $N(s) = 0 = G(s),\forall_s \in S$ \
($N(s)$: number of visitation, $G(s)$: total expected return in state $s$)
2. Loop
   1. Generate Episode $i = s_{i,1}, a_{i,1}, r_{i,1}, s_{i,2}, a_{i,2}, r_{i,2}, \dots, s_{i,T_i}$
   2. Define return of $i$-th episode from time step $t$:\
   $G_{i,t} = r_{i,t} + \gamma r_{i, t+1}, \gamma^2r_{i,t+2} + \cdots + \gamma^{T_i-1} r_{i, T_i}$
   3. For each state $s$ visited in $i$-th episode \
      -For **every** time $t$ that state $s$ is visited in $i$-th episode,
         - Increment counter of total first visits: $N(s) = N(s) + 1$
         - Increment total return: $G(s) = G(s) + G_{i, t}$
         - Update estimate: $V^{\pi}(s) = \dfrac{G(s)}{N(s)}$ 

## Incremental Monte Carlo (MC) Policy Evaluation
Nonetheless we choose first-visit or every-visit MC, we incrementally update state value function $V^{\pi}(s)$

1. Initialize $N(s) = 0,\forall_s \in S$ \
($N(s)$: number of visitation, $G(s)$: total expected return in state $s$)
2. Loop
   1. Generate Episode $i = s_{i,1}, a_{i,1}, r_{i,1}, s_{i,2}, a_{i,2}, r_{i,2}, \dots, s_{i,T_i}$
   2. Define return of $i$-th episode from time step $t$:\
   $G_{i,t} = r_{i,t} + \gamma r_{i, t+1}, \gamma^2r_{i,t+2} + \cdots + \gamma^{T_i-1} r_{i, T_i}$
   3. For each state $s$ visited in $i$-th episode \
      -For **every (or first)** time $t$ that state $s$ is visited in $i$-th episode,
         - Increment counter of total first visits: $N(s) = N(s) + 1$
         - Assume that, \
         $V_{old} = \dfrac{G_{old}}{N} \rarr G_{old} = N * V_{old}$ \
         if we gather new expected return $G_{i, t}$,\
         $\begin{aligned} V_{new} &= \dfrac{N * V_{old} + G_{i, t}}{N + 1} \\ &= \frac{N}{N+1} V_{old} + \frac{1}{N+1} G_{i, t} \end{aligned}$ \
         We can replace: $N+1 \rarr N(s)$

         - Update estimate: \
         $\begin{aligned}V^{\pi}(s) &= V^{\pi}(s)\dfrac{N(s)-1}{N(s)} + G_{i, t} \frac{1}{N(s)} \\ &= V^{\pi}(s) + \frac{1}{N(s)}\big(G_{i, t} - V^{\pi}(s)\big) \end{aligned}$ \
         (Here, we can consider to replace the weight value $\frac{1}{N(s)}$)

## Incremental Monte Carlo (MC) Policy Evaluation with learning-rate
Nonetheless we choose first-visit or every-visit MC, we incrementally update state value function $V^{\pi}(s)$

1. Initialize $N(s) = 0,\forall_s \in S$ \
($N(s)$: number of visitation, $G(s)$: total expected return in state $s$)
2. Loop
   1. Generate Episode $i = s_{i,1}, a_{i,1}, r_{i,1}, s_{i,2}, a_{i,2}, r_{i,2}, \dots, s_{i,T_i}$
   2. Define return of $i$-th episode from time step $t$:\
   $G_{i,t} = r_{i,t} + \gamma r_{i, t+1}, \gamma^2r_{i,t+2} + \cdots + \gamma^{T_i-1} r_{i, T_i}$
   3. For each state $s$ visited in $i$-th episode \
      -For **every (or first)** time $t$ that state $s$ is visited in $i$-th episode,
         - Increment counter of total first visits: $N(s) = N(s) + 1$
         - Update estimate using **learning rate $\alpha$**: \
         $V^{\pi}(s) = V^{\pi}(s) + \alpha\big(G_{i, t} - V^{\pi}(s)\big)$
> Note: if $\alpha > \frac{1}{N(s)}$, then we choose more weight for newer data. This is helpful for non-stationary domains. 

## Bias, Variance and Mean Squared Error

Consider a statistical model that is parameterized by $\theta$ and that determins a probability distribution over observed data $P(x \vert \theta)$. And this model generates the statistics $\hat{\theta}$ that provides an estimate of $\theta$ using observed data $x$.

Then Bias of an estimator $\hat{\theta}$ is defined by,
$$ \text{bias}(\hat{\theta}) = E_{x \vert \theta}[\hat{\theta}] - \theta $$

And Variance of an estimator $\hat{\theta}$ is defined by,
$$ \text{Var}(\hat{\theta}) = E_{x \vert \theta}[(\hat{\theta} - E_{x \vert \theta}[\hat{\theta}])^2] $$

MSE (Mean Squared Error) of an estimator $\hat{\theta}$ is defined:
$$ \begin{aligned} \text{MSE}(\hat{\theta}) &= E_{x \vert \theta}[(\hat{\theta} - \theta)^2] \\ &= \text{Var}(\hat{\theta}) + \text{bias}(\hat{\theta})^2 \end{aligned}$$

And it is called the bias-variance decomposition of MSE.

## Bias and Variance of MC Policy Evaluation

In First-visit MC policy evaluation,
- $V^{\pi}$ estimator is an unbiased estimator of true $E_{\pi}[G_t \vert s_t = s]$ \
(value unchanged during episode)
- By law of large numbers, we have 
$$ V^{\pi}(s) \rarr E_{\pi}[G_t \vert s_t = s] \text{   as   } N(s) \rarr \infty $$

In Every-visit MC policy evaluation,
- $V^{\pi}$ estimator is a biased estimator of true $E_{\pi}[G_t \vert s_t = s]$ \
(value changed in every visitation)
- But consistent estimator and often has better MSE

Generally, MC policy Evaluation rely on the experience we gather. So its estimator has high variance, and requires lots of data to reduce variance.

## Limitation of MC Policy Evaluation
To estimate expectation, it requires episodic settings. Episode must end before we use data of that episode for updating the value function. And we cannot make sure when the estimated value is converged in true value.(it requires some assumption..)
