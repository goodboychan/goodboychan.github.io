---
toc: true
layout: post
description: A summary of "Understanding Deep Reinforcement Learning"
categories: [Reinforcement Learning]
title: epsilon-Soft Random Action
image: 
---

# $\epsilon$-soft Random Action

- A policy is $\epsilon$-soft if all actions $a$ have probability of being chosen
  
$$ P(a) \geq \frac{\epsilon}{\vert A \vert} $$

where $\vert A \vert$ is the number of possible actions.

- Example ($\epsilon$-greedy policy)

    - Let action $a$ from policy $\pi$
    - Select a random number $p$ between 0 and 1
      - if $p < 1 - \epsilon$: (condition 1) \
      choose action $a$
      - else: (condition 2) \
      choose random action among all possible actions

- Proof:
    - Probability of selecting action $a_k$ from condition 1 : $1- \epsilon$
    - Probability of selecting action $a_k$ from condition 2 : $\frac{\epsilon}{\vert A \vert}$
    - Probability of selection action $a_k$ : \
    $P(a_k) = (1 - \epsilon) + \frac{\epsilon}{\vert A \vert}$
    - Sum probabilities of all action : 
    $$\begin{aligned} P(a_k) + \sum_{i \neq k} P(a_i) &= (1-\epsilon) + \frac{\epsilon}{\vert A \vert} + \sum_{i \neq k} \frac{\epsilon}{\vert A \vert} \\ &= (1 - \epsilon) + \frac{\epsilon}{\vert A \vert} + \frac{\epsilon \vert A - 1 \vert}{\vert A \vert} \\ &= (1 - \epsilon) + \epsilon \\ &= 1 \end{aligned}$$
    - Thus, minimum probability of selection action $a$ is $\frac{\epsilon}{\vert A \vert}$
