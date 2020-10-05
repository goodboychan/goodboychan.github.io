---
toc: true
layout: post
description: In this post, it will be explained about the causal graphical model. Especially, we will learn about bayesian networks with aspect of conditional independence and its analysis tool "D-separation". Also we will cover bayesian networks with have different characteristics compared to formal bayesian networks. This post is the summary of "Mathematical principles in Machine Learning" offered from UNIST.
categories: [Machine_Learning]
title: Causal Graphical Model
image: 
---

# Causal Graphical Model

## Directed Acyclic Graph (DAG)

Graph is a visual notation of relationship among a set of nodes, or vertices, and a set of edges which connects between nodes. The expression "Directed" means that each nodes have direction. And if there is a path from a node, goes back to the starting node throught directed edges, the path is called **Cyclic**. Without this, it is called [Directed Acyclic Graph](https://en.wikipedia.org/wiki/Directed_acyclic_graph)(DAG for short) 

![Directed Acyclic Graph]({{site.baseurl}}/assets/image/dag.png "Fig 1. Example of Directed Acyclic Graph (ref. wikipedia)")

Shown in figure, there is no path to return to the starting node throught directed edges, so this is DAG. Usually, we express them with hierarchical relationship, and the node placed in higher level is called ancestor, and node in lower level is called descendant.

## Conditional Dependency

Suppose we have random variable $X$ and $Y$. Its conditional probability of $X$ on $Y$ is defined as a joint probability of $X$ and $Y$ over probability of $Y$.

$$ P(X \vert Y) = \frac{P(X, Y)}{P(Y)} $$

Here, we can assume $X$ and $Y$ are independent if these conditions are satisfied,

- $P(X \vert Y) = P(X)$
- $P(Y \vert X) = P(Y)$
- $P(X, Y) = P(X) \cdot P(Y)$

(actually those conditions are derived from the definition of conditional probability)

And it can be represented with following mathematical symbol

$$ X \perp Y$$

Let's look at three random variables $X, Y, Z$. In this case, $X$ and $Y$ are indepedent conditioned on $Z$ if this condition is safisfied,

$$ P(X, Y \vert Z) = P(X \vert Z) \cdot P(Y \vert Z) $$

## Bayesian Networks

Why should we review the concept of DAG and conditional probabilities? That's because the bayesian network is implemented with DAG and have specific properties related with conditional probabilities.

![Bayesian Network]({{site.baseurl}}/assets/image/bayesian_network.png "Fig 2. DAG of Bayesian Network")

[Bayesian Networks](https://en.wikipedia.org/wiki/Bayesian_network) is structured, graphical representation of probabilistic relationships between several random variables. Here, Nodes represent random variables and edges between nodes represent conditional dependency.

![Bayesian Network]({{site.baseurl}}/assets/image/SimpleBayesNet.png "Fig 3. Simple Bayesian Network (ref. Wikipedia)")

We can draw the conditional probability table of each states. Using this, we calculates the probabilty of future states, like
"What is probability of Grass Wet when it is raining and sprinkler is working?". As shown in the example, the bayesian networks can represent a set of variables and their conditional dependence via a DAG. We can also think about the conditional independence. In this bayesian network.

As a result, the joint probability can be factorized based on the conditional independence in the bayesian networks.

## Dependency in Bayesian Network

Each nodes have the rule of either head and tail. If we have three nodes, $a, b, c$, we can define the relationship in views of rules.

### Tail-to-Tail

![Tail to Tail]({{site.baseurl}}/assets/image/tail_to_tail.png "Fig 4. Tail-to-Tail dependency")

In views of $c$, $a$ and $b$ are considered to tail. So it is called **Tail-to-Tail** dependency. 

- Joint Probability: $p(a, b, c) = p(a \vert c) p(b \vert c) p(c)$
- Independence test: $p(a, b) = \sum_c p(a \vert c) p(b \vert c) p(c) \neq p(a) p(b)$
- Notation: $a \not \perp b \vert \emptyset$

In this case, it cannot be decomposed as product of probability of $a$ and probability of $b$ in general. Therefore, $a$ and $b$ are dependent.

### Head-to-Tail

![Head to Tail]({{site.baseurl}}/assets/image/head_to_tail.png "Fig 5. Head-to-Tail dependency")

It is called **Head-to-Tail** dependency, since $a$ is considered as head, and $b$ is tail.

- Joint Probability: $p(a, b, c) = p(a) p(c \vert a) p(b \vert c)$
- Independence test: $p(a, b) = p(a) \sum_c p(c \vert a) p(b \vert c) = p(a) p(b \vert a) \neq p(a) p(b)$
- Notation: $a \not \perp b \vert \emptyset$

From the independence test, $a$ and $b$ are dependent, same as before.

### Head-to-Head

![Head to Head]({{site.baseurl}}/assets/image/head_to_head.png "Fig 6. Head-to-Head dependency")

**Head-to-Head** dependency is a little bit different from previous case. Here, $a$ and $b$ are not conditionally dependent on any node.

- Joint Probability: $p(a, b, c) = p(a) p(b) p(c \vert a, b)$
- Independence test: $p(a, b) = p(a)p(b)
- Notation: $a \perp b \vert \emptyset$

Thus, $a$ and $b$ are independent.

What if we conditionalize $c$ by observing $c$ in all three cases?

![Tail to Tail observed]({{site.baseurl}}/assets/image/tail_to_tail_ob.png "Fig 7. Tail-to-Tail conditional dependency")

In tail-to-tail case, we can define the joint probability,

$$ p(a, b, c) = p(a \vert c) p(b \vert c) p(c)) $$

Using this, we can redo the independence test,

$$ \begin{aligned} p(a, b \vert c) &= \frac{p(a, b, c)}{p(c)} \\ &= \frac{p(a \vert c) p(b \vert c) p(c)}{p(c)} \\ &= p(a \vert c) p(b \vert c) \end{aligned} $$

In this case, $a$ and $b$ are (conditionally) independent given $c$.

$$ a \perp b \vert c $$

![Head to Tail observed]({{site.baseurl}}/assets/image/head_to_tail_ob.png "Fig 8. Head-to-Tail conditional dependency")

In Head-to-tail case, same process is held,

- Joint probability: $p(a, b, c) = p(a) p(c \vert a) p(b \vert c)$
- Independence test: $\begin{aligned} p(a, b \vert c) &= \frac{p(a, b, c)}{p(c)} \\ &= \frac{p(a) p(c \vert a) p(b \vert c)}{p(c)} = p(a \vert c) p(b \vert c) \end{aligned}$

Same as before, $a$ and $b$ are conditionally independent given $c$.

![Head to Head observed]({{site.baseurl}}/assets/image/head_to_head_ob.png "Fig 9. Head-to-Head conditional dependency")

In contrast of previous cases, head-to-head case $a$ and $b$ are not conditionally independent on $c$.

- Joint probability: $p(a, b, c) = p(a) p(b) p(c \vert a, b)$
- Independence test: $\begin{aligned} p(a, b \vert c) &= \frac{p(a, b, c)}{p(c)} \\ &= \frac{p(a) p(b) p(c \vert a, b)}{p(c)} \neq p(a \vert c) p(b \vert c) \end{aligned}$
- $a \not \perp b \vert c$

## D-separation

From the result, if there is no connection (or path) between $a$ and $b$, A path from $a$ to $b$ is **blocked**. We saw the blocked cases in previous section,

- Tail-to-Tail and observed case
- Head-to-Tail and observed case
- Head-to-Head and not observed case (including its descendents)

If all paths from $a$ to $b$ is blocked by nodes in a set $C$, it is called that $a$ is **D-separated** from $b$ by set $C$. Here D-separation implies conditional independency.

![D separation]({{site.baseurl}}/assets/image/d_separation.png "Fig 10. Example of D-separation")

Suppose we have two simple bayesian network, and we want to know if $a$ and $b$ are conditionally independent whether $c$ is observed or not. We can simply draw the path from $a$ to $b$.

In figure (a), $c$ is observed. In this case, node $e$ shows head-to-head relation. If a node is in head-to-head relation, the node and its descendent should not be observed in order to block the path. (see the Head-to-Head blocked case). But node $c$, which is a descendent of $e$ is observed. So $e$ does not block the path.

Node $f$ is in tail-to-tail relation, but it is not observed. As it doesnot satisfy the blocking condition, it does not block the path either.

As a result, all nodes in path does not satifies the blocking condition, so the path from $a$ to $b$ is not blocked.

$$ a \not \perp b \vert c $$

In figure (b), $f$ is observed. In this case, the path is blocked by $f$, as $f$ is in tail-to-tail relation and $f$ is observed, it satifies the blocking condition. So we can conclude $a$ and $b$ are conditionally independent when $f$ is observed.

## Association vs. Causation

Causation implies association, but association does not necessarily imply causation, so usually conditionally probabilty rather represents association, than causation.

If we are interested in causal relations, it is more natural to represent causal relations directly.

![Causal Bayesian Network]({{site.baseurl}}/assets/image/causal_bayesian_network.png "Fig 11. Causal Bayesian Network")

A bayesian network where each node represents a variable and the edges represent causal relations, it is called [Causal Bayesian Network](https://deepmind.com/blog/article/Causal_Bayesian_Networks) (CBN for short). It has stronger assumptions than bayesian networks, as all the relation should correspond to causal relations. 

CBN describes the causal relationships among variables.
