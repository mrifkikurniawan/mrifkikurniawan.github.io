---
title: Catastrophic Forgetting in Neural Networks Explained
date: 2021-03-30
permalink: /blog-posts/Catastrophic_Forgetting/
related: true
comments: true
excerpt: 'Catastrophic Forgetting in Neural Networks Explained'
tags:
  - Continual Learning
  - Deep Learning
  - Machine Learning
---

# What is Catastrophic Forgetting?

How humans learn is both extremely fascinating and mysterious especially when it comes to the capability to continuously learn new knowledge and skills without forgetting the past experiences. As an example, while we observe the physics phenomena such as the gravitation mechanism and, afterward, acquire new knowledge how the chemistry works, we are able to remember what gravitation is about and explain it effortlessly. In contrast, from the learning intelligence machine perspective, deep learning scientists highly struggle to incorporate the lifelong learning ability into machine learning architecture such as neural networks.  
{: style="text-align: justify;"}

The catastrophic forgetting or alternatively called catastrophic
interference was observed initially by McColskey and Cohen{% cite McCloskey1989 %} in 1898 on shallow 3-layers neural networks who realized that connectionist networks --- a common term in 19's substituting 'neural networks' --- trained on sequential learning prone
to erasing the past learned knowledge. They concluded that adjusting
networks weights representing the old knowledge while training caused
catastrophic interference and it was precipitated and compounded by
distributed representation as the recognized useful properties of
Multi-layer Perceptrons.  
{: style="text-align: justify;"}

Later, this is considered as a more expanded discipline of
'plasticity-stability dilemma{% cite French1999 %}. As a means of the study of tuning the parameters by discovering the most optimum learning algorithm to let the neural networks acquire new knowledge and be sensitive to distributional shifting --- known as plasticity --- but maintaining the past knowledge to address the forgetting --- known as stability. Highly plastic networks potentially suffer from forgetting the past encoded knowledge and oppositely very stable networks could be trouble with efficient information encoding at synapse level{% cite Mermillod2013 %}.  
{: style="text-align: justify;"}

In contrast, cognitive sciences see beyond the field as studying
determining whether the earlier acquired knowledge in life is more
memorized than the knowledge acquired in the coming age or called 'The
Entrenchment Effect'{% cite Mermillod2013 %}. Therefore, it seems a little bit different between what plasticity-stability stands for in deep learning and the cognitive science community.  
{: style="text-align: justify;"}

While the neural networks adapt flexibly to the new incoming knowledge,
it will serendipitously experience catastrophic forgetting. Conversely,
networks that are prone to being unable to discriminate the new incoming inputs if the networks are extremely stable or commonly known as catastrophic remembering{% cite Kaushik2021 %}.  
{: style="text-align: justify;"}

Contemporarily, deep learning is trained on top of a weak but useful
assumption of [i.i.d (independent and identically distributed)](https://deepai.org/machine-learning-glossary-and-terms/independent-and-identically-distributed-random-variables) setting which means that the data points are supposed to be mutually independent  --- single data is unrelated to other data point --- and
having similar distribution e.g. training data is assumed to have
equivalent distribution to test data. Therefore, the common training
setting takes the batch of samples and updates the model parameters with respect to the loss value on this batch. However, the assumption is not applicable for real-time application such as sequentially data stream training settings just like continual learning and accidentally leads to catastrophic forgetting.  
{: style="text-align: justify;"}

Shortly, catastrophic forgetting is the radical performance drops of the model $f(X;\theta)$ which parameterized by $\theta$ with input $X$ --- mostly neural networks exhibit distributed representation\[1\] --- that map $X \rightarrow Y$ performing on previously learned tasks $t_{t}$ after learning on task $t_{n}$ where *t* \< *n*.  
{: style="text-align: justify;"}

![](media/image1.jpeg){width="6.5in" height="3.466666666666667in"}

Figure 1

Consider as an illustration our neural networks train to discriminate
between two classes of cat and dog. Therefore, the network is trained on bunches of datasets containing any variants of cat and dog for some
epochs. Thereafter we want our model to recognize 2 additional classes
of tiger and elephant. Hence, we should train the model with task 2
dataset holding batches of samples of tiger and elephant. In the continual learning setting, we are not allowed to train the model on
both task datasets and getting access to the existing dataset only ---
cluster of tigers and elephants images in this case. As a result, the
model will update the parameters to optimizely perform good at present
task or task 2 and forget how to predict the task 1 classes given task 1 dataset; therefore, reducing the performance on task 1 or called
catastrophic forgetting.  
{: style="text-align: justify;"}

# How Do Neural Networks Forget?

Mostly the standard approach for training the neural networks model is
using standard backpropagation with gradient-based optimization in
particular stochastic gradient descent (SGD){% cite Robbins1951 %} or more sophisticated one like Adam{% cite Kingma2015 %}. Updating parameters via SGD as below  
{: style="text-align: justify;"}

$$\theta \leftarrow \theta - \eta\frac{\partial\mathcal{L}}{\partial\theta}$$

Require $\eta$ for tuning the updating magnitude or called learning rate on gradient of loss with respect to parameters. However, these networks trained by gradient-based optimization algorithms are prone to encounter catastrophic forgetting. The common reason is coming from the primary factor of parameters drift while the neural networks train by taking steps to updating parameters aiming to minimize the loss on task t. Thanks to Masana et al{% cite Masana2020 %} briefly summarize the factors of forgetting, those are including parameters shifting, logits shifting, and Inter-domain/inter-task confusion.  
{: style="text-align: justify;"}

## Parameters shifting

Parameters update while taking steps in order to minimize the loss while training the networks.  
{: style="text-align: justify;"}

## Logits shifting

Direct implication of parameters shifting that bring into distribution
shift of the logits output given the certain input.  
{: style="text-align: justify;"}

## Inter-domain/inter-task confusion

![](media/image2.png){width="3.1666666666666665in"
height="0.9666666666666667in"}

Figure 2

The decision boundary adjustment leading to inter-task or inter-domain
misclassification due to sequential learning setting on continual
learning.  
{: style="text-align: justify;"}

![](media/image3.png){width="5.708333333333333in" height="2.6in"}

Figure 3

Take an example of a binary classification task --- predicting whether
given input *X* resulting discrete label 0 or 1 --- as illustrated in
the figure 2 above, at the beginning the networks learn to predict the
dataset distribution on task 1 in such a way that resulting the model
$f(X;\theta_{0})$ with obtained parameters $\theta_{0}$.  Then whenever
the model acquires the new knowledge from dataset distribution on task 2 without certain continual learning technique, it will suffer from
catastrophic forgetting on distribution dataset on task 1 due to
parameters drift as consequence of distribution drift which lead to
accidentally changing in decision boundary. In contrast, the ideal case
should be like the right image in figure 2 which the model performs well by generating a decision boundary that captures discriminative features on both distributions. This setting can be conveniently achieved on [multi-task learning](https://ruder.io/multi-task/) settings while running the training on both dataset distributions but highly difficult for continual learning.  
{: style="text-align: justify;"}

# Measuring catastrophic forgetting

How to measure catastrophic forgetting could perhaps be separated into
two perspectives thus quantifying to what extent the networks model is
able to acquire new knowledge without forgetting and the other examine
how fast the networks models adapt to past knowledge while relearning
the past task after training on present task, both measurements called
**retention** and **relearning** respectively{% cite Ashley2021 %}.  
{: style="text-align: justify;"}

## Retention

Retention is most commonly used as a measuring technique for continual
learning including incremental class learning or task incremental
learning in the machine learning community nowadays. Simply training the networks until mastering on task 1, then moving forward to task 2 and let the networks mastering on task 2 and followed by measuring the
accuracy metrics on task 1 and 2 independently is categorized as one of
the retention measurements{% cite Ashley2021 %}. Additionally, {% cite Rebuffi2017 %} proposed widely adopted measuring technique called *average incremental accuracy* as formalized by following equation  
{: style="text-align: justify;"}

$$accuracy = \frac{1}{T}\sum_{t = 1}^{T}A_{t}$$

, where *T* is the number of tasks has been encountered so far.  
{: style="text-align: justify;"}

However, more complicated one has been proposed past two years by {% cite Kemker2018 %} which introducing  
{: style="text-align: justify;"}

$$\Omega_{\text{base}} = \frac{1}{T - 1}\sum_{i = 2}^{T}\frac{\alpha_{base,i}}{\alpha_{\text{ideal}}}$$

$$\Omega_{\text{new}} = \frac{1}{T - 1}\sum_{i = 2}^{T}\alpha_{new,i}$$

$$\Omega_{\text{all}} = \frac{1}{T - 1}\sum_{i = 2}^{T}\frac{\alpha_{all,i}}{\alpha_{\text{ideal}}}$$

$T$ is the total tasks/sessions have been trained so far,
$\alpha_{new,i}$ denotes accuracy on test set for session *i* after
learned on session *i*,$\ \alpha_{base,i}$ is the measurement of
accuracy on base class/first session after learning on sesion *i*, while $\alpha_{all,i}$ is accuracy metric on all session given model trained on session *i*, and $\alpha_{\text{ideal}}$ indicates the offline model accuracy on the base set, which assumed the ideal performance or sometime many experiments in continual learning anchor multi-task learning setting as ideal case as subsequently as the upper-bound.  
{: style="text-align: justify;"}

$\Omega_{\text{base}}$ indicates the model's retention relative to the
first session given trained model in later sessions. $\Omega_{\text{new}}$ measures the accuracy on training session *i* while the model is trained on session *i* as well, it is used for model's ability to immediately recall new tasks. While, $\Omega_{\text{all}}$ denotes the measurement for how well the model retain all session after trained on session *i*. Besides, the function of alpha ideal here for normalization for ease to compare between datasets.  
{: style="text-align: justify;"}

## Relearning

Frequently overlooked by existing recent experiments, relearning is
another essential measure in catastrophic forgetting which was initially proposed in physiological study by Hermann Ebbinghaus known as 'savings' but implemented as metrics in catastrophic forgetting by
Hetherington{% cite Hetherington1989 %}. 'Saving' metrics measure the saved knowledge and how fast the networks relearn the past knowledge. This metric is built on top of the assumption that possibly networks are not totally unlearned the past knowledge but that their connections may save encoded important information of the past.  
{: style="text-align: justify;"}

Practically it is measured via training the network on task 1 and task 2 sequentially, then retrain the networks on task 1 dataset and compare
the time required for the network to learn task 1 on the first time
against second time. Reducing time required to relearn the task 1
indicates that the networks still saved the past information.  
{: style="text-align: justify;"}

## Activation Overlap

Activation overlap initially proposed by French{% cite French1993 %} who argue that due to distributed representation causing connectionist networks, forgetting can be measured by quantifying the overlapping in activation output. Recently, this formalized and modified by {% cite Ashley2021 %} by suggesting dot product of two different samples from whether intra-class or inter-class given same hidden parameters as following,  
{: style="text-align: justify;"}

$$s\left( a,b \right) = \frac{1}{n}\sum_{i = 0}^{n}{g_{\text{hi}}\left( a \right)\text{\ .\ \ }g_{\text{hi}}\left( b \right)}$$

Where $g_{\text{hi}}$ indicates hidden layer *i* parameters of the
networks and $g_{\text{hi}}\left( x \right)$ indicating activation
output of input $x$ given parameters $g_{\text{hi}}$.  
{: style="text-align: justify;"}

## Pairwise Interference

Initially proposed by {% cite Liu2019a %} and then implemented by
{% cite Ghiassian2020 %} given sample *a* and sample *b* pairwise
interference measure how large the interference of sample *b*  for
trained model ton sample *a* which can be defined as follow  
{: style="text-align: justify;"}

$$\text{PI}\left( \theta_{t};a,b \right) = J\left( \theta_{t + 1};a \right) - \ J\left( \theta_{t};a \right)$$

. Where, $\theta_{t + 1}$ is a model obtained after training on sample
*b*, and $J(.)$ indicates objective function.  
{: style="text-align: justify;"}

# Overcoming Forgetting in Neural Networks

Contemporarily mitigating catastrophic forgetting highly involved in
subfield of machine learning so-called continual learning. Recent
advancement approaches in dealing with the issue encompassing
exemplar/prototypical/experience rehearsal/replay buffer, parameters
regularization, and architectural modification or otherwise named
modular approach. In spite of those, in the recent past one year some
scientists extend the study of moderating catastrophic forgetting a.k.a.  
{: style="text-align: justify;"}

continual learning to the search of connectivity with multi-task
learning {% cite Mirzadeh2020a %}, loss landscape approximation{% cite Mirzadeh2020a %}{% cite Yin2020a %}, relatedness with transfer learning{% cite Ke2020 %}, more challenging task settings {% cite SonglinDong2020 %}, {% cite Zhao2020 %}, {% cite Bertugli2020 %}, {% cite Caccia2020 %}, {% cite Ren2019 %}, {% cite Dhamija2021 %}, {% cite Rao2019 %} and even expanding beyond image classification task.  
{: style="text-align: justify;"}

## Rehearsal/Replay

Rehearsal/replay approach is dealing with catastrophic forgetting
modestly by replaying the bunch of memory of past knowledge so-called
"episodic memory", e.g., samples of images, into the existing training
steps while learning the novel knowledges e.g., new classes. Therefore,
the catastrophic interference can be diminished as consequence of the
updating parameters in respect of considering batch of combining
existing datasets with small buffer of replayed episodic memory. Among
others this technique was mostly explored and proposed in past five
years in continual learning seeing its simplicity and effectiveness as
baseline for continual learning experiments.  
{: style="text-align: justify;"}

![](media/image4.png){width="4.707699037620298in"
height="2.6611570428696414in"}

Figure 4. Image source: {% cite Parisi2019a %}

The similar mechanism also occurs in our brain when sleeping since our
brain will reactivate and rehearse the past freshly acquired knowledges
memorized periodically in hippocampus into peripheral permanent memory
in neocortex. As suggested in theory of Complementary Learning System
(CLS) {% cite McClelland1995 %} shown in [Figure 4](#_Toc68717821) above, the Hippocampus encodes recent events or experiences via fast learning and these will be unconsciously reactivated while sleeping for gradual consolidation mechanism into neocortical memory systems.  
{: style="text-align: justify;"}

However, the most challenging specialties are both how to sampling the
most significant examples and what kind of representations from the
dataset that necessarily be rehearsed into future learning phase while
minimizing catastrophic interference. Many of the latest research
concerned with this issue along with proposing novel sampling echniques
or including random sampling, uniform sampling, reservoir sampling{% cite Vitter1985 %}{% cite Kim2020a %}, distance-based sampling {% cite Pomponi2020a %}, maximally interfered sampling{% cite Aljundi2019a %}, among others. On the other hand, replaying expressive representations involve naïve image replay, embedding replay, anchor replay, and topological/relational replay.  
{: style="text-align: justify;"}

Among those are

-   Image replay:

-   Generative replay:

-   Embedding replay:

-   Relational representation replay

-   Anchor replay:

## Regularization

Measuring the any past information, including parameters, importance
relevant to both past task loss value and accuracy metrics and
restricting the extreme updates to this information while learning is
the other strategy named regularization approach. This is conceivably
conjectured as the mechanism to control plasticity-stability dilemma of
the neural networks on the subject of continual updates. As consequent,
the restraint adopted to the information of interest, such as parameters, guarantee the minimization of interfered information
essential for the prior task.  
{: style="text-align: justify;"}

Up till now, according to {% cite Delange2021 %}, some of experiments can be clustered into prior-focus/parameters-based and data-focused/logits-based regularization. Parameters-based control the model parameters distribution and plasticity-stability. While, data-focused distil the logits (model outputs before activation function) of given inputs inferenced on the present model as manoeuvre to recall past knowledge.  
{: style="text-align: justify;"}

**Positive/negative**

**Mention the forefront methods**

Among those are

-   Parameters-based

-   Data-based

**Mention latest methods**

## Architectural

**What is**

While architectural-based approach mainly concerned with constructing
progressive neural networks while learning novel tasks or knowledges
either by producing single-independent head on classifier per class/task or rewiring the connections in neural networks layers while
incrementally learning novel tasks.  
{: style="text-align: justify;"}

**Positive/negative**

**Biological Plausible**

**Mention the forefront methods**

**Mention latest methods**

# References

{% bibliography --cited %}