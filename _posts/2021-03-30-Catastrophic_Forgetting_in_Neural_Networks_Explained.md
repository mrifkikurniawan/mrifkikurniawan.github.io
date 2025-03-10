---
layout: distill
title:  Catastrophic Forgetting in Neural Networks Explained
date: 2021-05-06 15:53:00
description: Brief overview why forgetting happens and strategies to combat it
giscus_comments: true
categories: article
related_posts: true
tags: continual_learning deep_learning catastrophic_forgetting
thumbnail: assets/img/9.jpg
bibliography: all_bib.bib
authors:
  - name: Muhammad Rifki Kurniawan
    url: "https://mrifkikurniawan.github.io"
    affiliations:
      name: Xi'an Jiaotong University, Nodeflux

# Optionally, you can add a table of contents to your post.
# NOTES:
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - we may want to automate TOC generation in the future using
#     jekyll-toc plugin (https://github.com/toshimaru/jekyll-toc).
toc:
  - name: What is Catastrophic Forgetting?
    # if a section has subsections, you can add them as follows:
  - name: How Do Neural Networks Forget?
    subsections:
      - name: Parameters shifting
      - name: Logits shifting
      - name: Inter-domain and inter-task confusion
  - name: Measuring Catastrophic Forgetting
    subsections:
      - name: Retention
      - name: Relearning
      - name: Activation Overlap
      - name: Pairwise Interference
  - name: Overcoming Forgetting in Neural Networks
    subsections:
      - name: Rehearsal
      - name: Regularization
      - name: Architectural

_styles: >
  .fake-img {
    background: #bbb;
    border: 1px solid rgba(0, 0, 0, 0.1);
    box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
    margin-bottom: 12px;
  }
  .fake-img p {
    font-family: monospace;
    color: white;
    text-align: left;
    margin: 12px 0;
    text-align: center;
    font-size: 16px;
  }

---
> The existing neural networks is trained on top of a useful assumption of i.i.d setting while contrasting with sequential continual learning problem setting. As a result, the neural networks trained on continual tasks setting will suffer from catastrophic interference which means the networks forget how to do previously learned tasks when they encounter new tasks. This article will dig deep into the reason for forgetting, how to measure this problem, and introduced some available approaches proposed to reducing the abandoning of prior knowledge.  
{: style="text-align: justify;"}  
  
> [Updates]  
> <span style="color:#8B0000">06-05-2021</span>: <span style="color:#1E90FF">Initial article publication</span>  

--- 
# What is Catastrophic Forgetting?

How humans learn is both extremely fascinating and mysterious especially when it comes to the capability to continuously learn new knowledge and skills without forgetting the past experiences. As an example, while we observe the physics phenomena such as the gravitation mechanism and, afterward, acquire new knowledge how the chemistry works, we are able to remember what gravitation is about and explain it effortlessly. In contrast, from the learning intelligence machine perspective, deep learning scientists highly struggle to incorporate the lifelong learning ability into machine learning architecture such as neural networks.  
{: style="text-align: justify;"}

The catastrophic forgetting or alternatively called catastrophic
interference was observed initially by McColskey and Cohen <d-cite key="McCloskey1989"></d-cite> in 1898 on shallow 3-layers neural networks who realized that connectionist networks --- a common term in 19's substituting 'neural networks' --- trained on sequential learning prone
to erase the past learned knowledge. They concluded that adjusting
networks weights representing the old knowledge while training caused
catastrophic interference and it was precipitated and compounded by
distributed representation as the recognized useful properties of
Multi-layer Perceptrons.  
{: style="text-align: justify;"}

Later, this is considered as a more expanded discipline of
'plasticity-stability dilemma' <d-cite key="French1999"></d-cite>. As a means of the study of tuning the parameters by discovering the most optimum learning algorithm to let the neural networks acquire new knowledge and be sensitive to distributional shifting --- known as plasticity --- but maintaining the past knowledge to reduce the forgetting --- known as stability. Highly plastic networks potentially suffer from forgetting the past encoded knowledge and oppositely very stable networks could be trouble with efficient information encoding at synapse level <d-cite key="Mermillod2013"></d-cite>.  
{: style="text-align: justify;"}

In contrast, cognitive sciences see beyond the field as studying
determining whether the earlier acquired knowledge in life is more
memorized than the knowledge acquired in the coming age or called 'The
Entrenchment Effect' <d-cite key="Mermillod2013"></d-cite>. Therefore, it seems a little bit different between what plasticity-stability stands for in deep learning and the cognitive science community.  
{: style="text-align: justify;"}

While the neural networks adapt flexibly to the new incoming knowledge,
it will serendipitously experience catastrophic forgetting. Conversely,
networks that are prone to being unable to discriminate the new incoming inputs if the networks are extremely stable or commonly known as catastrophic remembering <d-cite key="Kaushik2021"></d-cite>.  
{: style="text-align: justify;"}

Contemporarily, deep learning is trained on top of a weak but useful
assumption of [i.i.d (independent and identically distributed)](https://deepai.org/machine-learning-glossary-and-terms/independent-and-identically-distributed-random-variables) setting which means that the data points are supposed to be mutually independent  --- single data is unrelated to other data point --- and
having similar distribution e.g. training data is assumed to have
equivalent distribution to test data. Therefore, the common training
setting takes the batch of samples and updates the model parameters with respect to the loss value on this batch. However, the assumption is not applicable for real-time application such as sequentially data stream training settings just like continual learning and accidentally leads to catastrophic forgetting.  
{: style="text-align: justify;"}

Shortly, catastrophic forgetting is the radical performance drops of the model $f(X;\theta)$ which parameterized by $\theta$ with input $X$ --- mostly neural networks exhibit distributed representation <d-cite key="McCloskey1989"></d-cite> --- that map $X \rightarrow Y$ performing on previously learned tasks $t_{t}$ after learning on task $t_{n}$ where *t* \< *n*.  
{: style="text-align: justify;"}

<div class="fake-img l-body">
    <div class="col-sm mt-3 mt-md-0 text-center">
        {% include figure.liquid path="assets/img/catastrophic_forgetting/forgetting_cl_task.jpg" zoomable=true class="img-fluid rounded z-depth-1 center"%}
    </div>
</div>
<div class="caption">
    Continual learning task setting is designed for the model to learning multiple tasks incrementally which each individual task encompass a set of some classes.
</div>


Consider as an illustration on figure 1 above, our neural networks train to discriminate
between two classes of cat and dog. Therefore, the network is trained on bunches of datasets containing any variants of cat and dog for some
epochs. Thereafter we want our model to recognize 2 additional classes
of tiger and elephant in task 2. Hence, we should train the model with task 2
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
particular stochastic gradient descent (SGD) <d-cite key="Robbins1951"></d-cite> or more sophisticated one like Adam <d-cite key="Kingma2015"></d-cite>. Updating parameters via SGD as below  
{: style="text-align: justify;"}

$$\theta \leftarrow \theta - \eta\frac{\partial\mathcal{L}}{\partial\theta},$$

require $\eta$ for tuning the updating magnitude or called learning rate on the parameters gradient $\frac{\partial\mathcal{L}}{\partial\theta}$. However, these networks trained by gradient-based optimization algorithms are prone to encounter catastrophic forgetting. The common reason is coming from the primary factor of parameters drift while the neural networks train by taking steps to updating parameters aiming to minimize the loss on task $t$. Thanks to Masana et al <d-cite key="Masana2020"></d-cite> briefly summarize the factors of forgetting, those are including parameters shifting, logits shifting, and Inter-domain/inter-task confusion.  
{: style="text-align: justify;"}

## Parameters shifting

While the networks are being trained on the current task, the parameters will be tuned with respect to loss value in the current training dataset task. It means that the networks are optimized to perform maximum on the current task by changing the parameters. As a result, the optimization and parameters update will not consider previous task distribution which lead to forgetting how to do preceding tasks.    
{: style="text-align: justify;"}

## Logits shifting

The direct ramification of parameters shifting outputs distribution deviation of the logits given the certain input e.g., image of the previous task. In the effort to alleviate this detriment, distilling the knowledge <d-cite key="Hinton2015"></d-cite> of the previous model parameters respecting the old inputs squeezes the logits outputs of the current model to be equal to the previous model logits while allowing the parameters inconsistent to the old model.  
{: style="text-align: justify;"}

## Inter-domain and inter-task confusion
<div class="fake-img l-body">
    <div class="col-sm mt-3 mt-md-0 text-center">
        {% include figure.liquid path="assets/img/catastrophic_forgetting/forgetting_inter_task.svg" zoomable=true class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
  The networks are susceptible to misclassify task 3 (right) classes due to the model having not trained to create discriminative decision boundary for 4 classes in task 3 because of sequential learning on task 1 and task 2 separately. Source: <a href="https://arxiv.org/abs/2010.15277">Masana, M. et al.</a>
</div>

The decision boundary adjustment leading to inter-task or inter-domain
misclassification due to sequential learning setting on continual
learning.  
{: style="text-align: justify;"}

<div class="fake-img l-body">
    <div class="col-sm mt-3 mt-md-0 text-center">
        {% include figure.liquid path="assets/img/catastrophic_forgetting/forgetting_forgetting.svg" style="width:85%" zoomable=true class="img-fluid rounded z-depth-1 center" %}
    </div>
</div>
<div class="caption">
    Catastrophic forgetting in binary classification while the networks are trained on task 2 suffering from distributional shift which leads to forgetting to do discrimination on task 1. Source: <a href="https://www.semanticscholar.org/paper/Attention-Based-Selective-Plasticity-Kolouri-Ketz/fd45befff6852def1ba78ef0d2cd18f5e0f62f68">Kolouri, S. et al</a>
</div>

Take an example of a binary classification task --- predicting whether
given input *X* resulting discrete label 0 or 1 --- as illustrated in
the figure 3 <d-cite key="French1999"></d-cite> above, at the beginning the networks learn to predict the
dataset distribution on task 1 in such a way that resulting the model
$f(X;\theta_{0})$ with obtained parameters $\theta_{0}$.  Then whenever
the model acquires the new knowledge from dataset distribution on task 2 without certain continual learning technique, it will suffer from
catastrophic forgetting on distribution dataset on task 1 due to
parameters drift as consequence of distribution drift which lead to
accidentally changing the decision boundary. In contrast, the ideal case
should be like the right image in figure 3 which the model performs well by generating a decision boundary that captures discriminative features on both distributions. This setting can be conveniently achieved on [multi-task learning](https://ruder.io/multi-task/) settings while running the training on both dataset distributions but highly difficult for continual learning.  
{: style="text-align: justify;"}

# Measuring Catastrophic Forgetting

How to measure catastrophic forgetting could perhaps be separated into
two perspectives thus quantifying to what extent the networks model is
able to acquire new knowledge without forgetting and the other examine
how fast the networks models adapt to past knowledge while relearning
the past task after training on present task, both measurements called
**retention** and **relearning** respectively <d-cite key="Ashley2021"></d-cite>.  
{: style="text-align: justify;"}

## Retention

Retention is most commonly used as a measuring technique for continual
learning including incremental class learning or task incremental
learning in the machine learning community nowadays. Simply training the networks until mastering on task 1, then moving forward to task 2 and let the networks mastering on task 2 and followed by measuring the
accuracy metrics on task 1 and 2 independently is categorized as one of
the retention measurements <d-cite key="Ashley2021"></d-cite>. Additionally, <d-cite key="Rebuffi2017"></d-cite> proposed widely adopted measuring technique called ***average incremental accuracy*** as formalized by following equation  
{: style="text-align: justify;"}

$$accuracy = \frac{1}{T}\sum_{t = 1}^{T}A_{t},$$

where *T* is the number of tasks has been encountered so far and $A_{t}$ means accuracy on tast $t$.  
{: style="text-align: justify;"}

However, more complicated one has been proposed by <d-cite key="Kemker2018"></d-cite> which introducing  
{: style="text-align: justify;"}

$$\Omega_{\text{base}} = \frac{1}{T - 1}\sum_{i = 2}^{T}\frac{\alpha_{base,i}}{\alpha_{\text{ideal}}},$$

$$\Omega_{\text{new}} = \frac{1}{T - 1}\sum_{i = 2}^{T}\alpha_{new,i},$$

$$\Omega_{\text{all}} = \frac{1}{T - 1}\sum_{i = 2}^{T}\frac{\alpha_{all,i}}{\alpha_{\text{ideal}}},$$

$T$ is the total tasks/sessions have been trained so far,
$\alpha_{new,i}$ denotes accuracy on test set for session *i* direcly after learning,$\ \alpha_{base,i}$ is the measurement of
accuracy on base class/first session after learning on sesion *i*, while $\alpha_{all,i}$ is accuracy metric on all session given model trained on session *i*, and $\alpha_{\text{ideal}}$ indicates the offline model accuracy on the base set, which assumes the ideal performance or sometimes many experiments in continual learning anchor multi-task learning setting as the upper-bound. While, the function of alpha ideal as divisor here for normalization for ease to compare between datasets.  
{: style="text-align: justify;"}

$\Omega_{\text{base}}$ indicates the model's retention relative to the
first session given trained model in later sessions. $\Omega_{\text{new}}$ measures the accuracy on training session *i* while the model is trained on session *i* as well, it is used for a model's ability to immediately recall new tasks. While, $\Omega_{\text{all}}$ denotes the measurement for how well the model retain all session after trained on session *i*.  
{: style="text-align: justify;"}

## Relearning

Frequently overlooked by existing recent experiments, relearning is
another essential measure in catastrophic forgetting which was initially proposed in physiological study by Hermann Ebbinghaus known as 'savings' but implemented as metrics in catastrophic forgetting by
Hetherington <d-cite key="Hetherington1989"></d-cite>. 'Saving' metrics measure the saved knowledge and how fast the networks relearn the past knowledge. This metric is built on top of the assumption that possibly networks are not totally unlearned the past knowledge but that their connections may save encoded important information of the past.  
{: style="text-align: justify;"}

Practically it is measured via training the network on task 1 and task 2 sequentially, then retrain the networks on task 1 dataset and compare
the time required for the network to learn task 1 on the first time
against second time. Reducing time required to relearn the task 1
indicates that the networks still saved the past information.  
{: style="text-align: justify;"}

## Activation Overlap

Activation overlap initially proposed by French <d-cite key="French1993"></d-cite> who argue that due to distributed representation causing connectionist networks, forgetting can be measured by quantifying the overlapping in activation output. Recently, this formalized and modified by <d-cite key="Ashley2021"></d-cite> by suggesting dot product of two different samples from whether intra-class or inter-class given same hidden parameters as following,  
{: style="text-align: justify;"}

$$s\left( a,b \right) = \frac{1}{n}\sum_{i = 0}^{n}{g_{\text{hi}}\left( a \right)\text{.}g_{\text{hi}}\left( b \right)}$$

where $g_{\text{hi}}$ indicates hidden layer *i* parameters of the
networks and $g_{\text{hi}}\left( x \right)$ indicating activation
output of input $x$ given parameters $g_{\text{hi}}$.  
{: style="text-align: justify;"}

## Pairwise Interference

Initially proposed by <d-cite key="Liu2019a"></d-cite> and then implemented by
<d-cite key="Ghiassian2020"></d-cite> given sample *a* and sample *b* pairwise
interference measure how large the interference of sample *b*  for
trained model on sample *a* which can be defined as follow  
{: style="text-align: justify;"}

$$\text{PI}\left( \theta_{t};a,b \right) = J\left( \theta_{t + 1};a \right) - \ J\left( \theta_{t};a \right).$$

Where, $\theta_{t + 1}$ is a model obtained after training on sample
*b*, and $J(.)$ indicates objective function.  
{: style="text-align: justify;"}

# Overcoming Forgetting in Neural Networks

Contemporarily mitigating catastrophic forgetting highly involved in
subfield of machine learning so-called continual learning. Recent
advancement approaches in dealing with the issue encompassing
exemplar/prototypical/experience rehearsal/replay buffer, parameters
regularization, and architectural modification or otherwise named
modular approach. In spite of those, in the recent past one year some
scientists extend the study of moderating catastrophic forgetting a.k.a. continual learning to the search of connectivity with multi-task
learning <d-cite key="Mirzadeh2020a"></d-cite>, loss landscape approximation <d-cite key="Mirzadeh2020a"></d-cite>, <d-cite key="Yin2020a"></d-cite>, relatedness with transfer learning <d-cite key="Ke2020"></d-cite>, more challenging task settings <d-cite key="SonglinDong2020"></d-cite>, <d-cite key="Zhao2020"></d-cite>, <d-cite key="Bertugli2020"></d-cite>, <d-cite key="Caccia2020"></d-cite>, <d-cite key="Ren2019"></d-cite>, <d-cite key="Dhamija2021"></d-cite>, <d-cite key="Rao2019"></d-cite> and even expanding beyond image classification task <d-cite key="Joseph2020a"></d-cite>, <d-cite key="Perez-Rua2020"></d-cite>, <d-cite key="Zheng2021"></d-cite>, <d-cite key="Chen2020d"></d-cite>.  
{: style="text-align: justify;"}

## Rehearsal

Rehearsal/replay approach is dealing with catastrophic forgetting
modestly by replaying the bunch of knowledge memory of past knowledge so-called
"episodic memory", e.g., samples of images, into the existing training
steps while learning the novel knowledge e.g., new classes. Therefore,
the catastrophic interference can be diminished as consequence of the
updating parameters in respect of considering batch of combining
existing datasets with small buffers of replayed episodic memory. Among
others this technique was mostly explored and proposed in past five
years in continual learning seeing its simplicity and effectiveness as
baseline for continual learning experiments.  
{: style="text-align: justify;"}

<div class="fake-img l-body">
    <div class="col-sm mt-3 mt-md-0 text-center">
        {% include figure.liquid path="assets/img/catastrophic_forgetting/forgetting_hippo.svg" style="width:85%" zoomable=true class="img-fluid rounded z-depth-1 center" %}
    </div>
</div>
<div class="caption">
    Knowledge replays in the brain while sleeping involving the Neocortex and Hippocampus. Source: <a href="https://www.semanticscholar.org/paper/Continual-Lifelong-Learning-with-Neural-Networks%3A-A-Parisi-Kemker/9ea50b3408f993853f1c5e374690e5fbe73c2a3c">Parisi, G. I. et al.</a>
</div>


The similar mechanism also occurs in our brain when sleeping since our
brain will reactivate and rehearse the past freshly acquired knowledges
memorized periodically in hippocampus into peripheral permanent memory
in the neocortex. As suggested in theory of Complementary Learning System
(CLS) <d-cite key="McClelland1995"></d-cite> shown in Figure above, the Hippocampus encodes recent events or experiences via fast learning and these will be unconsciously reactivated while sleeping for gradual consolidation mechanism into neocortical memory systems.  
{: style="text-align: justify;"}

However, the most challenging in rehearsal approach is both how to sampling the
most significant examples and what kind of representations from the
dataset that necessarily be rehearsed into future learning phase while
minimizing catastrophic interference. Many of the latest research
concerned with this issue along with proposing novel sampling techniques
or including random sampling, uniform sampling, reservoir sampling <d-cite key="Vitter1985"></d-cite>, <d-cite key="Kim2020a"></d-cite>, distance-based sampling <d-cite key="Pomponi2020a"></d-cite>, maximally interfered sampling <d-cite key="Aljundi2019a"></d-cite>, among others. On the other hand, replaying expressive representations involve naïve image replay, embedding replay, anchor replay, and topological/relational replay.  
{: style="text-align: justify;"}

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

Up till now, according to <d-cite key="Delange2021"></d-cite>, some of experiments can be clustered into prior-focus/parameters-based and data-focused/logits-based regularization. Parameters-based control the model parameters distribution and plasticity-stability. While, data-focused distil the logits (model outputs before activation function) of given inputs inferenced on the present model as manoeuvre to recall past knowledge.  
{: style="text-align: justify;"}

<div class="fake-img l-body">
    <div class="col-sm mt-3 mt-md-0 text-center">
        {% include figure.liquid path="assets/img/catastrophic_forgetting/ewc.svg" style="width:85%" zoomable=true class="img-fluid rounded z-depth-1 center" %}
    </div>
</div>
<div class="caption">
    Training with EWC as shown on red trajectories will finding out low loss elevation in space on both task A (old) and task B (new) such that the obtained parameters are capable to perform accurately both on task A and B. Source: <a href="https://www.pnas.org/content/114/13/3521">Kirkpatrick et al.</a>
</div>

The earliest method proposed this idea was Elastic Weight Consolidation (EWC) <d-cite key="Kirkpatrick2017a"></d-cite>. The basic idea is to measure weight importance for the previous task while controlling these previous weights $\theta_{A}^{*}$ and avoid significant updates to these weights via fisher information matrix $F$ as measure of the importance. While EWC minimize the loss of  
{: style="text-align: justify;"}

$$\mathcal{L}\left ( \theta  \right ) = \mathcal{L}_{B}\left ( \theta \right ) + \sum_{i}^{} \frac{\lambda}{2}F_{i}\left ( \theta_{i} - \theta_{A,i}^{*} \right )^{2},$$

where $\mathcal{L}_{B}$ is the task B loss, $\lambda$ denotes the relation of old task to new, $i$ is each parameter index, and $\theta$ is the current parameters. As exhibited on the loss equation above, the new parameters will be enforced to close to old parameters to alleviate forgetting which the precision will be controlled by fisher information matrix $F$.  
{: style="text-align: justify;"}

## Architectural

While architectural-based approach mainly concerned with constructing progressive neural networks while learning novel tasks or knowledges either by growing task-specific architecture <d-cite key="Rusu2016a"></d-cite>, producing single-independent head on classifier per class/task <d-cite key="Li2019LearnTG"></d-cite>, or rewiring the connections in neural networks layers while incrementally learning novel tasks <d-cite key="Wortsman2020a"></d-cite>.  
{: style="text-align: justify;"}

<div class="fake-img l-body">
    <div class="col-sm mt-3 mt-md-0 text-center">
        {% include figure.liquid path="assets/img/catastrophic_forgetting/progessive_networks.svg" style="width:85%" zoomable=true class="img-fluid rounded z-depth-1 center" %}
    </div>
</div>
<div class="caption">
    Progressive networks exhibit three columns networks which each column is associated to task-specific networks e.g., networks on column 1 and 2 for performing task 1 and 2, respectively. Whilst column 3 networks solve task 3 that this networks is enabled getting access to previous leaned features. Source: <a href="https://www.semanticscholar.org/paper/Progressive-Neural-Networks-Rusu-Rabinowitz/53c9443e4e667170acc60ca1b31a0ec7151fe753">Rusu, A. A. et al.</a>
</div>

Among others is Progressive Neural Networks proposed in 2016 as depicted in the figure 6 above. The progressive networks framework proposed addressing catastrophic forgetting through evolving task-specific networks instanting on a column for working on a task being solved. Then, as the task encountered is incremental growth, the novel column networks will be introduced which the previously learned features feasibly transferred to the new networks via lateral connections. Therefore, the last task with its associated networks are allowed to exploit all the features learned so far.  
{: style="text-align: justify;"}  
  
  

> [Notes]  
> <span style="color:#8B0000">If you have any disapproval, correction, and critique to this article feel free to <a href="mailto:mrifkikurniawan17@gmail.com">email me</a>, I will happily adjusting and modifying this published contents respecting the corrections.</span>  
{: style="text-align: justify;"}