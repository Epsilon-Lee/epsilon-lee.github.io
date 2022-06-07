```
layout: post
title: "Intern Supervision Project"
author: Guanlin Li
tag: diary
```



- toc
{:toc}
### May 29 2022

- Ruili
  - 这个是我按照昨天讨论的思路初步整理的流程图，上面的应该和昨天讨论的基本一样；下面的是我感觉说不定也可行的一种思路，这里和之前的流程的主要区别是之前说的第一次句子级检索也许不一定需要，或者说能不能直接用原始的bert进行mention级检索得到的带标签样例来微调bert-tagger来同时实现bert的domain adaptation和label-aware的encoding，这样就比原来的流程少了一次pre-tune bert的过程，也能用更多的带标签样本来训练bert-tagger，但是没有任何微调的bert进行mention级检索的结果是存在标签噪声的，用来训练bert-tagger有可能会影响到后续检索系统的性能
  - 关于去掉mention embedding，只用context的token embedded的检索结果，我大概看了一下，还是不算太好，虽然也不算很差，主要是容易出现句子的语境有关联，句子中也有正确的mention，但同时把句中其它不正确mention也检索到了的情况，检索时添加的label信息似乎也没有很明显的作用，具体结果我也打包传上来吧
  - 这周主要是按照利用检索实现数据扩增的新思路把之前的流程重新整理了一遍，也把上周提到的去掉mention的表示的检索结果大致看了一下，整体流程的流程图、检索结果以及大致结论都在早些时候发在群里了。目前我还在看bert-tagger实现的相关信息，一些细节可以明天讨论一下



- Guanlin
  - Hearst Pattern

- Scaling-up labelling

---

- Dense Retriever as simple labeller
  - $$D = \{(x, y)\}$$, $$D'=\{x\}$$
  - $$\text{BERT}(x)$$ tuning, contrastive learning
    - similar to DPR
  - For every $$(x, y) \in D$$, retrieve top-k $$x'$$ in $$D'$$ and label it to $$y$$ as psudo training set $$D''$$
  - Training:
    - Train model $$M$$ with $$D$$ 
    - use $$M$$ to filter $$D''$$ as importance sampling
      - Uncertainty
- A general framework:
  - used for every NLP semi-supervised tasks
  - FET
    - Long-tailed problem

---

- Cai Deng outstanding paper
- 

---

TODO:

1. Retriever tuning
   - 在D上训练BERT Tagger/+作为retriever, span based training
   - CL
2. Retriever indexing and pseudo data construction

---

1. 用检索的数据+有监督数据一起训练一个M去进行测试



x = I live [in San Fransisco City .]

y = O O O  B     I                O



x = I live in Shanghai .



Classifier:

BERT(x)[m]

ctx embedding + mention embedding (avg San City)

- in, .



---

Procedure:

- (BERT)  BERT + span representation training classifier as retriever (encoder): ctx avg + mention embedding
  - mention
    - mention 外边沿两词的embedding + mention avg: 3d --> d --> softmax
  - Table valid, test
- BERT --> Use d-dim vector for Dense indexing
- Pseudo data construction D''
- D'' + D: train final model
  - Top-k 10

---



### May 22 2022

- Ruili
  - 这周主要是整理了一下FIGER和BBN两个测试集的样本标签分布情况，然后从这里面占比较高的几个类别里选择了一些样例进行检索，粗略来看这些类别的mention很多都是纯专有名词的形式，初步感觉检索的准确率还是不算很高；至于把标签信息加入query的方法，我目前采用的是提取标签词（比如/location/city就取"location city"，所有标签词不重复）后将其作为句子对中的第二句，用bert编码后仅考虑原句相关的词向量进行后续处理，粗略来看效果也不是很明显

- Lemao
  - Weak retrieval-and-label is faster than self-training: convergence might be faster than SL
  - FET task
  - label --> instance --> retrieval --> label
  - Zero-shot FET task
    - 阶段一：用prompt做initialization --> initial psuedulabels
      - 贡献：zero-shot的系统，充分利用scaling-up的数据
      - Pool embedding 的模型需要fix
    - 



### May 15 2022

- Ruili
  - `Ctx`/`Cls` concat `Mention`
  - 这周我试了用mention token的平均作为mention representation，用cls token作为整句的contex representation（clscat）和除去mention token和特殊token（头尾的cls和sep）之外其余token的平均作为contex representation（ctxavg），并将两者concat起来作为整体representation的检索效果，
  - 我粗略的看了下，应该是比上周的方法好了不少，context token取平均的效果似乎比直接用cls要好一点，但和理想的效果还是有点差距的，
  - 除了上周的FIGER数据集之外我还试了一下从BBN数据集的测试集里采的example，
    - 记下了每个example的前20个、100左右的10个
    - 和10000个结果中的最后10个检索结果和相应的distance（应该是l2距离），
  - 我就先把目前的结果发上来，明天我再试试加一点自己构造的有标签样本的检索效果

- Guanlin

  - `query representation` should take advantage of **label** to become label-aware query so as to sieve away unrelated context

    - e.g.

      - > *"Spencer Anderson was in [[veterinary school]] when he joined the {U.S. Army} in 2000."* -- label: */person/doctor*

  - 普通名词的accuracy较低，上下文更强

    - e.g.

      - > *"While serving he taught himself xxx."*

  - mention表示的string matching effect and its over-generalization to label-irrelevant surface pattern

    - e.g.

      - > "*This speaks very highly of the quality of hiring we do in spite of the budget problems, \" said professor [Charles Campbell], who was elected as an Fellow in 2010.*" -- label: */person/author*

      - Pattern: `said ..., who ...` `Chareles` `Campbell`

    - e.g.

      - > *"After his military graduation, Spencer spent three years tending the Army's dogs and horses -- nearly a year of that in Iraq."*

      - Find

        - > *"Brother - in - law of Pippa Middleton, Spencer first embarked on the road to riches in 2011 when he became known as Made In Chelsea's premier Lothario. At the same time, he worked as a city trader, earning"*
          >
          > **string matching** without context generalization

        - > *"His father was a paratrooper during World War II, while his Uncle Herman, who played mandolin, was a musical influence on young Spencer : At age 6 he learned harmonica and accordion."*
          >
          > **both string matching and context generalization** but wrong

  - Global threshold may not be possible

    - e.g. 

      - > *"Washington women's head basketball coach Kevin McGuff anticipated his team's depth would be one of its major strengths entering this season."* -- label: *"/person/coach*
        >
        > we can find some followings:

        - > *"U. S. teammate Ryan McDonagh"*, *"The Houston Chronicle's John McClain reported"*
          >
          > *"coach Todd McLellan"* *"Giants special teams coordinator Thomas McGaughey"*

    - Use them all but use a robust training method to overcome noise

  - [Impossible to] or cannot do well in situation where context is under-specified to identify label

    - e.g.

      - > *"According to PhotographyBlog, SanDisk and Lexar have no immediate plans to produce XQD or WiFi SD cards."* -- label */internet/website*

    - However, the result might be of good-quality

  - Impossibility to generalize:

    - e.g.

      - > *Many respondents felt their knowledge of issues relating to pain recognition, [anaesthesia] and analgesia in rabbits and guinea pigs was inadequate.* -- label: */medicine/symptom*

        - 检索结果都和医学相关，但是和症状关系差异很大
          - 是否加入label作为query之后会好一些？

      - Relationship between our method and zero-shot QA based method

  - Too strong string matching can help generalization or overfitting in downstream classifier tuning

    - 不同上下文的同一个entity的训练样例有助于这类别的识别还是有害于

    - e.g.

      - > *"When he left the Army, Spencer got a job in Bozeman, where he used acupuncture to save a dog that couldn't walk anymore."*



#### The benefit and problem of using mention embedding as query

- **Problems**
  - hard string matching, could ignore context and label information
  - $$\Rightarrow$$ 100% accurate pseudo annotation $$\Rightarrow$$ increase classifier performance?
  - 
- **Benefits**
  - Compound entity
    - xxx Hospital
    - xxx Award
    - Bachelor of Arts $$\Rightarrow$$ xxx of Arts Master of XXX

---

1. label info.
2. string matching：不使用mention
3. compound用mention embedding可能很好



---

### May 8 2022



- Ruili
  - 这周建好了index之后，试着从figer数据集的测试集(这个数据集的dev集是直接从训练集里采的，感觉不太合用)里随机采了一些example，用之前说的边界词向量和mention词向量两种方式对全部无标签数据检索了一下，取了前5的example大概看了一下，
  - 感觉两种方法都有点问题，用边界词检索出来的都是差不多一样的边界词
  - 用mention词的就几乎都是一样的mention
  - 有些情况下可能有关的global context就只是大的主题沾一点边，甚至几乎不相关
  - 一些具体的例子可以明天晚上展示一下
  - 把有标签数据集也放进去的检索现在还没试，目前的figer数据集的测试集太小了(只有500多个mention)，有些出现在类别ontology里的mention只出现过一次或者压根没出现，如果要做的话可能还得考虑一些别的有标签数据集(目前考虑ontonote的dev集和bbn的测试集，bbn也没有dev集



- 有几个问题：
  - 编码的策略和测试样例是一致的么？



- `x1...x[i..j]...xN`  $$\Rightarrow$$ `BERT(.)`
  - 首尾：`BERT(x)[i] concat BERT(x)[j]`
  - 平均：$$\frac{1}{j - i + 1}$$`sum( BERT(x)[i..j] )`
- 首尾、平均，能检索出语义相似的mention
