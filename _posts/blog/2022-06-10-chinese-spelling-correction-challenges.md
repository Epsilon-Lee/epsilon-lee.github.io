---
layout: post
title: "Chinese Spelling Correction: the challenges"
author: Guanlin Li
tag: blog
---



- toc
{:toc}
## An observation


Today, there are master students of my lab that are preparing for their thesis presentation. One biggest demands of them is to check false Chinese characters in their thesis. For example:

- from "国际电台苦名丰持人" to "国际电台著名主持人"

> This exmaple is borrowed from [FASPell](https://github.com/iqiyi/FASPell) from `iqiyi`, a Chinese IT company.

But when I test this example on the following online demo of Chinese Spell Correctors, they all failed.

- [pycorrector](https://huggingface.co/spaces/shibing624/pycorrector): the output is - `('国际电台苦名坚持人', [('丰', '坚', 6, 7)])`
- [sapling.ai](https://sapling.ai/lang/chinese): this tool tends to mark the grammatically incorrect collocations, such as marking the `只` in `一只街`, but it seems unable to correct wrong characters such as `丰持人`, which is a wierd phrase
- [Huolongguo](https://web.mypitaya.com/writing), the input `国际电台著名丰持人` can be changed to `国际电台著名主持人`, however, the input `中国北境市举办运动会` cannot be corrected to `北京市`, however `pycorrector` can (`('中国北京市举办运动会', [('境', '京', 3, 4)])`)

## Discussion

So, the performance of current Chinese spelling checking algorithms/models is very unsatisfactory. But what are the challenges?

The above test examples mainly focus on wrong Chinese character(s) in a Chinese phrase within a context. The following reference gives modelling methods and benchmarks for Chinese spelling correction.

One reason for my feeling of the bad performance of the online demo might be my expectation of real products. It is interesting to discuss the difference and connection between academic papers and real productions. When I find that `pycorrector` can achieve very high performance on benchmarks, I relieve a lot. However, when I first treat it as product, I think its over-confidence to change  `丰持人` to `坚持人` is unbearable.

So it is better to think of the interaction and presentation of model's prediction with/to end users. So what is a better way to fix this bias?

One solution might be first showing the model's confidence and results of detecting errors; and then the potential correction choices that might contain the user expected outcome.

> Maybe conformal prediction can used here as well.

## Reference

- [FASPell](https://github.com/iqiyi/FASPell), EMNLP Workshop on Noisy User-generated Text 2019.
- [Soft-Mased BERT](https://github.com/didi/ChineseNLP/blob/master/zh/docs/spell_correction.md), ACL 2020.
- [SpellGCN](https://github.com/ACL2020SpellGCN/SpellGCN), ACL 2020, `tensorflow`.
- [word-checker](https://github.com/houbb/word-checker), rule-based word checker.
- [ccheng16/correction](https://github.com/ccheng16/correction), n-gram KenLM based checker.