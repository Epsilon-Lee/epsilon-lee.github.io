- toc
{:toc}


### Changjiang Guo:

生成式文本摘要：

- 关键短语抽取
- 生成式文本摘要
- 现状：三类
  - 统计学、图模型、词嵌入
  - 关键短语抽取技术
    - TextRank算法（图模型）
    - 位置权重（第一次出现的位置，靠后圈子更小）
    - 如何eval？数据集：Inspec, SemEval、神策数据2018
  - generative摘要
    - Eval: LCSTS中文数据
  - 文摘系统设计实现
    - Flask
    - 微服务
- Q:
  - 低资源：不能这么说
  - 为什么关键短语有帮助
    - 主题怎么引入？LDA
    - 关键词怎么引入？
  - 图画得有点儿乱
  - 矩阵是不变的？对于不同文档？

### Chenyi Wang

- CSC
- 挑战？4个？
- 拼写检查，错别字？检查
  - BERT比？
  - BERT不是subword么？输入是字符序列？char
- 语法错误诊断
  - 四种：冗余、缺失、误用、乱序
  - ELECTRA/BERT+CRF
  - Eval: CGED数据
- Q：
  - 语速略快，有点儿小声
  - 感觉组织有点儿乱（讲得）
  - CSC的Recall这么呈现感觉有些奇怪

### Tianshu Liu

- 议论文生成
  - 数据集构建
    - 爬虫--人工标注（三千+）-- 二分类器 --
    - 例证数据集：
      - 题目、论据、论证三元组
        - 论据、论证的区别是？
  - 首尾段落生成
  - 检索匹配
- 关键词生成：
  - TextRank
- Q:
  - MSE loss? cossim?

