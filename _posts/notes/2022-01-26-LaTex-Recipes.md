---
layout: post
title: "Recipes for LaTex"
author: Guanlin Li
tag: notes
---

- toc
{:toc}
## LaTex

- Make text in a tabular cell with `line break` (this [link](https://tex.stackexchange.com/questions/2441/how-to-add-a-forced-line-break-inside-a-table-cell))

```latex
\renewcommand{\cellalign/theadalign}{vh} % v: t, c, b; h: l, c, r
% e.g.
\usepackage{makecell}
\renewcommand\theadalign{bc}

% --- %
\thead{xxx//xxx}
```

