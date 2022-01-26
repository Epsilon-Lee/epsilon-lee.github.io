---
layout: post
title: "Recipes for Linux/Conda"
author: Guanlin Li
tag: notes
---

- toc
{:toc}

## Linux

#### Look up Linux Release Version

```bash
cat /etc/*release
lsb_release -a
```
#### Look up the disk space of the current directory

```bash
du -h --max-depth=1 $dir_path
df -h  # show the disk usage of the whole system
```

