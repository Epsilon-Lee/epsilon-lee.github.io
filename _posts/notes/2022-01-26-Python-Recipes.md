---
layout: post
title: "Python Recipes"
author: Guanlin Li
tag: notes
---

- toc
{:toc}
## Python

### String manipulation

#### Replace sub string in a str

```python
# substr must be in `s`
s.replace(substr, new_substr)
```

### Executing command lines in `python` program

```python
# refer to https://www.section.io/engineering-education/how-to-execute-linux-commands-in-python/
## METHOD 1
import os
cmd = 'ls -l'
os.system(cmd)

## METHOD 2: using subprocess
### subprocess module allows you to spawn new processes, connect to their input/output/error pipes, and obtain their return codes.
import subprocess
cmd = 'ping', server = 'www.baidu.com'
tmp = subprocess.Popen([cmd, '-c 1', server], stdout=subprocess.PIPE)
output = str(tmp.communitate())
```

- if you don’t want to dump a large output onto the terminal, you can use `subprocess.PIPE` to send the output of one command to the next. This corresponds to the `|` option in Linux.

