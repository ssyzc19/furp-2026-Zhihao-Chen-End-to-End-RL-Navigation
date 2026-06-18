*首先在anaconda prompt进入miniconda amr环境
(base) PS C:\Users\user> conda activate amr
(amr) PS C:\Users\user> python
Python 3.10.20 | packaged by Anaconda, Inc. | (main, Mar 11 2026, 17:42:35) [MSC v.1942 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license" for more information.
*在miniconda amr环境中也就是在python环境下进入pytorch
>>>import torch
print(torch.__version__)
