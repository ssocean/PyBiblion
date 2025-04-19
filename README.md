<p align="center">
  <img src="demo/icon.png" alt="icon" width="25%">
</p>

<h1 align="center">
  PyBiblion
</h1>


## Introduction

This repository contains the official code implementation for the Paper [A Literature Review of Literature Reviews in Pattern Analysis and Machine
Intelligence](https://arxiv.org/abs/2402.12928) ([arXiv](https://arxiv.org/abs/2402.12928)).Our goal is to provide a set of tools to assist researchers and scholars in conducting efficient and in-depth bibliometric analyses in the field.

Note: The term "biblion" is derived from the ancient Greek word "βιβλίον" (biblíon), means book or literature.

## Features

- **Meta-data Retrieval**: Automated tools for fetching literature data from multiple databases.
- **Extensibility**: The code is structured clearly, making it easy for other researchers to modify and extend according to their needs.

## News and Updates
* 2025.4.19 🔧 We have decoupled the project. Starting from this update, the main branch will retain only retrieval and metric-related functionalities. Additional features (such as plotting and other paper-related code) have been moved to the academic-support branch.
* 2024.11.27 🔧 We have implemented extensive fixes to the codebase, including code refactoring, cache optimization, and more.
* 2024.06.27 🔧 We identified a duplicated request bug in most GPT-related function. Besides, we fixed few errors in S2Paper class.
* 2024.04.25 🔥 We've integrated the popular Langchain (with OpenAI 1.0 version enabled) into our framework, making everything GPT-related run smoother and faster. 


## Installation

This project is implemented in Python. To get started, you need to install Python and some dependencies. 
In most cases, you just need to install the required Python packages according to the missing packages warnings.

Or you may install the required dependencies with the following command:

```bash
pip install -r requirements.txt
```

## Demo
You can calculate our metrics with only **one line of code**!!! Just kick off with 

```python
from retrievers.semantic_scholar_paper import S2paper
S2paper('INPUT TITLE HERE').TNCSI
```

See more details in lesson_101_demo.ipynb~.

## Citation
If you find this work useful for your research, please consider citing it as follows:
```
@article{zhao2024literature,
  title={A Literature Review of Literature Reviews in Pattern Analysis and Machine Intelligence},
  author={Zhao, Penghai and Zhang, Xin and Cheng, Ming-Ming and Yang, Jian and Li, Xiang},
  journal={arXiv preprint arXiv:2402.12928},
  year={2024}
}
```

## Acknowledgements
This project uses a few code snippets from the [litstudy](https://github.com/NLeSC/litstudy) project, which is aimed at facilitating literature study in scientific research. We are grateful to the authors of litstudy for their valuable contributions to the open-source community. 

