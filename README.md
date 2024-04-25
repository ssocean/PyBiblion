<p align="center">
  <img src="demo/icon.png" alt="icon" width="25%">
</p>

<h1 align="center">
  PyBiblion
</h1>

### [Early Access Version]
######  The [paper](https://arxiv.org/abs/2402.12928) are currently undergoing peer review, and the code may be adjusted at any time. If you have any issues about the paper or this repo, contact us via Email: oceanytech@gmail.com or issues as you prefer.

## Introduction

This repository contains the official code implementation for the Paper [A Literature Review of Literature Reviews in Pattern Analysis and Machine
Intelligence](https://arxiv.org/abs/2402.12928). Our goal is to provide a set of tools to assist researchers and scholars in conducting efficient and in-depth bibliometric analyses in the field.

Note: The term "biblion" is derived from the ancient Greek word "βιβλίον" (biblíon), means book or literature.

## Features

- **Meta-data Retrieval**: Automated tools for fetching literature data from multiple databases.
- **Data Analysis**: Offers a variety of literature data analysis tools, including but not limited to keyword extraction, and metric calculation.
- **Visualization**: Generates intuitive charts and diagrams to help understand the distribution and relationships within the literature data.
- **Extensibility**: The code is structured clearly, making it easy for other researchers to modify and extend according to their needs.

## News and Updates
* 2024.04.25 🔥 We've integrated the popular Langchain (with OpenAI 1.0 version enabled) into our framework, making everything GPT-related run smoother and faster. 
Now, you can calculate our metrics with only **one line of code**!!! Just kick off with `S2paper('INPUT TITLE HERE').TNCSI`.

## Installation

This project is implemented in Python. To get started, you need to install Python and some dependencies. 
In most cases, you just need to install the required Python packages according to the missing packages warnings.

Or you may install the required dependencies with the following command:

```bash
pip install -r requirements.txt
```

## Demo
You may check lesson_101_demo.ipynb for more details.

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
This project makes use of code from the [litstudy](https://github.com/NLeSC/litstudy) project, which is aimed at facilitating literature study in scientific research. We are grateful to the authors of litstudy for their valuable contributions to the open-source community. 

