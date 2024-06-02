This repository contains the official implementation of "CATS: Contextually-Aware Thresholding for Sparsity in Large Language Models" by Je-Yong Lee, Donghyun Lee, Genghan Zhang, Mo Tiwari, and Azalia Mirhoseini, as described in our paper on [arXiv](https://arxiv.org/abs/2404.08763).

## Overview
Our paper, "CATS: Contextually-Aware Thresholding for Sparsity in Large Language Models," introduces CATS—a new method aimed at reducing the computational demands of deploying LLMs without sacrificing their performance on downstream tasks. This method centers around a novel activation function that enhances activation sparsity effectively and efficiently.

The CATS approach can be applied to various base models such as Mistral-7B and Llama2-7B, demonstrating a minimal performance drop (within 1-2% of the base models) even at 50% activation sparsity levels. Importantly, CATS not only accelerates convergence but also integrates a custom GPU kernel that enhances inference speeds by approximately 15%.

## Reproducing Results

To reproduce the experimental results and figures presented in our work, please follow the steps outlined below. The process has been simplified into a single script to ensure ease of use and to maintain consistency across different environments.

### Prerequisites

Ensure you have the following prerequisites installed:
- Bash shell (Unix/Linux/Mac)
- Required Python packages (listed in `requirements.txt`)
- Set an `accelerate` configuration file based on your environment by running `accelerate config` 

### Steps

1. Open a terminal in the root directory of the project.
2. Run the following command:

```bash
bash reproduction_script.sh [path1] [path2]
```
- [path1]: Directory where the checkpoints for fine-tuned models will be stored.
- [path2]: Directory where the results of the experiments, such as figures and histograms, will be saved.

## Work in progress
We are currently developing a framework that will enable CATS to be easily integrated with any model from the HuggingFace library. 
