# Subspace Selection to Suppress Confounding Source Domain Information in AAM Transfer Learning


## Table of Contents 
- [Introduction](#introduction) 
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Evaluation](#evaluation)
- [Contact](#contact)


## Introduction

This repository contains the implementation of the transfer learning method posposed in 

**Subspace Selection to Suppress Confounding Source Domain Information in AAM Transfer Learning**    
Azin Asgarian, Ahmed Bilal Ashraf, David Fleet, Babak Taati   
IJCB 2017, preprint [arXiv:1708.08508](https://arxiv.org/abs/1708.08508)   

This implementation is built on [Menpo Project](https://github.com/menpo).

## Prerequisites
- Python 2.7
- numpy
- scipy
- menpo
- menpofit

## Installation


Please make sure all prerequisites including [menpo](https://github.com/menpo/menpo) and [menpofit](https://github.com/menpo/menpofit) are installed. For instaling menpo and menpofit if you already have [Conda](https://conda.io/miniconda.html) installed on your computer, you only need to run the following codes:   

```
conda create -n myenv python=2.7
source activate myenv
conda install -c menpo menpo
conda install -c menpo menpofit
``` 
   
For installing Menpo and Menpofit without Conda, please see [Menpo installation guide](http://www.menpo.org/installation/).

## Evaluation
Once you have all the prerequisites installed, you can run each method separately and evaluate their performance.


## Contact

In case you found anything problematic or if you have any comments and suggestions, please let me know. I'm happy to hear the issues and I'd appreciate your comments. My email address is azinasg@cs.toronto.edu.
