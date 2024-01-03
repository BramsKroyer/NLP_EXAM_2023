# NLP_EXAM_2023
This repository contains code used in the exam for the course Natural Language Processing at the MSc Cognitive Science programme at Aarhus University. Contributors: Klara Fomsgaard and Pernille Brams.

## Structure of repo
- /**nbs**: Notebooks used to drive functions in /src. Notebooks carry out everything from a) preprocessing of data, b) sentiment analysis, c) topic modelling, d) efforts to balance and downsample datasets, e) make visualisations, f) get responses from RAG and LLM models in Q&A tasks, and g) drive evaluation scripts.
- /**src**: Scripts and functions used in /nbs. Includes scripts to setup and build the RAGs and LLM used in the paper, along with evaluation scripts, utility functions and more.
- /**data**: A data folder containing data pertaining to The Synthetic News dataset used in the paper. The News Dataset consisting of real news articles could not be shared openly due to terms of use of the used sites from which articles were retrieved prohibiting redistribution.
- /**Rstats**: Markdowns containing code for the hierarchical Bayesian beta regressions fit in the paper to investigate effect of query type on sentiment of retrieved nodes in the RAG, along with visualisation code.
