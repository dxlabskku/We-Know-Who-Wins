# We-Know-Who-Win

This repository contains the Pytorch implementation code for the paper "We Know Who Wins: Graph-oriented Approaches of Passing Networks for Predictive Football Match Outcomes"

## Architectures

![Model_Structure](model_structure.png)

Both the passing graph and in-game features of each team are input to the model. The passing graph passes through three convolution blocks consisting of graph convolution-dropout-elu layer to be pooled into graph embedding. The graph embeddings of both teams that have been fused with the in-game features are combined again to finally return the probability for each class.

## Dependencies

Please refer to [requirements.txt](requirements.txt) for package dependency to reproduce the code in this repository.

## Data

The data used in this study is from England's 1st division professional football league (Premier League) and 2nd division professional football league (Championship League) for the 20/21, 21/22, and 22/23 seasons, and consists of a total of 2835 games. All data was crawled from [Whoscored.com](https://1xbet.whoscored.com/), and the raw data is the HTML file of the web page for each match. The HTML files are shared through the link below due to size issues.

[Filelink](https://drive.google.com/drive/folders/1w2XSlFA7iWhVxeO2IGEC8JGbf-X7YHNc?usp=drive_link)

## Usage

You can run all steps from preprocessing the raw html data to testing new example through the following commands.

```python
python3 preprocess.py --gpu_num [YOUR GPU NUM] --pred_min [Prediction Minute]
```



