# Petfinder Kaggle Competition

## Introduction
- 2021년 글로벌창업사관학교 프로젝트 멘토링을 위해 참여한 Petfinder Competition입니다. 애완동물의 사진과 메타 데이터를 활용해 인기도를 예측하는 과제입니다. ([참조링크](https://www.kaggle.com/c/petfinder-pawpularity-score))

## Directory
~~~
├── EDA
│   └── tutorial-part-1-eda-for-beginners.ipynb 
├── ensemble_experiments
│   ├── 01 save-every-pred.ipynb
│   ├── 02 result-analysis.ipynb 
│   └── 03 ensemble analysis.ipynb 
├── head_param_search
│   ├── 01 - simplehead_swin_2021.py
│   ├── 02 - QRhead_swin_2021.py
│   ├── 03 - Bayeshead_swin_2021.py
|   └──...    
├── input
│   ├── petfinder-pawpularity-score
|   |    ├── train
|   |    └── test
│   ├── train.csv
│   └── test.csv
├── output
│   └── score_history.csv
├── train_source
│   ├── Petfinder_swintransformer.ipynb 
│   ├── Petfinder_swintransformer+QuantileRegression.ipynb 
│   └── Petfinder_swintransformer+BNN.ipynb 
└── README.md
~~~

## train_source
- Petfinder_swintransformer.ipynb : Train code for swin transformer with pytorch lightining 
- Petfinder_swintransformer+QuantileRegression.ipynb : Train code for swin transformer + Quantile Regression with pytorch lightning
- Petfinder_swintransformer+BNN : Train code for swin transformer + Bayesian neural network with pytorch lightning

### Reference
Swin Transformer : [링크](https://arxiv.org/abs/2103.14030)
Quantile Regression : [링크](https://www.aeaweb.org/articles?id=10.1257/jep.15.4.143)
BNN (Bayesian Neural Network) : [링크](https://arxiv.org/abs/1703.04977) 
