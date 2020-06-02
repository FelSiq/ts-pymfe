# ts-pymfe
A backup for the pymfe expansion for time-series data. Currently, this repository contains the methods for meta-feature extraction and an modified pymfe core to run extract the meta-features.

Please note that tspymfe is not intended to be a stand-alone package, and will be oficially merged (hopefully soon) to the original Pymfe package. Until then, this package is available as a beta version.

There is 149 distinct metafeature extraction methods in this version, distributed in the following groups:

1. General
2. Local statistics
3. Global statistics
4. Statistical tests
5. Autocorrelation
6. Frequency domain
7. Information theory
8. Randomize
9. Landmarking
10. Model based

## Install
From pip:
```
pip install -U tspymfe
```
or:
```
python3 -m pip install -U tspymfe
```

## Usage
To extract the meta-features, the API behaves pretty much like the original Pymfe API:
```python
import pymfe.tsmfe
import numpy as np

# random time-series
ts = 0.3 * np.arange(100) + np.random.randn(100)

extractor = pymfe.tsmfe.TSMFE()
extractor.fit(ts)
res = extractor.extract()

print(res)
```

## Dev-install
If you downloaded directly from github, install the required packages using:
```
pip install -Ur requirements.txt
```

You can run some test scripts:
```
python test_a.py <data_id> <random_seed> <precomp 0/1>
python test_b.py <data_id> <random_seed> <precomp 0/1>
```
Where the first argument is the test time-series id (check [data/comp-engine-export-sample.20200503.csv](https://github.com/FelSiq/ts-pymfe/tree/master/data) file.) and must be between 0 (inclusive) and 19 (also inclusive), the random seed must be an integer, and precomp is a boolean argument ('0' or '1') to activate the precomputation methods, used to calculate common values between various methods and, therefore, speed the main computations.

Example:
```
python test_a.py 0 16 1
python test_b.py 0 16 1
```

The code format style is checked using flake8, pylint and mypy. You can use the Makefile to run all verifications by yourself:
```
pip install -Ur requirements-dev.txt
make code-check
```

# Main references
## Papers
1. [T.S. Talagala, R.J. Hyndman and G. Athanasopoulos. Meta-learning how to forecast time series (2018).](https://www.monash.edu/business/econometrics-and-business-statistics/research/publications/ebs/wp06-2018.pdf).
2. [Kang, Yanfei., Hyndman, R.J., & Smith-Miles, Kate. (2016). Visualising Forecasting Algorithm Performance using Time Series Instance Spaces (Department of Econometrics and Business Statistics Working Paper Series 10/16).](https://www.monash.edu/business/ebs/research/publications/ebs/wp10-16.pdf)
3. [C. Lemke, and B. Gabrys. Meta-learning for time series forecasting and forecast combination (Neurocomputing
Volume 73, Issues 10–12, June 2010, Pages 2006-2016)](https://www.sciencedirect.com/science/article/abs/pii/S0925231210001074)
4. [B.D. Fulcher and N.S. Jones. hctsa: A computational framework for automated time-series phenotyping using massive feature extraction. Cell Systems 5, 527 (2017).][1]
5. [B.D. Fulcher, M.A. Little, N.S. Jones. Highly comparative time-series analysis: the empirical structure of time series and their methods. J. Roy. Soc. Interface 10, 83 (2013).](https://royalsocietypublishing.org/doi/full/10.1098/rsif.2013.0048)


## Books
1. [Hyndman, R.J., & Athanasopoulos, G. (2018) Forecasting: principles and practice, 2nd edition, OTexts: Melbourne, Australia. OTexts.com/fpp2. Accessed on April 29 2020.](https://otexts.com/fpp2/)


## Packages
1. [tsfeatures (R language)](https://github.com/robjhyndman/tsfeatures)
2. [hctsa (Matlab language)](https://github.com/benfulcher/hctsa)

[1]: https://www.cell.com/cell-systems/fulltext/S2405-4712(17)30438-6

## Data
Data sampled from: https://comp-engine.org/
