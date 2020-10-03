# ts-pymfe: meta-feature extractor for one-dimensional time-series
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

# Available meta-features by group
Below I present the full list of available meta-features in this package separated by meta-feature group. Also note that you can use the following methods to recover the available meta-feature, groups, and summary functions:
```python
import pymfe.tsmfe

groups = pymfe.tsmfe.TSMFE.valid_groups()
print(groups)

metafeatures = pymfe.tsmfe.TSMFE.valid_metafeatures()
print(metafeatures)

summaries = pymfe.tsmfe.TSMFE.valid_summary()
print(summaries)
```

## landmarking:
1. model_arima_010_c
2. model_arima_011_c
3. model_arima_011_nc
4. model_arima_021_c
5. model_arima_100_c
6. model_arima_110_c
7. model_arima_112_nc
8. model_exp
9. model_gaussian
10. model_hwes_ada
11. model_hwes_adm
12. model_linear
13. model_linear_acf_first_nonpos
14. model_linear_embed
15. model_linear_seasonal
16. model_loc_mean
17. model_loc_median
18. model_mean
19. model_mean_acf_first_nonpos
20. model_naive
21. model_naive_drift
22. model_naive_seasonal
23. model_ses
24. model_sine

## general:
1. bin_mean
2. cao_e1
3. cao_e2
4. diff
5. emb_dim_cao
6. emb_lag
7. embed_in_shell
8. fnn_prop
9. force_potential
10. frac_cp
11. fs_len
12. length
13. moving_threshold
14. peak_frac
15. period
16. pred
17. step_changes
18. step_changes_trend
19. stick_angles
20. trough_frac
21. turning_points
22. turning_points_trend
23. walker_cross_frac
24. walker_path

## global-stat:
1. corr_dim
2. dfa
3. exp_hurst
4. exp_max_lyap
5. ioe_tdelta_mean
6. kurtosis_diff
7. kurtosis_residuals
8. kurtosis_sdiff
9. opt_boxcox_coef
10. sd_diff
11. sd_residuals
12. sd_sdiff
13. season_strenght
14. skewness_diff
15. skewness_residuals
16. skewness_sdiff
17. spikiness
18. t_mean
19. trend_strenght

## local-stat:
1. local_extrema
2. local_range
3. lumpiness
4. moving_acf
5. moving_acf_shift
6. moving_approx_ent
7. moving_avg
8. moving_avg_shift
9. moving_gmean
10. moving_gmean_shift
11. moving_kldiv
12. moving_kldiv_shift
13. moving_kurtosis
14. moving_kurtosis_shift
15. moving_lilliefors
16. moving_sd
17. moving_sd_shift
18. moving_skewness
19. moving_skewness_shift
20. moving_var
21. moving_var_shift
22. stability

## model-based:
1. avg_cycle_period
2. curvature
3. des_level
4. des_slope
5. ets_level
6. ets_season
7. ets_slope
8. gaussian_r_sqr
9. ioe_std_adj_r_sqr
10. ioe_std_slope
11. linearity

## info-theory:
1. low_freq_power
2. ps_entropy
3. ps_freqs
4. ps_peaks
5. ps_residuals

## stat-tests:
1. test_adf
2. test_adf_gls
3. test_dw
4. test_earch
5. test_kpss
6. test_lb
7. test_lilliefors
8. test_pp
9. test_za

## autocorr:
1. acf
2. acf_detrended
3. acf_diff
4. acf_first_nonpos
5. acf_first_nonsig
6. autocorr_crit_pt
7. autocorr_out_dist
8. first_acf_locmin
9. gen_autocorr
10. gresid_autocorr
11. gresid_lbtest
12. pacf
13. pacf_detrended
14. pacf_diff
15. tc3
16. trev

## randomize:
1. itrand_acf
2. itrand_mean
3. itrand_sd
4. resample_first_acf_locmin
5. resample_first_acf_nonpos
6. resample_std
7. surr_tc3
8. surr_trev

## freq-domain:
1. ami
2. ami_curvature
3. ami_detrended
4. ami_first_critpt
5. approx_entropy
6. control_entropy
7. hist_ent_out_diff
8. hist_entropy
9. lz_complexity
10. sample_entropy
11. surprise

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
