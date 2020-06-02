"""Random ts-pymfe manual tests.

This test code is made pragmatically and therefore is not clean.
"""
import sys
import inspect

import numpy as np
import pandas as pd
import sklearn.metrics

from pymfe import _period
from pymfe import _detrend

from pymfe.general import MFETSGeneral
from pymfe.info_theory import MFETSInfoTheory
from pymfe.local_stats import MFETSLocalStats
from pymfe.global_stats import MFETSGlobalStats
from pymfe.autocorr import MFETSAutocorr
from pymfe.freq_domain import MFETSFreqDomain
from pymfe.landmarking import MFETSLandmarking
from pymfe.model_based import MFETSModelBased
from pymfe.stat_tests import MFETSStatTests
from pymfe.randomize import MFETSRandomize


def load_data(data_id: int, max_obs_num: int = 512) -> np.ndarray:
    data = pd.read_csv("data/comp-engine-export-sample.20200503.csv",
                       header=0,
                       index_col=0,
                       nrows=1,
                       skiprows=np.arange(1, data_id + 1),
                       squeeze=True,
                       low_memory=True)

    ts = np.asarray(data.values[0].split(","), dtype=float)[-max_obs_num:]

    return ts


def _test() -> None:
    if len(sys.argv) <= 3:
        print("usage:", sys.argv[0], "<data_id> <random_seed> <precomp 0/1>")
        sys.exit(1)

    data_id = int(sys.argv[1])
    random_state = int(sys.argv[2])
    precomp = bool(int(sys.argv[3]))

    if not 0 <= data_id < 20:
        print(f"Require 0 <= data_id < 20 (got {data_id}).")
        sys.exit(2)

    print("Chosen id:", data_id)
    print("Random_state:", random_state)

    ts = load_data(data_id)

    ts_period = _period.get_ts_period(ts)
    ts_trend, ts_season, ts_residuals = _detrend.decompose(ts,
                                                           ts_period=ts_period,
                                                           plot=True)

    ts_detrended = ts - ts_trend
    ts_deseasonalized = ts - ts_season

    score = sklearn.metrics.mean_squared_error

    components = {
        "ts": ts,
        "ts_trend": ts_trend,
        "ts_season": ts_residuals,
        "ts_residuals": ts_residuals,
        "ts_detrended": ts_detrended,
        "ts_deseasonalized": ts_deseasonalized,
        "random_state": random_state,
        "score": score,
    }
    initial_len = len(components)

    precomps = (
        MFETSGeneral.precompute_walker,
        MFETSGeneral.precompute_embed_caos_method,
        MFETSGeneral.precompute_period,
        MFETSGeneral.precompute_ts_scaled,
        MFETSFreqDomain.precompute_ps_residuals,
        MFETSGlobalStats.precompute_period,
        MFETSAutocorr.precompute_detrended_acf,
        MFETSAutocorr.precompute_gaussian_model,
        MFETSLocalStats.precompute_ts_scaled,
        MFETSLocalStats.precompute_rolling_window,
        MFETSModelBased.precompute_ts_scaled,
        MFETSModelBased.precompute_period,
        MFETSModelBased.precompute_model_ets,
        MFETSModelBased.precompute_ioe_std_linear_model,
        MFETSRandomize.precompute_ts_scaled,
        MFETSRandomize.precompute_itrand_stats,
        MFETSInfoTheory.precompute_ts_scaled,
        MFETSInfoTheory.precompute_detrended_ami,
    )

    methods = (
        MFETSGeneral.ft_emb_lag,
        MFETSGeneral.ft_stick_angles,
        MFETSGeneral.ft_fs_len,
        MFETSGeneral.ft_fnn_prop,
        MFETSGeneral.ft_embed_in_shell,
        MFETSGeneral.ft_force_potential,
        MFETSGeneral.ft_walker_cross_frac,
        MFETSGeneral.ft_walker_path,
        MFETSGeneral.ft_pred,
        MFETSGeneral.ft_moving_threshold,
        MFETSGeneral.ft_turning_points,
        MFETSGeneral.ft_step_changes,
        MFETSGeneral.ft_turning_points_trend,
        MFETSGeneral.ft_step_changes_trend,
        MFETSGeneral.ft_length,
        MFETSGeneral.ft_frac_cp,
        MFETSGeneral.ft_bin_mean,
        MFETSGeneral.ft_period,
        MFETSGeneral.ft_peak_frac,
        MFETSGeneral.ft_trough_frac,
        MFETSGeneral.ft_diff,
        MFETSGeneral.ft_cao_e1,
        MFETSGeneral.ft_cao_e2,
        MFETSGeneral.ft_emb_dim_cao,
        MFETSAutocorr.ft_gen_autocorr,
        MFETSAutocorr.ft_trev,
        MFETSAutocorr.ft_acf_first_nonsig,
        MFETSAutocorr.ft_acf_first_nonpos,
        MFETSAutocorr.ft_tc3,
        MFETSAutocorr.ft_autocorr_out_dist,
        MFETSAutocorr.ft_first_acf_locmin,
        MFETSAutocorr.ft_gresid_autocorr,
        MFETSAutocorr.ft_autocorr_crit_pt,
        MFETSAutocorr.ft_acf_detrended,
        MFETSAutocorr.ft_pacf,
        MFETSAutocorr.ft_acf,
        MFETSAutocorr.ft_acf_diff,
        MFETSAutocorr.ft_pacf_diff,
        MFETSAutocorr.ft_pacf_detrended,
        MFETSAutocorr.ft_gresid_lbtest,
        MFETSFreqDomain.ft_low_freq_power,
        MFETSFreqDomain.ft_ps_residuals,
        MFETSFreqDomain.ft_ps_freqs,
        MFETSFreqDomain.ft_ps_peaks,
        MFETSFreqDomain.ft_ps_entropy,
        MFETSGlobalStats.ft_dfa,
        MFETSGlobalStats.ft_corr_dim,
        MFETSGlobalStats.ft_ioe_tdelta_mean,
        MFETSGlobalStats.ft_t_mean,
        MFETSGlobalStats.ft_opt_boxcox_coef,
        MFETSGlobalStats.ft_sd_diff,
        MFETSGlobalStats.ft_sd_sdiff,
        MFETSGlobalStats.ft_skewness_diff,
        MFETSGlobalStats.ft_skewness_sdiff,
        MFETSGlobalStats.ft_kurtosis_diff,
        MFETSGlobalStats.ft_kurtosis_sdiff,
        MFETSGlobalStats.ft_exp_max_lyap,
        MFETSGlobalStats.ft_exp_hurst,
        MFETSGlobalStats.ft_skewness_residuals,
        MFETSGlobalStats.ft_kurtosis_residuals,
        MFETSGlobalStats.ft_sd_residuals,
        MFETSGlobalStats.ft_trend_strenght,
        MFETSGlobalStats.ft_season_strenght,
        MFETSGlobalStats.ft_spikiness,
        MFETSLocalStats.ft_moving_lilliefors,
        MFETSLocalStats.ft_moving_approx_ent,
        MFETSLocalStats.ft_moving_avg,
        MFETSLocalStats.ft_moving_avg_shift,
        MFETSLocalStats.ft_moving_var_shift,
        MFETSLocalStats.ft_moving_skewness_shift,
        MFETSLocalStats.ft_moving_kurtosis_shift,
        MFETSLocalStats.ft_moving_gmean_shift,
        MFETSLocalStats.ft_moving_sd_shift,
        MFETSLocalStats.ft_moving_acf_shift,
        MFETSLocalStats.ft_moving_kldiv_shift,
        MFETSLocalStats.ft_lumpiness,
        MFETSLocalStats.ft_stability,
        MFETSLocalStats.ft_moving_var,
        MFETSLocalStats.ft_moving_skewness,
        MFETSLocalStats.ft_moving_kurtosis,
        MFETSLocalStats.ft_moving_gmean,
        MFETSLocalStats.ft_moving_sd,
        MFETSLocalStats.ft_moving_acf,
        MFETSLocalStats.ft_moving_kldiv,
        MFETSLocalStats.ft_local_extrema,
        MFETSLocalStats.ft_local_range,
        MFETSModelBased.ft_avg_cycle_period,
        MFETSModelBased.ft_ioe_std_adj_r_sqr,
        MFETSModelBased.ft_ioe_std_slope,
        MFETSModelBased.ft_gaussian_r_sqr,
        MFETSModelBased.ft_linearity,
        MFETSModelBased.ft_curvature,
        MFETSModelBased.ft_des_level,
        MFETSModelBased.ft_des_slope,
        MFETSModelBased.ft_ets_level,
        MFETSModelBased.ft_ets_slope,
        MFETSModelBased.ft_ets_season,
        MFETSLandmarking.ft_model_linear_seasonal,
        MFETSLandmarking.ft_model_linear_embed,
        MFETSLandmarking.ft_model_exp,
        MFETSLandmarking.ft_model_sine,
        MFETSLandmarking.ft_model_loc_median,
        MFETSLandmarking.ft_model_loc_mean,
        MFETSLandmarking.ft_model_naive_seasonal,
        MFETSLandmarking.ft_model_naive_drift,
        MFETSLandmarking.ft_model_gaussian,
        MFETSLandmarking.ft_model_hwes_ada,
        MFETSLandmarking.ft_model_hwes_adm,
        MFETSLandmarking.ft_model_naive,
        MFETSLandmarking.ft_model_mean,
        MFETSLandmarking.ft_model_mean_acf_first_nonpos,
        MFETSLandmarking.ft_model_ses,
        MFETSLandmarking.ft_model_arima_100_c,
        MFETSLandmarking.ft_model_arima_010_c,
        MFETSLandmarking.ft_model_arima_110_c,
        MFETSLandmarking.ft_model_arima_011_nc,
        MFETSLandmarking.ft_model_arima_011_c,
        MFETSLandmarking.ft_model_arima_021_c,
        MFETSLandmarking.ft_model_arima_112_nc,
        MFETSLandmarking.ft_model_linear,
        MFETSLandmarking.ft_model_linear_acf_first_nonpos,
        MFETSRandomize.ft_resample_first_acf_nonpos,
        MFETSRandomize.ft_resample_first_acf_locmin,
        MFETSRandomize.ft_surr_tc3,
        MFETSRandomize.ft_surr_trev,
        MFETSRandomize.ft_itrand_mean,
        MFETSRandomize.ft_itrand_sd,
        MFETSRandomize.ft_itrand_acf,
        MFETSRandomize.ft_resample_std,
        MFETSStatTests.ft_test_lilliefors,
        MFETSStatTests.ft_test_lb,
        MFETSStatTests.ft_test_earch,
        MFETSStatTests.ft_test_adf,
        MFETSStatTests.ft_test_adf_gls,
        MFETSStatTests.ft_test_kpss,
        MFETSStatTests.ft_test_pp,
        MFETSStatTests.ft_test_dw,
        MFETSStatTests.ft_test_za,
        MFETSInfoTheory.ft_ami_detrended,
        MFETSInfoTheory.ft_ami,
        MFETSInfoTheory.ft_lz_complexity,
        MFETSInfoTheory.ft_sample_entropy,
        MFETSInfoTheory.ft_approx_entropy,
        MFETSInfoTheory.ft_control_entropy,
        MFETSInfoTheory.ft_surprise,
        MFETSInfoTheory.ft_ami_curvature,
        MFETSInfoTheory.ft_ami_first_critpt,
        MFETSInfoTheory.ft_hist_entropy,
        MFETSInfoTheory.ft_hist_ent_out_diff,
    )

    errors = []

    if precomp:
        for i, method in enumerate(precomps, 1):
            print(
                f"Precomputation method {i} of {len(precomps)}: {method.__name__}..."
            )

            params = inspect.signature(method).parameters.keys()
            component_names = frozenset(components.keys())
            intersec = component_names.intersection(params)

            args = {
                name: comp
                for name, comp in components.items() if name in intersec
            }
            print(3 * " ", f"Args {len(args)}: ", args.keys())

            try:
                res = method(**args)
                components.update(res)

            except Exception as ex:
                errors.append(("P", ex, method.__name__))

    component_names = frozenset(components.keys())

    for i, method in enumerate(methods, 1):
        print(f"method {i} of {len(methods)}: {method.__name__}...")

        sig = inspect.signature(method)
        params = sig.parameters.keys()
        intersec = component_names.intersection(params)

        args = {
            name: comp
            for name, comp in components.items() if name in intersec
        }
        print(3 * " ", f"Args {len(args)}: ", args.keys())

        try:
            res = method(**args)

            type_ = type(res)
            type_ = float if type_ is np.float64 else type_
            type_ = int if type_ is np.int64 else type_

            exp_ret_type = sig.return_annotation
            is_single_type = not hasattr(sig.return_annotation, "__args__")

            if type_ is not exp_ret_type and (is_single_type or type_
                                              not in exp_ret_type.__args__):
                raise TypeError(
                    f"Return ({res}) type {type(res)} does not conform to the return type ({sig.return_annotation})."
                )

        except Exception as ex:
            errors.append(("M", ex, method.__name__))

    if errors:
        for typ, err, method in errors:
            print(f"-> ({typ})", err, method)

    print("Time-series estimated period:", ts_period)
    print(f"Total of {len(errors)} exceptions raised.")
    print("Chosen id:", data_id, "Random_state:", random_state)
    print(f"Components got (total of {len(components) - initial_len} new):",
          len(components))


if __name__ == "__main__":
    _test()
