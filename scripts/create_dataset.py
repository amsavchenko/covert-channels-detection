import itertools
import pickle

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
import yaml

from calculate_statistics import *


config = yaml.safe_load(open('../config.yml'))

with open(config['paths']['overt'], 'rb') as f:
    overt = pickle.load(f)

with open(config['paths']['covert'], 'rb') as f:
    covert = pickle.load(f)

window_size = config['window_size']
step_size = config['step_size']


def calculate_statistics_for_all_traffic(stat_function, *args, **kwargs):
    return np.append(
        calculate_statistics(covert, stat_function, *args, **kwargs),
        calculate_statistics(overt, stat_function, *args, **kwargs)
    )


if __name__ == "__main__":

    data_len = int((len(covert) + 1 - window_size) / step_size)

    data = pd.DataFrame(index=[i for i in range(2 * data_len)])

    data['Covert'] = (np.append(np.ones(data_len), np.zeros(data_len))).astype(int)
    data['Mean'] = calculate_statistics_for_all_traffic(calculate_mean)
    data['Std'] = calculate_statistics_for_all_traffic(calculate_std)
    data['Median'] = calculate_statistics_for_all_traffic(calculate_median)
    data['Skew']  = calculate_statistics_for_all_traffic(calculate_skewness)
    data['Kurtosis'] = calculate_statistics_for_all_traffic(calculate_kurtosis)
    data['NumberUniqueValues'] = calculate_statistics_for_all_traffic(calculate_number_unique_values)
    data['ModeFreq'] = calculate_statistics_for_all_traffic(calculate_mode_frequency)

    min_value, max_value = config['ip_ctc'][0]['left'], config['ip_ctc'][1]['right']
    linspace = np.linspace(min_value, max_value, window_size)
    data['NumberModesKDE'] = calculate_statistics_for_all_traffic(calculate_number_modes_kde,
                                                                  linspace)

    data['NumberModesPareto'] = calculate_statistics_for_all_traffic(calculate_number_modes_pareto)

    n_values = [2, 3, 5, 10]
    for function_name, column_name in zip(
            [calculate_mean_nth, calculate_unique_nth, calculate_max_nth, calculate_min_nth, calculate_max_difference_nth],
            ['Mean', 'Unique', 'Max', 'Min', 'Max_difference']):
        for n in n_values:
            data[column_name + str(n) + 'th'] = calculate_statistics_for_all_traffic(function_name, n)

    data['Sum'] = calculate_statistics_for_all_traffic(calculate_sum)

    lags = [8, 16, 24, 40]
    for lag in lags:
        data[f'Autocorrelation_{lag}' ] = calculate_statistics_for_all_traffic(calculate_autocorrelation, lag)

    data['SumAutocorrCoeffs'] = calculate_statistics_for_all_traffic(calculate_sum_of_autocorrelation_coeffs, lags)
    data['MaxAutocorrCoeffs'] = calculate_statistics_for_all_traffic(calculate_max_of_autocorrelation_coeffs, lags)

    data['BerkMethod'] = calculate_statistics_for_all_traffic(calculate_berk_method)

    data['AverageDistWidth'] = calculate_statistics_for_all_traffic(calculate_average_distribution_width, linspace)

    data['RunsTest'] = calculate_statistics_for_all_traffic(calculate_runs_test)
    data['SignTest'] = calculate_statistics_for_all_traffic(calculate_sign_test)

    data['eSimilarity'] = calculate_statistics_for_all_traffic(calculate_e_similarity, 0.005)

    data['Entropy'] = calculate_statistics_for_all_traffic(calculate_entropy)
    data['ApproximateEntropy'] = calculate_statistics_for_all_traffic(calculate_approximate_entropy)
    data['Gini'] = calculate_statistics_for_all_traffic(calculate_gini)
    data['KolmogorovComplexity'] = calculate_statistics_for_all_traffic(calculate_kolmogorov_complexity)
    data['HurstExponent'] = calculate_statistics_for_all_traffic(calculate_hurst_exponent)

    diff_f = [np.std, skew, kurtosis]
    stat_f = [np.mean, np.std, np.median]

    diff_names = ['Std', 'Skew', 'Kurtosis']
    stat_names = ['Mean', 'Std', 'Median']

    functions = list(itertools.product(diff_f, stat_f))
    names = list(itertools.product(diff_names, stat_names))
    for i in range(len(functions)):

        data['RegularityTest' + "".join(names[i])] = calculate_statistics_for_all_traffic(
            calculate_regularity_test, window_size // 10, *functions[i])

    data['HaarCD1'] = calculate_statistics_for_all_traffic(calculate_energy_haar_cD1)

    data['ADFTest'] = calculate_statistics_for_all_traffic(calculate_adf_test)

    data['BenfordCorrelation'] = calculate_statistics_for_all_traffic(calculate_benford_correlation)

    for lag in lags:
        data[f'C3_{lag}'] = calculate_statistics_for_all_traffic(calculate_c3, lag)

    data['ComplexityEstimation'] = calculate_statistics_for_all_traffic(calculate_complexity_estimation)

    data['CountAbove35'] = calculate_statistics_for_all_traffic(calculate_count_above, 35)
    data['CountAbove43'] = calculate_statistics_for_all_traffic(calculate_count_above, 43)

    data['1stLocOfMaximum'] = calculate_statistics_for_all_traffic(calculate_first_location_of_maximum)
    data['1stLocOfMinimum'] = calculate_statistics_for_all_traffic(calculate_first_location_of_minimum)

    data['RatioBeyond2.5'] = calculate_statistics_for_all_traffic(calculate_ratio_beyond_r_sigma, 2.5)

    value_counts = [4, 6, 9, 43, 44]
    for value in value_counts:
        data[f'ValueCounts_{value}'] = calculate_statistics_for_all_traffic(calculate_value_counts, value)

    data['RMSE'] = calculate_statistics_for_all_traffic(calculate_rmse)

    data['MAE'] = calculate_statistics_for_all_traffic(calculate_mae)

    data['SymmetricTest'] = calculate_statistics_for_all_traffic(calculate_symmetric_test)

    data.to_csv(config['paths']['save_df'], index=False)