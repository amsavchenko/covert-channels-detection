from hurst import compute_Hc
import nolds
import numpy as np
import pywt
from scipy.signal import find_peaks, peak_widths
from scipy.stats import skew, kurtosis, norm
from sklearn.neighbors import KernelDensity
from statsmodels.tsa.stattools import adfuller
from tsfresh.feature_extraction import feature_calculators
import yaml
import zlib


config = yaml.safe_load(open('../config.yml'))
window_size = config['window_size']
step_size = config['step_size']


def calculate_statistics(traffic, stat_function, *args, window_size=window_size, step_size=step_size, precision=3,
                         **kwargs):
    stat_list = []

    for right_bound in range(window_size, len(traffic) + 1, step_size):
        stat = stat_function(traffic[right_bound - window_size:right_bound], *args, **kwargs)
        stat_list.append(round(stat, precision))

    return stat_list


def calculate_mean(traffic):
    return np.mean(traffic)


def calculate_std(traffic):
    return np.std(traffic, ddof=1)


def calculate_median(traffic):
    return np.median(traffic)


def calculate_skewness(traffic):
    return skew(traffic)


def calculate_kurtosis(traffic):
    return kurtosis(traffic)


def calculate_number_unique_values(traffic):
    traffic = [round(interval) for interval in traffic]
    return len(set(traffic))


def calculate_mode_frequency(traffic):
    traffic = [round(interval) for interval in traffic]
    mode = max(set(traffic), key=traffic.count)
    frequency = traffic.count(mode) / len(traffic)
    return frequency


def calculate_number_modes_kde(traffic, linspace, bandwidth=2.5):
    linspace = linspace.reshape(-1, 1)
    traffic = np.array(traffic).reshape(-1, 1)
    kde = KernelDensity(bandwidth=bandwidth).fit(traffic)
    logprob = kde.score_samples(linspace)
    modes_number = len(find_peaks(np.exp(logprob))[0])
    return modes_number


def calculate_number_modes_pareto(traffic):
    traffic = [round(interval) for interval in traffic]
    frequencies = {interval: traffic.count(interval) for interval in set(traffic)}
    sorted_frequencies = {interval: count for interval, count in
                          sorted(frequencies.items(), key=lambda item: item[1], reverse=True)}

    count = 0
    keys = list(sorted_frequencies.keys())
    values = list(sorted_frequencies.values())

    for j in range(len(keys)):
        count += values[j]
        if count > window_size // 2:
            break

    max_difference = 0
    symbols_number = 0
    for i in range(j):
        difference = values[i] - values[i + 1]
        if difference > max_difference:
            max_difference = difference
            symbols_number = i
    return symbols_number


def calculate_mean_nth(traffic, n):
    return np.mean([traffic[i] for i in range(len(traffic)) if i % n == 0])


def calculate_unique_nth(traffic, n):
    traffic = [round(interval) for interval in traffic]
    return len(set([traffic[i] for i in range(len(traffic)) if i % n == 0]))


def calculate_max_nth(traffic, n):
    traffic = [traffic[i] for i in range(len(traffic)) if i % n == 0]
    return max(traffic)


def calculate_min_nth(traffic, n):
    traffic = [traffic[i] for i in range(len(traffic)) if i % n == 0]
    return min(traffic)


def calculate_max_difference_nth(traffic, n):
    traffic = [traffic[i] for i in range(len(traffic)) if i % n == 0]
    return max(traffic) - min(traffic)


def calculate_sum(traffic):
    return sum(traffic)


def calculate_autocorrelation(traffic, lag):
    mean = np.mean(traffic)
    autocorrelation = 0

    for i in range(len(traffic) - lag):
        autocorrelation += (traffic[i] - mean) * (traffic[i + lag] - mean)

    return autocorrelation / sum([(element - mean) ** 2 for element in traffic])


def calculate_sum_of_autocorrelation_coeffs(traffic, lags=[i for i in range(1, 11)]):
    sum_of_autocorr_coeffs = 0
    for lag in lags:
        sum_of_autocorr_coeffs += np.abs(calculate_autocorrelation(traffic, lag))
    return sum_of_autocorr_coeffs / len(lags)


def calculate_max_of_autocorrelation_coeffs(traffic, lags=[i for i in range(1, 33)]):
    max_autocorrelation = None

    for lag in lags:
        autocorrelation = calculate_autocorrelation(traffic, lag)
        if max_autocorrelation is None or np.abs(autocorrelation) > np.abs(max_autocorrelation):
            max_autocorrelation = autocorrelation

    return max_autocorrelation


def calculate_berk_method(traffic):
    traffic = [round(element) for element in traffic]
    max_frequency = 0

    for element in set(traffic):
        max_frequency = max(max_frequency, traffic.count(element))

    probability = 1 - (traffic.count(round(np.mean(traffic))) / max_frequency)
    return probability


def calculate_average_distribution_width(traffic, linspace, bandwidth=2.5):
    linspace = linspace.reshape(-1, 1)
    traffic = np.array(traffic).reshape(-1, 1)
    kde = KernelDensity(bandwidth=bandwidth).fit(traffic)
    probabilities = np.exp(kde.score_samples(linspace))
    peaks = find_peaks(probabilities)[0]
    widths = peak_widths(probabilities, peaks, rel_height=1.0)[0]
    return np.mean(widths)


def calculate_runs_test(traffic):
    signs = np.sign(np.array(traffic) - np.mean(traffic))
    runs_number = 0
    for i in range(len(signs) - 1):
        if signs[i] != signs[i + 1]:
            runs_number += 1
    runs_number += 1

    n_positive = sum(signs > 0)
    n_negative = sum(signs < 0)
    mean = (2 * n_positive * n_negative) / (n_positive + n_negative) + 1
    variance = ((mean - 1) * (mean - 2)) / (n_positive + n_negative - 1)

    z_value = (runs_number - mean) / np.sqrt(variance)
    return np.abs(z_value)


def calculate_sign_test(traffic, p0=0.5):
    subtracted_pairs = []
    for i in range(1, len(traffic)):
        subtracted_pairs.append(traffic[i] - traffic[i - 1])

    n_positive = sum(np.array(subtracted_pairs) > 0) / len(subtracted_pairs)
    z_value = (n_positive - p0) / np.sqrt((p0 * (1 - p0)) / len(subtracted_pairs))
    return z_value


def calculate_e_similarity(traffic, epsilon):
    differences = []
    traffic = sorted(traffic)

    for i in range(len(traffic) - 1):
        difference = traffic[i + 1] / (traffic[i] + 1e-5) - 1
        differences.append(difference)

    similarity_score = (np.array(differences) < epsilon).sum() / (len(traffic) - 1)
    return similarity_score


def calculate_entropy(traffic):
    traffic = [round(element) for element in traffic]
    entropy = 0

    for interval in set(traffic):
        probability = traffic.count(interval) / len(traffic)
        entropy += probability * np.log2(probability)

    return entropy * (-1)


def calculate_approximate_entropy(traffic):
    return nolds.sampen(traffic)


def calculate_gini(traffic):
    traffic = np.array([round(interval) for interval in traffic])
    probabilities = [np.mean(traffic == n) for n in set(traffic)]
    return sum([prob * (1 - prob) for prob in probabilities])


def calculate_kolmogorov_complexity(traffic):
    traffic = [str(round(interval)) for interval in traffic]
    traffic_str = ",".join(traffic).encode('ascii')
    compressed_len = len(zlib.compress(traffic_str))
    return compressed_len / len(traffic_str)


def calculate_hurst_exponent(traffic):
    hurst_exponent, _, _ = compute_Hc(traffic)
    return hurst_exponent


def calculate_regularity_test(traffic, small_window_size, diff_stat_function, res_stat_function):
    """
    Calculate different regularity tests.

    The traffic separates into non-ovelapping windows of size small_window_size and
    statistics calculates changing 2 functions:
    diff_stat_function: function used when calculating differences between windows of small_window_size
    res_stat_function: function used for final calculation of resulted statistic for all windows

    Base idea: Cabuk 2004
    Addition:  Archibald 2014 (for different diff and res functions)
    """
    statistics, differences = [], []

    for left_bound in range(0, len(traffic), small_window_size):
        window = traffic[left_bound:left_bound + small_window_size]
        statistics.append(diff_stat_function(window))

    for i in range(len(statistics)):
        for j in range(i + 1, len(statistics)):
            difference = np.abs(statistics[i] - statistics[j]) / statistics[i]
            differences.append(difference)

    return res_stat_function(differences)


def calculate_energy_haar_cD1(traffic):
    """ Mou 2012 """
    cA3, cD3, cD2, cD1 = pywt.wavedec(traffic, 'haar', level=3)

    for coeffs_list in [cA3, cD3, cD2, cD1]:
        coeffs_list = [element ** 2 for element in coeffs_list]

    return sum(cD1) / (sum(cD3) + sum(cD2) + sum(cD1) + sum(cA3))


def calculate_adf_test(traffic):
    """ Augmented Dickey–Fuller test """
    return adfuller(traffic, regression='nc')[0]


def calculate_benford_correlation(traffic):
    traffic = np.array([int(str(interval)[:1]) for interval in traffic])

    benford_distribution = np.array([np.log10(1 + 1 / i) for i in range(1, 10)])
    traffic_distribution = np.array([(traffic == i).mean() for i in range(1, 10)])

    correlation = np.corrcoef(benford_distribution, traffic_distribution)[0, 1]
    return correlation


def calculate_c3(traffic, lag):
    return feature_calculators.c3(traffic, lag)


def calculate_complexity_estimation(traffic):
    return feature_calculators.cid_ce(traffic, normalize=True)


def calculate_count_above(traffic, value_above):
    return feature_calculators.count_above(np.array(traffic), value_above)


def calculate_first_location_of_maximum(traffic):
    return feature_calculators.first_location_of_maximum(traffic)


def calculate_first_location_of_minimum(traffic):
    return feature_calculators.first_location_of_minimum(traffic)


def calculate_ratio_beyond_r_sigma(traffic, r):
    return feature_calculators.ratio_beyond_r_sigma(traffic, r)


def calculate_value_counts(traffic, value):
    traffic = [round(interval) for interval in traffic]
    return feature_calculators.value_count(traffic, value)


def calculate_rmse(traffic):
    mean = np.mean(traffic)
    return np.sqrt(np.mean([(interval - mean) ** 2 for interval in traffic]))


def calculate_mae(traffic):
    median = np.median(traffic)
    return np.mean([np.abs(interval - median) for interval in traffic])


def calculate_symmetric_test(traffic):
    return max(traffic) - min(traffic) - np.abs(np.mean(traffic) - np.median(traffic))
