import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from scipy.stats import anderson, kstest, mannwhitneyu, shapiro, ttest_ind, ttest_rel, wilcoxon
from statsmodels.stats.contingency_tables import mcnemar
from termcolor import colored

SAMPLE_A_PATH = Path("outputs/eval/ema.json")
SAMPLE_B_PATH = Path("outputs/eval/no_ema.json")
OUTPUT_DIR = Path("outputs/compare/")
LOG_FILE = OUTPUT_DIR / "comparison.log"


logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.FileHandler(LOG_FILE, mode="w"), logging.StreamHandler()],
)

logging.getLogger("matplotlib.font_manager").disabled = True


def log_section(title):
    section_title = f"\n{'-'*21}\n {title.center(19)} \n{'-'*21}"
    logging.info(section_title)


def get_eval_info_episodes(eval_info_path: Path):
    with open(eval_info_path) as f:
        eval_info = json.load(f)

    stats = {
        "sum_rewards": [ep_stat["sum_reward"] for ep_stat in eval_info["per_episode"]],
        "max_rewards": [ep_stat["max_reward"] for ep_stat in eval_info["per_episode"]],
        "successes": [ep_stat["success"] for ep_stat in eval_info["per_episode"]],
        "seeds": [ep_stat["seed"] for ep_stat in eval_info["per_episode"]],
        "num_episodes": len(eval_info["per_episode"]),
    }
    return stats


def descriptive_stats(stats_a, stats_b, metric_name):
    a_mean, a_std = np.mean(stats_a[metric_name]), np.std(stats_a[metric_name])
    b_mean, b_std = np.mean(stats_b[metric_name]), np.std(stats_b[metric_name])
    logging.info(f"{metric_name} - Sample A: Mean = {a_mean:.3f}, Std Dev = {a_std:.3f}")
    logging.info(f"{metric_name} - Sample B: Mean = {b_mean:.3f}, Std Dev = {b_std:.3f}")


def cohens_d(x, y):
    return (np.mean(x) - np.mean(y)) / np.sqrt((np.std(x, ddof=1) ** 2 + np.std(y, ddof=1) ** 2) / 2)


def normality_tests(data, name):
    shapiro_stat, shapiro_p = shapiro(data)
    ks_stat, ks_p = kstest(data, "norm", args=(np.mean(data), np.std(data)))
    ad_stat = anderson(data)

    log_test(f"{name} - Shapiro-Wilk Test: statistic = {shapiro_stat:.3f}", shapiro_p)
    log_test(f"{name} - Kolmogorov-Smirnov Test: statistic = {ks_stat:.3f}", ks_p)
    logging.info(f"{name} - Anderson-Darling Test: statistic = {ad_stat.statistic:.3f}")
    for i in range(len(ad_stat.critical_values)):
        cv, sl = ad_stat.critical_values[i], ad_stat.significance_level[i]
        logging.info(f"    Critical value at {sl}%: {cv:.3f}")

    return shapiro_p > 0.05 and ks_p > 0.05


def plot_boxplot(data_a, data_b, labels, title, filename):
    plt.boxplot([data_a, data_b], labels=labels)
    plt.title(title)
    plt.savefig(filename)
    plt.close()


def plot_histogram(data_a, data_b, labels, title, filename):
    plt.hist(data_a, bins=30, alpha=0.7, label=labels[0])
    plt.hist(data_b, bins=30, alpha=0.7, label=labels[1])
    plt.title(title)
    plt.legend()
    plt.savefig(filename)
    plt.close()


def plot_qqplot(data, title, filename):
    stats.probplot(data, dist="norm", plot=plt)
    plt.title(title)
    plt.savefig(filename)
    plt.close()


def log_test(msg, p_value):
    if p_value < 0.01:
        color, interpretation = "red", "H_0 Rejected"
    elif 0.01 <= p_value < 0.05:
        color, interpretation = "orange", "Inconclusive"
        return "warning", "Inconclusive"
    else:
        color, interpretation = "green", "H_0 Not Rejected"
    logging.info(
        msg
        + ", p-value = "
        + colored(f"{p_value:.3f}", color)
        + " -> "
        + colored(f"{interpretation}", color, attrs=["bold"])
    )


def perform_paired_tests(stats_a, stats_b):
    max_rewards_a, max_rewards_b = np.array(stats_a["max_rewards"]), np.array(stats_b["max_rewards"])
    sum_rewards_a, sum_rewards_b = np.array(stats_a["sum_rewards"]), np.array(stats_b["sum_rewards"])
    successes_a, successes_b = np.array(stats_a["successes"]), np.array(stats_b["successes"])
    max_reward_diff = max_rewards_a - max_rewards_b
    sum_reward_diff = sum_rewards_a - sum_rewards_b

    log_section("Normality tests")
    normal_max_reward_diff = normality_tests(max_reward_diff, "Max Reward Difference")
    normal_sum_reward_diff = normality_tests(sum_reward_diff, "Sum Reward Difference")

    log_section("Paired-sample tests")
    if normal_max_reward_diff:
        t_stat_max_reward, p_val_max_reward = ttest_rel(max_rewards_a, max_rewards_b)
        log_test(f"Paired t-test for Max Reward: t-statistic = {t_stat_max_reward:.3f}", p_val_max_reward)
    else:
        w_stat_max_reward, p_wilcox_max_reward = wilcoxon(max_rewards_a, max_rewards_b)
        log_test(f"Wilcoxon test for Max Reward: statistic = {w_stat_max_reward:.3f}", p_wilcox_max_reward)

    if normal_sum_reward_diff:
        t_stat_sum_reward, p_val_sum_reward = ttest_rel(sum_rewards_a, sum_rewards_b)
        log_test(f"Paired t-test for Sum Reward: t-statistic = {t_stat_sum_reward:.3f}", p_val_sum_reward)
    else:
        w_stat_sum_reward, p_wilcox_sum_reward = wilcoxon(sum_rewards_a, sum_rewards_b)
        log_test(f"Wilcoxon test for Sum Reward: statistic = {w_stat_sum_reward:.3f}", p_wilcox_sum_reward)

    table = np.array(
        [
            [
                np.sum((successes_a == 1) & (successes_b == 1)),
                np.sum((successes_a == 1) & (successes_b == 0)),
            ],
            [
                np.sum((successes_a == 0) & (successes_b == 1)),
                np.sum((successes_a == 0) & (successes_b == 0)),
            ],
        ]
    )
    mcnemar_result = mcnemar(table, exact=True)
    log_test(f"McNemar's test for Success: statistic = {mcnemar_result.statistic:.3f}", mcnemar_result.pvalue)


def perform_independent_tests(stats_a, stats_b):
    max_rewards_a, max_rewards_b = np.array(stats_a["max_rewards"]), np.array(stats_b["max_rewards"])
    sum_rewards_a, sum_rewards_b = np.array(stats_a["sum_rewards"]), np.array(stats_b["sum_rewards"])

    log_section("Normality tests")
    normal_max_rewards_a = normality_tests(max_rewards_a, "Max Rewards Sample A")
    normal_max_rewards_b = normality_tests(max_rewards_b, "Max Rewards Sample B")
    normal_sum_rewards_a = normality_tests(sum_rewards_a, "Sum Rewards Sample A")
    normal_sum_rewards_b = normality_tests(sum_rewards_b, "Sum Rewards Sample B")

    log_section("Independent samples tests")
    if normal_max_rewards_a and normal_max_rewards_b:
        t_stat_max_reward, p_val_max_reward = ttest_ind(max_rewards_a, max_rewards_b, equal_var=False)
        log_test(f"Two-Sample t-test for Max Reward: t-statistic = {t_stat_max_reward:.3f}", p_val_max_reward)
    else:
        u_stat_max_reward, p_u_max_reward = mannwhitneyu(max_rewards_a, max_rewards_b)
        log_test(f"Mann-Whitney U test for Max Reward: U-statistic = {u_stat_max_reward:.3f}", p_u_max_reward)

    if normal_sum_rewards_a and normal_sum_rewards_b:
        t_stat_sum_reward, p_val_sum_reward = ttest_ind(sum_rewards_a, sum_rewards_b, equal_var=False)
        log_test(f"Two-Sample t-test for Sum Reward: t-statistic = {t_stat_sum_reward:.3f}", p_val_sum_reward)
    else:
        u_stat_sum_reward, p_u_sum_reward = mannwhitneyu(sum_rewards_a, sum_rewards_b)
        log_test(f"Mann-Whitney U test for Sum Reward: U-statistic = {u_stat_sum_reward:.3f}", p_u_sum_reward)


def perform_tests(sample_a_stats, sample_b_stats):
    log_section("Descriptive Stats")
    logging.info(f"Number of episode - sample A: {sample_a_stats['num_episodes']}")
    logging.info(f"Number of episode - sample B: {sample_b_stats['num_episodes']}")

    seeds_a, seeds_b = sample_a_stats["seeds"], sample_b_stats["seeds"]
    if seeds_a == seeds_b:
        logging.info("Samples are paired (identical seeds).")
        paired = True
    else:
        logging.info("Samples are considered independant (seeds are different).")
        paired = False

    descriptive_stats(sample_a_stats, sample_b_stats, "successes")
    descriptive_stats(sample_a_stats, sample_b_stats, "max_rewards")
    descriptive_stats(sample_a_stats, sample_b_stats, "sum_rewards")

    log_section("Effect Size")
    d_max_reward = cohens_d(sample_a_stats["max_rewards"], sample_b_stats["max_rewards"])
    d_sum_reward = cohens_d(sample_a_stats["sum_rewards"], sample_b_stats["sum_rewards"])
    logging.info(f"Cohen's d for Max Reward: {d_max_reward:.3f}")
    logging.info(f"Cohen's d for Sum Reward: {d_sum_reward:.3f}")

    if paired:
        perform_paired_tests(sample_a_stats, sample_b_stats)
    else:
        perform_independent_tests(sample_a_stats, sample_b_stats)

    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    plot_boxplot(
        sample_a_stats["max_rewards"],
        sample_b_stats["max_rewards"],
        ["Sample A Max Reward", "Sample B Max Reward"],
        "Boxplot of Max Rewards",
        f"{OUTPUT_DIR}/boxplot_max_reward.png",
    )
    plot_boxplot(
        sample_a_stats["sum_rewards"],
        sample_b_stats["sum_rewards"],
        ["Sample A Sum Reward", "Sample B Sum Reward"],
        "Boxplot of Sum Rewards",
        f"{OUTPUT_DIR}/boxplot_sum_reward.png",
    )

    plot_histogram(
        sample_a_stats["max_rewards"],
        sample_b_stats["max_rewards"],
        ["Sample A Max Reward", "Sample B Max Reward"],
        "Histogram of Max Rewards",
        f"{OUTPUT_DIR}/histogram_max_reward.png",
    )
    plot_histogram(
        sample_a_stats["sum_rewards"],
        sample_b_stats["sum_rewards"],
        ["Sample A Sum Reward", "Sample B Sum Reward"],
        "Histogram of Sum Rewards",
        f"{OUTPUT_DIR}/histogram_sum_reward.png",
    )

    plot_qqplot(
        sample_a_stats["max_rewards"],
        "Q-Q Plot of Sample A Max Rewards",
        f"{OUTPUT_DIR}/qqplot_sample_a_max_reward.png",
    )
    plot_qqplot(
        sample_b_stats["max_rewards"],
        "Q-Q Plot of Sample B Max Rewards",
        f"{OUTPUT_DIR}/qqplot_sample_b_max_reward.png",
    )
    plot_qqplot(
        sample_a_stats["sum_rewards"],
        "Q-Q Plot of Sample A Sum Rewards",
        f"{OUTPUT_DIR}/qqplot_sample_a_sum_reward.png",
    )
    plot_qqplot(
        sample_b_stats["sum_rewards"],
        "Q-Q Plot of Sample B Sum Rewards",
        f"{OUTPUT_DIR}/qqplot_sample_b_sum_reward.png",
    )


if __name__ == "__main__":
    sample_a_stats = get_eval_info_episodes(SAMPLE_A_PATH)
    sample_b_stats = get_eval_info_episodes(SAMPLE_B_PATH)

    perform_tests(sample_a_stats, sample_b_stats)
