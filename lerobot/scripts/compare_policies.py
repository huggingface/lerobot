"""Compare two policies on based on metrics computed from an eval.

Usage example:

You just made changes to a policy and you want to assess its new performance against
the reference policy (i.e. before your changes).

```
python lerobot/scripts/compare_policies.py \
    output/eval/ref_policy/eval_info.json \
    output/eval/new_policy/eval_info.json
```

This script can accept `eval_info.json` dicts with identical seeds between each eval episode of ref_policy and new_policy
(paired-samples) or from evals performed with different seeds (independent samples).
"""

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from scipy.stats import anderson, kstest, mannwhitneyu, shapiro, ttest_ind, ttest_rel, wilcoxon
from statsmodels.stats.contingency_tables import mcnemar
from termcolor import colored


def init_logging(output_dir: Path) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[logging.StreamHandler()],
    )
    logging.getLogger("matplotlib.font_manager").disabled = True


def log_section(title: str) -> None:
    section_title = f"\n{'-'*21}\n {title.center(19)} \n{'-'*21}"
    logging.info(section_title)


def get_eval_info_episodes(eval_info_path: Path) -> dict:
    with open(eval_info_path) as f:
        eval_info = json.load(f)

    return {
        "sum_rewards": np.array([ep_stat["sum_reward"] for ep_stat in eval_info["per_episode"]]),
        "max_rewards": np.array([ep_stat["max_reward"] for ep_stat in eval_info["per_episode"]]),
        "successes": np.array([ep_stat["success"] for ep_stat in eval_info["per_episode"]]),
        "seeds": [ep_stat["seed"] for ep_stat in eval_info["per_episode"]],
        "num_episodes": len(eval_info["per_episode"]),
    }


def descriptive_stats(ref_sample: dict, new_sample: dict, metric_name: str):
    ref_mean, ref_std = np.mean(ref_sample[metric_name]), np.std(ref_sample[metric_name])
    new_mean, new_std = np.mean(new_sample[metric_name]), np.std(new_sample[metric_name])
    logging.info(f"{metric_name} - Ref sample: mean = {ref_mean:.3f}, std = {ref_std:.3f}")
    logging.info(f"{metric_name} - New sample: mean = {new_mean:.3f}, std = {new_std:.3f}")


def cohens_d(x, y):
    return (np.mean(x) - np.mean(y)) / np.sqrt((np.std(x, ddof=1) ** 2 + np.std(y, ddof=1) ** 2) / 2)


def normality_tests(array: np.ndarray, name: str):
    shapiro_stat, shapiro_p = shapiro(array)
    ks_stat, ks_p = kstest(array, "norm", args=(np.mean(array), np.std(array)))
    ad_stat = anderson(array)

    log_test(f"{name} - Shapiro-Wilk Test: statistic = {shapiro_stat:.3f}", shapiro_p)
    log_test(f"{name} - Kolmogorov-Smirnov Test: statistic = {ks_stat:.3f}", ks_p)
    logging.info(f"{name} - Anderson-Darling Test: statistic = {ad_stat.statistic:.3f}")
    for i in range(len(ad_stat.critical_values)):
        cv, sl = ad_stat.critical_values[i], ad_stat.significance_level[i]
        logging.info(f"    Critical value at {sl}%: {cv:.3f}")

    return shapiro_p > 0.05 and ks_p > 0.05


def plot_boxplot(data_a: np.ndarray, data_b: np.ndarray, labels: list[str], title: str, filename: str):
    plt.boxplot([data_a, data_b], labels=labels)
    plt.title(title)
    plt.savefig(filename)
    plt.close()


def plot_histogram(data_a: np.ndarray, data_b: np.ndarray, labels: list[str], title: str, filename: str):
    plt.hist(data_a, bins=30, alpha=0.7, label=labels[0])
    plt.hist(data_b, bins=30, alpha=0.7, label=labels[1])
    plt.title(title)
    plt.legend()
    plt.savefig(filename)
    plt.close()


def plot_qqplot(data: np.ndarray, title: str, filename: str):
    stats.probplot(data, dist="norm", plot=plt)
    plt.title(title)
    plt.savefig(filename)
    plt.close()


def log_test(msg, p_value):
    if p_value < 0.01:
        color, interpretation = "red", "H_0 Rejected"
    elif 0.01 <= p_value < 0.05:
        color, interpretation = "orange", "Inconclusive"
    else:
        color, interpretation = "green", "H_0 Not Rejected"
    logging.info(
        f"{msg}, p-value = {colored(f'{p_value:.3f}', color)} -> {colored(f'{interpretation}', color, attrs=['bold'])}"
    )


def paired_sample_tests(ref_sample: dict, new_sample: dict):
    log_section("Normality tests")
    max_reward_diff = ref_sample["max_rewards"] - new_sample["max_rewards"]
    sum_reward_diff = ref_sample["sum_rewards"] - new_sample["sum_rewards"]
    normal_max_reward_diff = normality_tests(max_reward_diff, "Max Reward Difference")
    normal_sum_reward_diff = normality_tests(sum_reward_diff, "Sum Reward Difference")

    log_section("Paired-sample tests")
    if normal_max_reward_diff:
        t_stat_max_reward, p_val_max_reward = ttest_rel(ref_sample["max_rewards"], new_sample["max_rewards"])
        log_test(f"Paired t-test for Max Reward: t-statistic = {t_stat_max_reward:.3f}", p_val_max_reward)
    else:
        w_stat_max_reward, p_wilcox_max_reward = wilcoxon(
            ref_sample["max_rewards"], new_sample["max_rewards"]
        )
        log_test(f"Wilcoxon test for Max Reward: statistic = {w_stat_max_reward:.3f}", p_wilcox_max_reward)

    if normal_sum_reward_diff:
        t_stat_sum_reward, p_val_sum_reward = ttest_rel(ref_sample["sum_rewards"], new_sample["sum_rewards"])
        log_test(f"Paired t-test for Sum Reward: t-statistic = {t_stat_sum_reward:.3f}", p_val_sum_reward)
    else:
        w_stat_sum_reward, p_wilcox_sum_reward = wilcoxon(
            ref_sample["sum_rewards"], new_sample["sum_rewards"]
        )
        log_test(f"Wilcoxon test for Sum Reward: statistic = {w_stat_sum_reward:.3f}", p_wilcox_sum_reward)

    table = np.array(
        [
            [
                np.sum((ref_sample["successes"] == 1) & (new_sample["successes"] == 1)),
                np.sum((ref_sample["successes"] == 1) & (new_sample["successes"] == 0)),
            ],
            [
                np.sum((ref_sample["successes"] == 0) & (new_sample["successes"] == 1)),
                np.sum((ref_sample["successes"] == 0) & (new_sample["successes"] == 0)),
            ],
        ]
    )
    mcnemar_result = mcnemar(table, exact=True)
    log_test(f"McNemar's test for Success: statistic = {mcnemar_result.statistic:.3f}", mcnemar_result.pvalue)


def independent_sample_tests(ref_sample: dict, new_sample: dict):
    log_section("Normality tests")
    normal_max_rewards_a = normality_tests(ref_sample["max_rewards"], "Max Rewards Ref Sample")
    normal_max_rewards_b = normality_tests(new_sample["max_rewards"], "Max Rewards New Sample")
    normal_sum_rewards_a = normality_tests(ref_sample["sum_rewards"], "Sum Rewards Ref Sample")
    normal_sum_rewards_b = normality_tests(new_sample["sum_rewards"], "Sum Rewards New Sample")

    log_section("Independent samples tests")
    if normal_max_rewards_a and normal_max_rewards_b:
        t_stat_max_reward, p_val_max_reward = ttest_ind(
            ref_sample["max_rewards"], new_sample["max_rewards"], equal_var=False
        )
        log_test(f"Two-Sample t-test for Max Reward: t-statistic = {t_stat_max_reward:.3f}", p_val_max_reward)
    else:
        u_stat_max_reward, p_u_max_reward = mannwhitneyu(ref_sample["max_rewards"], new_sample["max_rewards"])
        log_test(f"Mann-Whitney U test for Max Reward: U-statistic = {u_stat_max_reward:.3f}", p_u_max_reward)

    if normal_sum_rewards_a and normal_sum_rewards_b:
        t_stat_sum_reward, p_val_sum_reward = ttest_ind(
            ref_sample["sum_rewards"], new_sample["sum_rewards"], equal_var=False
        )
        log_test(f"Two-Sample t-test for Sum Reward: t-statistic = {t_stat_sum_reward:.3f}", p_val_sum_reward)
    else:
        u_stat_sum_reward, p_u_sum_reward = mannwhitneyu(ref_sample["sum_rewards"], new_sample["sum_rewards"])
        log_test(f"Mann-Whitney U test for Sum Reward: U-statistic = {u_stat_sum_reward:.3f}", p_u_sum_reward)


def perform_tests(ref_sample: dict, new_sample: dict, output_dir: Path):
    log_section("Descriptive Stats")
    logging.info(f"Number of episode - Ref Sample: {ref_sample['num_episodes']}")
    logging.info(f"Number of episode - New Sample: {new_sample['num_episodes']}")

    seeds_a, seeds_b = ref_sample["seeds"], new_sample["seeds"]
    if seeds_a == seeds_b:
        logging.info("Samples are paired (identical seeds).")
        paired = True
    else:
        logging.info("Samples are considered independent (seeds are different).")
        paired = False

    descriptive_stats(ref_sample, new_sample, "successes")
    descriptive_stats(ref_sample, new_sample, "max_rewards")
    descriptive_stats(ref_sample, new_sample, "sum_rewards")

    log_section("Effect Size")
    d_max_reward = cohens_d(ref_sample["max_rewards"], new_sample["max_rewards"])
    d_sum_reward = cohens_d(ref_sample["sum_rewards"], new_sample["sum_rewards"])
    logging.info(f"Cohen's d for Max Reward: {d_max_reward:.3f}")
    logging.info(f"Cohen's d for Sum Reward: {d_sum_reward:.3f}")

    if paired:
        paired_sample_tests(ref_sample, new_sample)
    else:
        independent_sample_tests(ref_sample, new_sample)

    output_dir.mkdir(exist_ok=True, parents=True)

    plot_boxplot(
        ref_sample["max_rewards"],
        new_sample["max_rewards"],
        ["Ref Sample Max Reward", "New Sample Max Reward"],
        "Boxplot of Max Rewards",
        f"{output_dir}/boxplot_max_reward.png",
    )
    plot_boxplot(
        ref_sample["sum_rewards"],
        new_sample["sum_rewards"],
        ["Ref Sample Sum Reward", "New Sample Sum Reward"],
        "Boxplot of Sum Rewards",
        f"{output_dir}/boxplot_sum_reward.png",
    )

    plot_histogram(
        ref_sample["max_rewards"],
        new_sample["max_rewards"],
        ["Ref Sample Max Reward", "New Sample Max Reward"],
        "Histogram of Max Rewards",
        f"{output_dir}/histogram_max_reward.png",
    )
    plot_histogram(
        ref_sample["sum_rewards"],
        new_sample["sum_rewards"],
        ["Ref Sample Sum Reward", "New Sample Sum Reward"],
        "Histogram of Sum Rewards",
        f"{output_dir}/histogram_sum_reward.png",
    )

    plot_qqplot(
        ref_sample["max_rewards"],
        "Q-Q Plot of Ref Sample Max Rewards",
        f"{output_dir}/qqplot_sample_a_max_reward.png",
    )
    plot_qqplot(
        new_sample["max_rewards"],
        "Q-Q Plot of New Sample Max Rewards",
        f"{output_dir}/qqplot_sample_b_max_reward.png",
    )
    plot_qqplot(
        ref_sample["sum_rewards"],
        "Q-Q Plot of Ref Sample Sum Rewards",
        f"{output_dir}/qqplot_sample_a_sum_reward.png",
    )
    plot_qqplot(
        new_sample["sum_rewards"],
        "Q-Q Plot of New Sample Sum Rewards",
        f"{output_dir}/qqplot_sample_b_sum_reward.png",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("ref_sample_path", type=Path, help="Path to the reference sample JSON file.")
    parser.add_argument("new_sample_path", type=Path, help="Path to the new sample JSON file.")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/compare/"),
        help="Directory to save the output results. Defaults to outputs/compare/",
    )
    args = parser.parse_args()
    init_logging(args.output_dir)

    ref_sample = get_eval_info_episodes(args.ref_sample_path)
    new_sample = get_eval_info_episodes(args.new_sample_path)
    perform_tests(ref_sample, new_sample, args.output_dir)
