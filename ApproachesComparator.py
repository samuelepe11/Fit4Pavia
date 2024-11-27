# Import packages
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import os
from scipy.stats import shapiro, levene, ttest_ind, mannwhitneyu

from StatsHolder import StatsHolder
from Simulator import Simulator
from PatientDivisionSimulator import PatientDivisionSimulator


# Class
class ApproachesComparator:

    results_fold = Simulator.results_fold
    comparable_stats = ["loss", "acc", "sens", "spec", "precis", "f1", "mcc"]

    def __init__(self, working_dir, folder_name, simulator_name1, simulator_name2, alpha, is_rehab=False):
        self.working_dir = working_dir
        self.folder_name = folder_name
        if is_rehab:
            self.folder_name = "rehab_" + self.folder_name
            self.results_fold = "../IntelliRehabDS/" + self.results_fold
        self.results_dir = working_dir + self.results_fold + self.folder_name + "/"
        self.simulator1 = Simulator.load_simulator(working_dir, folder_name, simulator_name1, is_rehab=is_rehab)
        self.simulator2 = Simulator.load_simulator(working_dir, folder_name, simulator_name2, is_rehab=is_rehab)
        self.alpha = alpha

        if isinstance(self.simulator2, PatientDivisionSimulator):
            self.descr2 = "cross-subject"
            self.descr1 = "non-" + self.descr2
        else:
            self.descr2 = "simulator2"
            self.descr1 = "simulator1"

        # Extract simulated values
        self.loss1, self.acc1, self.tp1, self.tn1, self.fp1, self.fn1, self.sens1, self.spec1, self.precis1, self.f11, \
            self.mcc1 = StatsHolder.get_stats_lists(self.simulator1.test_stats)
        self.loss2, self.acc2, self.tp2, self.tn2, self.fp2, self.fn2, self.sens2, self.spec2, self.precis2, self.f12, \
            self.mcc2 = StatsHolder.get_stats_lists(self.simulator2.test_stats)

    def compare_stat(self, stat_name):
        # Create histograms folder
        if "histograms" not in os.listdir(self.results_dir):
            os.mkdir(self.results_dir + "histograms")

        # Assess whether the statistics are normally distributed
        norms = []
        _, axs = plt.subplots(2, 1, figsize=(10, 5))
        axs[0].set_title(self.folder_name + " " + stat_name)
        axs[1].set_xlabel("Values")

        for attribute_name, attribute_value in self.__dict__.items():
            for sim in [1, 2]:
                if attribute_name == stat_name + str(sim):
                    # Verify data normality

                    normality = self.verify_normality(attribute_name, attribute_value)
                    norms.append(normality)

                    # Store histograms
                    ApproachesComparator.draw_hist(axs[sim - 1], stat_name, attribute_value)

        plt.savefig(self.results_dir + "histograms/" + stat_name + "_hist.jpg", dpi=300)
        print()

        # Compare variance values
        same_var = self.compare_variance(stat_name, norms)
        print()

        # Compare mean values
        self.compare_mean(stat_name, norms, same_var)

    def verify_normality(self, stat_name, values):
        if len(np.unique(values)) == 1:
            # Avoid issues owed to identical stats values
            p_value = 1.0
        else:
            _, p_value = shapiro(values)

        normality = p_value > self.alpha
        if normality:
            addon = ""
        else:
            addon = " NOT"
        print(stat_name + " is" + addon + " normally distributed (p-value: {:.2e}".format(p_value) + ")")

        return normality

    def compare_variance(self, stat_name, normality):
        arr1 = self.__dict__[stat_name + "1"]
        arr2 = self.__dict__[stat_name + "2"]

        if len(np.unique(arr1)) == 1 and len(np.unique(arr2)) == 1 and arr1[0] == arr2[0]:
            # Avoid issues owed to identical stats values
            p_value = 1.0
        else:
            if normality[0] and normality[1]:
                # Levene test
                _, p_value = levene(arr1, arr2, center="mean")
            else:
                # Brown–Forsythe test
                _, p_value = levene(arr1, arr2, center="median")

        same_var = p_value > self.alpha
        if same_var:
            addon = ""
        else:
            addon = " DON'T"
        print("The " + stat_name + " samples" + addon + " have the same variance (p-value: {:.2e}".format(p_value) +
              ")")

        return same_var

    def compare_mean(self, stat_name, normality, same_variance, equality_check=False):
        arr1 = self.__dict__[stat_name + "1"]
        arr2 = self.__dict__[stat_name + "2"]

        if equality_check:
            alternative = "two-sided"
        else:
            if stat_name != "loss":
                alternative = "greater"
            else:
                alternative = "less"

        if len(np.unique(arr1)) == 1 and len(np.unique(arr2)) == 1 and arr1[0] == arr2[0]:
            # Avoid issues owed to identical stats values
            p_value = 1.0
        else:
            if normality[0] and normality[1]:
                # T-test if same variance, Welch's T-test (analog to T-test with Satterhwaite method) otherwise
                results = ttest_ind(arr1, arr2, equal_var=same_variance, alternative=alternative)
                p_value = results.pvalue
                if np.isnan(p_value):
                    p_value = 1.0
            else:
                # Mann-Whitney U rank test
                _, p_value = mannwhitneyu(arr1, arr2, alternative=alternative, method="auto")

        h = p_value < self.alpha
        if not equality_check:
            # Call from the main function
            m1_wins = h
            if m1_wins:
                addon = ""
            else:
                addon = " NOT"
            print("The " + self.descr1 + " " + stat_name + " is" + addon +
                  " better than the " + self.descr2 + " one (p-value: {:.2e}".format(p_value) + ")")

            if not m1_wins:
                self.compare_mean(stat_name, normality, same_variance, equality_check=True)

        else:
            # Call from another compare_mean method for equality check
            equal = not h
            if equal:
                addon = ""
            else:
                addon = " NOT"
            print("The " + stat_name + " samples are" + addon + " equal (p-value: {:.2e}".format(p_value) + ")")

        return h

    def compare_all_stats(self):
        print("COMPARE " + self.descr1.upper() + " VS " + self.descr2.upper() + " TEST PERFORMANCES")
        print()
        # Print test means and CIs
        print(self.descr1.upper() + ":")
        print("Test mean accuracy: " + str(round(self.simulator1.mean_test_stats.acc * 100, 2)) + "% >",
              self.simulator1.mean_test_stats.acc_ci, "> std: " +
              str(round(self.simulator1.dev_test_stats.acc * 100, 2)) + "%")
        print("Test mean F1-score: " + str(round(self.simulator1.mean_test_stats.f1 * 100, 2)) + "% >",
              self.simulator1.mean_test_stats.f1_ci, "> std: " +
              str(round(self.simulator1.dev_test_stats.f1 * 100, 2)) + "%")
        print(self.descr2.upper() + ":")
        print("Test mean accuracy: " + str(round(self.simulator2.mean_test_stats.acc * 100, 2)) + "% >",
              self.simulator2.mean_test_stats.acc_ci, "> std: " +
              str(round(self.simulator2.dev_test_stats.acc * 100, 2)) + "%")
        print("Test mean F1-score: " + str(round(self.simulator2.mean_test_stats.f1 * 100, 2)) + "% >",
              self.simulator2.mean_test_stats.f1_ci, "> std: " +
              str(round(self.simulator2.dev_test_stats.f1 * 100, 2)) + "%")
        print()

        for stat in self.comparable_stats:
            print("----------------------------------------------------------------------")
            self.compare_stat(stat)

    def draw_compare_plots(self, desired_stats=None, desired_stats_names=None):
        if desired_stats is None:
            all_stats = True
            desired_stats = self.comparable_stats
            fig_size = (10, 5)
        else:
            all_stats = False
            if desired_stats_names is None:
                desired_stats_names = desired_stats
            fig_size = (6, 4)

        # Transform data into DataFrames
        d1 = dict((k, self.__dict__[k + "1"]) for k in desired_stats)
        df1 = pd.DataFrame(data=d1)
        df1 = pd.melt(df1, value_vars=desired_stats, var_name="Metric type", value_name="Value")
        df1["Training type"] = self.descr1

        d2 = dict((k, self.__dict__[k + "2"]) for k in desired_stats)
        df2 = pd.DataFrame(data=d2)
        df2 = pd.melt(df2, value_vars=desired_stats, var_name="Metric type", value_name="Value")
        df2["Training type"] = self.descr2

        df = pd.concat([df1, df2])

        # Rename variables
        if not all_stats:
            for i in range(len(desired_stats)):
                df["Metric type"] = np.where(df["Metric type"].eq(desired_stats[i]), desired_stats_names[i],
                                             df["Metric type"])

        # Draw box-plot
        plt.figure(figsize=fig_size)
        sns.boxplot(x="Metric type", y="Value", data=df, hue="Training type", palette="Greens")
        plt.ylim([0, 1])
        plt.title("Statistics on the test set")

        if all_stats:
            plt.ylim([-0.05, 1.25])
            addon = "_all"
        else:
            addon = "_" + "_".join(desired_stats)
        title_start = self.results_dir + self.folder_name + addon
        plt.savefig(title_start + "_boxplot.jpg", dpi=300)
        plt.close()

        # Draw bar-plot
        plt.figure(figsize=fig_size)
        ax = sns.barplot(x="Metric type", y="Value", data=df, hue="Training type", width=0.4,
                         errorbar=("ci", 95), capsize=0.2, palette="Greens")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.ylim([0, 1])
        plt.title("Statistics on the test set")
        plt.ylabel("Value with 95% CI")

        plt.savefig(title_start + "_barplot.jpg", dpi=400)
        plt.close()

    @staticmethod
    def draw_hist(fig, stat_name, values):
        fig.hist(values, bins=1000, color="seagreen")
        fig.spines["top"].set_visible(False)
        fig.spines["right"].set_visible(False)
        fig.set_ylabel("Absolute frequency")

        if stat_name == "loss":
            fig.set_xlim([-0.05, 10.05])
            fig.set_ylim([0, 35])
        else:
            fig.set_xlim([0.45, 1.05])
            if stat_name in ["sens", "spec", "precis", "mcc"]:
                # fig.set_ylim([0, 45])
                fig.set_ylim([0, 2])
            else:
                fig.set_ylim([0, 7])


# Main
if __name__ == "__main__":
    # Define variables
    working_dir1 = "./../"
    sim_name1 = "random_division"
    sim_name2 = "patient_division"
    folder_name1 = "patientVSrandom_division_conv2d"
    alpha1 = 0.05
    is_rehab1 = True

    # Define comparator
    comparator = ApproachesComparator(working_dir=working_dir1, folder_name=folder_name1, simulator_name1=sim_name1,
                                      simulator_name2=sim_name2, alpha=alpha1, is_rehab=is_rehab1)
    comparator.compare_all_stats()

    # Visually compare results
    comparator.draw_compare_plots()

    # Visually compare only Accuracy and F1-score
    comparator.draw_compare_plots(desired_stats=["acc", "f1"], desired_stats_names=["Test Accuracy", "Test F1-score"])
