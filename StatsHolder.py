# Import packages
import numpy as np
from scipy.stats import norm


# Class
class StatsHolder:

    # Define class attributes
    eps = 1e-7

    def __init__(self, loss, acc, tp, tn, fp, fn, extra_stats=None):
        # Initialize attributes
        self.loss = loss
        self.acc = acc
        self.tp = tp
        self.tn = tn
        self.fp = fp
        self.fn = fn

        # Compute extra stats
        if extra_stats is not None:
            self.n_vals, self.sens, self.spec, self.precis, self.f1, self.mcc = extra_stats
        else:
            self.n_vals = 1
            self.sens = tp / (tp + fn + self.eps)
            self.spec = tn / (tn + fp + self.eps)
            self.precis = tp / (tp + fp + self.eps)
            self.f1 = 2 * self.sens * self.precis / (self.sens + self.precis + self.eps)
            self.mcc = (tp * tn - fp * fn) / np.sqrt(np.float64(tp + fp) * (fn + tn) * (tp + fn) * (fp + tn) + self.eps)

        # Compute the Macro-Averaged statistics for the multiclass scenario
        if isinstance(self.f1, np.ndarray):
            self.sens = np.mean(self.sens)
            self.spec = np.mean(self.spec)
            self.precis = np.mean(self.precis)
            self.f1 = np.mean(self.f1)
            self.mcc = np.mean(self.mcc)

    def print_ci(self, ci_alpha=0.05):
        print(" Number of samples = " + str(self.n_vals))

        temp_dict = {}
        for attribute_name, attribute_value in self.__dict__.items():
            # Ignore extra attributes
            if attribute_name.endswith("_ci") or attribute_name == "n_vals" or attribute_name == "eps":
                continue

            # Ignore not-probability values
            if attribute_name == "mcc" or (np.floor(attribute_value) != 0 and attribute_value != 1.0):
                continue

            ci = StatsHolder.compute_ci(phat=attribute_value, n_vals=self.n_vals, ci_alpha=ci_alpha)
            temp_dict[attribute_name + "_ci"] = ci
            print(" > " + attribute_name + ": [" + str(ci[0]) + "; " + str(ci[1]) + "]")
        self.__dict__.update(temp_dict)

    @staticmethod
    def compute_ci(phat, n_vals, ci_alpha):
        z = norm.ppf(1 - ci_alpha / 2, loc=0, scale=1)
        delta = z * np.sqrt(phat * (1 - phat) / n_vals)

        p_min = phat - delta
        if p_min < 0:
            p_min = 0
        p_max = phat + delta
        if p_max > 1:
            p_max = 1

        return [p_min, p_max]

    @staticmethod
    def average_stats(stats_list, alpha_ci=None):
        loss, acc, tp, tn, fp, fn, sens, spec, precis, f1, mcc = StatsHolder.get_stats_lists(stats_list)
        n_vals = len(stats_list)

        # Get means and standard deviations
        s_loss = np.std(loss)
        m_loss = np.mean(loss)
        s_acc = np.std(acc)
        m_acc = np.mean(acc)
        s_tp = np.std(tp)
        m_tp = np.mean(tp)
        s_tn = np.std(tn)
        m_tn = np.mean(tn)
        s_fp = np.std(fp)
        m_fp = np.mean(fp)
        s_fn = np.std(fn)
        m_fn = np.mean(fn)

        s_sens = np.std(sens)
        m_sens = np.mean(sens)
        s_spec = np.std(spec)
        m_spec = np.mean(spec)
        s_precis = np.std(precis)
        m_precis = np.mean(precis)
        s_f1 = np.std(f1)
        m_f1 = np.mean(f1)
        s_mcc = np.std(mcc)
        m_mcc = np.mean(mcc)

        mean_stats = StatsHolder(loss=m_loss, acc=m_acc, tp=m_tp, tn=m_tn, fp=m_fp, fn=m_fn, 
                                 extra_stats=[n_vals, m_sens, m_spec, m_precis, m_f1, m_mcc])
        std_stats = StatsHolder(loss=s_loss, acc=s_acc, tp=s_tp, tn=s_tn, fp=s_fp, fn=s_fn,
                                extra_stats=[n_vals, s_sens, s_spec, s_precis, s_f1, s_mcc])

        if alpha_ci is None:
            return mean_stats, std_stats
        else:
            # Get 95% confidence intervals
            percentiles = [100 * alpha_ci / 2, 100 * 1 - alpha_ci / 2]
            loss_ci = np.percentile(loss, percentiles)
            acc_ci = np.percentile(acc, percentiles)
            tp_ci = np.percentile(tp, percentiles)
            tn_ci = np.percentile(tn, percentiles)
            fp_ci = np.percentile(fp, percentiles)
            fn_ci = np.percentile(fn, percentiles)
            sens_ci = np.percentile(sens, percentiles)
            spec_ci = np.percentile(spec, percentiles)
            precis_ci = np.percentile(precis, percentiles)
            f1_ci = np.percentile(f1, percentiles)
            mcc_ci = np.percentile(mcc, percentiles)

            ci_dict = {"loss": loss_ci, "acc": acc_ci, "tp": tp_ci, "tn": tn_ci, "fp": fp_ci, "fn": fn_ci,
                       "sens": sens_ci, "spec": spec_ci, "precis": precis_ci, "f1": f1_ci, "mcc": mcc_ci}
            stats_list_dict = {"loss": loss, "acc": acc, "tp": tp, "tn": tn, "fp": fp, "fn": fn, "sens": sens,
                               "spec": spec, "precis": precis, "f1": f1, "mcc": mcc}
            return mean_stats, std_stats, ci_dict, stats_list_dict



    @staticmethod
    def get_stats_lists(stats_list):
        loss = StatsHolder.get_stat_list("loss", stats_list)
        acc = StatsHolder.get_stat_list("acc", stats_list)
        tp = StatsHolder.get_stat_list("tp", stats_list)
        tn = StatsHolder.get_stat_list("tn", stats_list)
        fp = StatsHolder.get_stat_list("fp", stats_list)
        fn = StatsHolder.get_stat_list("fn", stats_list)
        sens = StatsHolder.get_stat_list("sens", stats_list)
        spec = StatsHolder.get_stat_list("spec", stats_list)
        precis = StatsHolder.get_stat_list("precis", stats_list)
        f1 = StatsHolder.get_stat_list("f1", stats_list)
        mcc = StatsHolder.get_stat_list("mcc", stats_list)

        return loss, acc, tp, tn, fp, fn, sens, spec, precis, f1, mcc

    @staticmethod
    def get_stat_list(stat_name, stats_list):
        stat = np.array([x.__dict__[stat_name] for x in stats_list])

        # Avoid issues related to the use of eps in the denominator for the computation of some statistics
        stat = np.round(stat, 6)
        return stat

    @staticmethod
    def average_bootstrap_values(stat_list, alpha_ci=0.05):
        n_vals = len(stat_list)
        loss = [stat_list[i].loss for i in range(n_vals)]
        acc = [stat_list[i].acc for i in range(n_vals)]
        sens = [stat_list[i].target_sens for i in range(n_vals)]
        spec = [stat_list[i].target_spec for i in range(n_vals)]
        precis = [stat_list[i].target_precis for i in range(n_vals)]
        neg_pred_val = [stat_list[i].target_neg_pred_val for i in range(n_vals)]
        f1 = [stat_list[i].target_f1 for i in range(n_vals)]
        mcc = [stat_list[i].target_mcc for i in range(n_vals)]
        auc = [stat_list[i].target_auc for i in range(n_vals)]
        fnr = [stat_list[i].target_fnr for i in range(n_vals)]
        fpr = [stat_list[i].target_fpr for i in range(n_vals)]

        extra_stats = (n_vals, sens, spec, precis, neg_pred_val, f1, mcc, auc, fnr, fpr)
        stats = StatsHolder(loss, acc, None, None, None, None, extra_stats=extra_stats,
                            get_distribution_params=True, alpha_ci=alpha_ci)
        return stats


# Main
if __name__ == '__main__':
    # Define a stats holder
    loss1 = 0.2
    acc1 = 0.8
    tp1 = 90
    tn1 = 70
    fp1 = 10
    fn1 = 30
    stats1 = StatsHolder(loss1, acc1, tp1, tn1, fp1, fn1)

    # Average two stats holder
    loss2 = 0.4
    acc2 = 1
    tp2 = 70
    tn2 = 60
    fp2 = 20
    fn2 = 30
    stats2 = StatsHolder(loss2, acc2, tp2, tn2, fp2, fn2)

    mean_stats1, dev_stats1 = StatsHolder.average_stats([stats1, stats2])

    # Print 95% CIs
    alpha = 0.05
    mean_stats1.print_ci(ci_alpha=alpha)
