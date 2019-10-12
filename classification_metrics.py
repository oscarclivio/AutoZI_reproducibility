from sklearn import metrics, mixture
import numpy as np
from scipy.stats import beta

def div_float_none(num, dem):
    if num is None or dem is None or dem == 0.:
        return None
    else:
        return float(num) / dem

def prod_none(a, b):
    if a is None or b is None:
        return None
    else:
        return a*b

def sum_none(a, b):
    if a is None or b is None:
        return None
    else:
        return a+b

def confusion_matrix_stats(labels_predicted, labels_gt, masks=None):
    results = {}

    results['total'] = len(labels_gt.reshape(-1))

    # Confusion matrix
    results['tp'] = (labels_predicted == labels_gt)[labels_gt].sum()
    results['tn'] = (labels_predicted == labels_gt)[~labels_gt].sum()
    results['fp'] = (labels_predicted != labels_gt)[~labels_gt].sum()
    results['fn'] = (labels_predicted != labels_gt)[labels_gt].sum()

    # Metrics
    results['tpr'] = div_float_none(results['tp'], sum_none(results['tp'], results['fn']))
    results['tnr'] = div_float_none(results['tn'], sum_none(results['tn'], results['fp']))
    results['ppv'] = div_float_none(results['tp'], sum_none(results['tp'], results['fp']))
    results['npv'] = div_float_none(results['tn'], sum_none(results['tn'], results['fn']))
    results['fnr'] = div_float_none(results['fn'], sum_none(results['tp'], results['fn']))
    results['fpr'] = div_float_none(results['fp'], sum_none(results['tn'], results['fp']))
    results['fdr'] = div_float_none(results['fp'], sum_none(results['tp'], results['fp']))
    results['for'] = div_float_none(results['fn'], sum_none(results['tn'], results['fn']))

    # Overall scores
    results['accuracy'] = (labels_predicted == labels_gt).mean()
    results['f1-score'] = prod_none(2, div_float_none(prod_none(results['ppv'], results['tpr']), \
                                                      sum_none(results['ppv'], results['tpr'])))

    if masks is not None:
        for name,mask in masks.items():
            results_split = confusion_matrix_stats(labels_predicted[mask], labels_gt[mask], masks=None)
            results_split = {name + '_' + key: value for key,value in results_split.items()}
            results.update(results_split)


    return results

class ClassificationMetric(object):

    def __init__(self, labels_gt):
        self.labels_gt = labels_gt
        self.tag = 'abstract'

    def compute(self, scores):
        return {}






class ConfusionMatrixMetric(ClassificationMetric):

    def __init__(self, labels_gt, decision_rule=None, masks=None):

        self.labels_gt = labels_gt
        self.tag = 'confusionmatrix'
        self.decision_rule = (lambda s: (s > 0.5)) if decision_rule is None else decision_rule
        self.masks = masks

    def compute(self, scores,):

        labels_predicted = self.decision_rule(scores)

        results = confusion_matrix_stats(labels_predicted, self.labels_gt, masks=self.masks)

        return results


class ROCMetric(ClassificationMetric):

    def __init__(self, labels_gt):
        super().__init__(labels_gt)
        self.tag = 'roc'

    def compute(self, scores):
        fpr, tpr, thresholds = metrics.roc_curve(self.labels_gt.astype(int), scores)
        auc = metrics.auc(fpr, tpr)

        results = {}

        results['fpr'] = fpr
        results['tpr'] = tpr
        results['thresholds'] = thresholds
        results['auc'] = auc

        return results


class PRCMetric(ClassificationMetric):

    def __init__(self, labels_gt):
        super().__init__(labels_gt)
        self.tag = 'prc'

    def compute(self, scores):
        ppv, tpr, thresholds = metrics.roc_curve(self.labels_gt.astype(int), scores)
        aps = metrics.average_precision_score(self.labels_gt.astype(int), scores)

        results = {}

        results['ppv'] = ppv
        results['tpr'] = tpr
        results['thresholds'] = thresholds
        results['aps'] = aps

        return results

class GMMMetric(ClassificationMetric):

    def __init__(self, labels_gt, masks=None, ineq='<', transform=None, gmm_args_list=None):
        super().__init__(labels_gt)

        self.tag = 'gmm'
        assert ineq in ['<', '>']
        self.ineq = ineq
        self.masks = masks

        self.transform = (lambda p: np.log((p + 1e-8) / (1 - p + 1e-8))) if transform is None else transform

        if gmm_args_list is None:
            self.args_gmm_list = \
            [{'weight_concentration_prior_type': weight_concentration_prior_type,\
             'weight_concentration_prior': weight_concentration_prior,\
             'mean_precision_prior': mean_precision_prior,\
             'transform': transform}\
             for weight_concentration_prior_type in ['dirichlet_process','dirichlet_distribution']\
             for weight_concentration_prior in [1/128, 1/64, 1/32, 1/8, 1/2, 2, 8]\
             for mean_precision_prior in [1/16, 1/4, 1, 4, 16]\
             for transform in [False,True]
             ]

    def compute(self, scores):

        best_accuracy_gmm = 0.
        best_args_gmm = {}

        for args_gmm in self.args_gmm_list:
            transform = args_gmm['transform']
            scores_copy = 1.*scores
            if transform:
                scores_copy = self.transform(scores_copy)

            args_gmm_copy = dict(args_gmm)
            del args_gmm_copy['transform']

            gmm = mixture.BayesianGaussianMixture(n_components=2, covariance_type='full', **args_gmm_copy) \
                .fit(scores_copy.reshape(-1, 1))

            labels = np.array(gmm.predict(scores_copy.reshape(-1, 1))).reshape(-1)
            label_pos = int(not(gmm.means_[0].item() < gmm.means_[1].item())) if self.ineq == '<' \
                else int(not(gmm.means_[0].item() > gmm.means_[1].item()))
            accuracy_gmm = (labels == label_pos)[self.labels_gt].sum() + (labels != label_pos)[~self.labels_gt].sum()

            if accuracy_gmm > best_accuracy_gmm:
                best_accuracy_gmm = accuracy_gmm
                best_args_gmm = args_gmm

        results = dict()
        results['best_accuracy_gmm'] = best_accuracy_gmm
        results['best_args_gmm'] = best_args_gmm

        labels_predicted = (labels == label_pos)

        results.update(confusion_matrix_stats(labels_predicted, self.labels_gt, masks=self.masks))

        return results







############################

class ModelScoreEval(object):
    def __init__(self, name, outputs, metric_list, mask_selection=None):
        self.name = name
        self.metric_list = metric_list

        self.scores = self.extract_scores(outputs).reshape(-1)
        if mask_selection is not None:
            self.scores = self.scores[mask_selection]

    def extract_scores(self, outputs):
        return None

    def compute_all_metrics(self):

        results = {}
        total_len_check = 0

        for metric in self.metric_list:
            results_metric = metric.compute(self.scores.reshape(-1))
            total_len_check += len(results_metric)
            for key,value in results_metric.items():
                results[self.name + '_' + metric.tag + '_' + key] = value

        assert(len(results) == total_len_check)

        return results




class AutoZIBernoulliAverageEval(ModelScoreEval):

    def extract_scores(self, outputs):
        alpha_posterior = outputs['alpha_posterior']
        beta_posterior = outputs['beta_posterior']
        bernoulli_means = alpha_posterior / (alpha_posterior + beta_posterior)

        return 1. - bernoulli_means


class AutoZIBernoulliThresholdEval(ModelScoreEval):

    def __init__(self, name, outputs, metric_list, ineq='<', threshold=0.5, **args):
        assert ineq in ['<','>']
        self.ineq = ineq
        self.threshold=threshold
        super().__init__(name, outputs, metric_list, **args)

    def extract_scores(self, outputs):
        alpha_posterior = outputs['alpha_posterior']
        beta_posterior = outputs['beta_posterior']
        scores = beta.cdf(self.threshold, alpha_posterior, beta_posterior)

        if self.ineq == '>':
            scores = 1. - scores

        return scores

class AutoZIBernoulliThresholdBayesFactorEval(ModelScoreEval):

    def __init__(self, name, outputs, metric_list, ineq='<', threshold=0.5, eps=1e-8, **args):
        assert ineq in ['<','>']
        self.ineq = ineq
        self.threshold=threshold
        self.eps = eps
        super().__init__(name, outputs, metric_list, **args)

    def extract_scores(self, outputs):
        alpha_posterior = outputs['alpha_posterior']
        beta_posterior = outputs['beta_posterior']
        scores = beta.cdf(self.threshold, alpha_posterior, beta_posterior)

        alpha_prior = outputs['alpha_prior']
        beta_prior = outputs['beta_prior']
        score_prior = beta.cdf(self.threshold, alpha_prior, beta_prior)

        if self.ineq == '>':
            scores = 1. - scores
            score_prior = 1. - score_prior

        bayes_factors = np.log(scores + self.eps) - np.log(1 - scores + self.eps)\
                       + np.log(1 - score_prior + self.eps) - np.log(score_prior + self.eps)

        return bayes_factors


class AutoZIBernoulliMedianEval(ModelScoreEval):

    def extract_scores(self, outputs):
        alpha_posterior = outputs['alpha_posterior']
        beta_posterior = outputs['beta_posterior']
        scores = beta.ppf(0.5, alpha_posterior, beta_posterior)

        return 1. - scores

class AutoZIAverageHeldOutDropoutEval(ModelScoreEval):

    def extract_scores(self, outputs):
        alpha_posterior = outputs['alpha_posterior']
        beta_posterior = outputs['beta_posterior']
        bernoulli_means = alpha_posterior / (alpha_posterior + beta_posterior)

        dropouts = (1-bernoulli_means)*outputs['dropout_probs_test']
        return dropouts

class FitDropoutEval(ModelScoreEval):

    def __init__(self, name, outputs, metric_list, key, **args):
        if key is not None:
            self.key = '_' + key
        else:
            self.key = ''
        super().__init__(name, outputs, metric_list, **args)


    def extract_scores(self, outputs):
        return outputs['dropouts' + self.key]


class FitDiffLLEval(ModelScoreEval):

    def extract_scores(self, outputs):
        diff = np.array(outputs['lls_zinb']) - np.array(outputs['lls_nb'])
        diff_sigma = 1. / (1. + np.exp(-diff))

        return diff_sigma


class FitDecisionEval(ModelScoreEval):

    def __init__(self, name, outputs, metric_list, key, **args):
        assert key in ['aic', 'bic', 'cv']
        self.key = '_' + key
        super().__init__(name, outputs, metric_list, **args)

    def extract_scores(self, outputs):
        return ~outputs['is_nb_pred' + self.key]





class AutoZIBernoulliAverageBatchesEval(ModelScoreEval):

    def __init__(self, name, outputs, metric_list, n_batches=4, **args):
        self.n_batches = n_batches
        super().__init__(name, outputs, metric_list, **args)

    def extract_scores(self, outputs):
        alpha_posterior = outputs['alpha_posterior']
        beta_posterior = outputs['beta_posterior']
        bernoulli_means = alpha_posterior / (alpha_posterior + beta_posterior)

        bernoulli_means = np.repeat(bernoulli_means[:,np.newaxis], self.n_batches, axis=1)

        return 1. - bernoulli_means


class AutoZIBernoulliThresholdBatchesEval(ModelScoreEval):

    def __init__(self, name, outputs, metric_list, n_batches=4, ineq='<', threshold=0.5, **args):
        assert ineq in ['<','>']
        self.n_batches = n_batches
        self.ineq = ineq
        self.threshold=threshold
        super().__init__(name, outputs, metric_list, **args)

    def extract_scores(self, outputs):
        alpha_posterior = outputs['alpha_posterior']
        beta_posterior = outputs['beta_posterior']
        scores = beta.cdf(self.threshold, alpha_posterior, beta_posterior)

        if self.ineq == '>':
            scores = 1. - scores

        scores = np.repeat(scores[:, np.newaxis], self.n_batches, axis=1)

        return scores

class AutoZIBernoulliThresholdBayesFactorBatchesEval(ModelScoreEval):

    def __init__(self, name, outputs, metric_list, n_batches=4,  ineq='<', threshold=0.5, eps=1e-8, **args):
        assert ineq in ['<','>']
        self.n_batches = n_batches
        self.ineq = ineq
        self.threshold=threshold
        self.eps = eps
        super().__init__(name, outputs, metric_list, **args)

    def extract_scores(self, outputs):
        alpha_posterior = outputs['alpha_posterior']
        beta_posterior = outputs['beta_posterior']
        scores = beta.cdf(self.threshold, alpha_posterior, beta_posterior)

        alpha_prior = outputs['alpha_prior']
        beta_prior = outputs['beta_prior']
        score_prior = beta.cdf(self.threshold, alpha_prior, beta_prior)

        if self.ineq == '>':
            scores = 1. - scores
            score_prior = 1. - score_prior

        bayes_factors = np.log(scores + self.eps) - np.log(1 - scores + self.eps)\
                       + np.log(1 - score_prior + self.eps) - np.log(score_prior + self.eps)

        bayes_factors = np.repeat(bayes_factors[:, np.newaxis], self.n_batches, axis=1)

        return bayes_factors


class AutoZIBernoulliMedianBatchesEval(ModelScoreEval):

    def __init__(self, name, outputs, metric_list, n_batches=4, **args):
        self.n_batches = n_batches
        super().__init__(name, outputs, metric_list, **args)

    def extract_scores(self, outputs):
        alpha_posterior = outputs['alpha_posterior']
        beta_posterior = outputs['beta_posterior']
        scores = beta.ppf(0.5, alpha_posterior, beta_posterior)

        scores = np.repeat(scores[:, np.newaxis], self.n_batches, axis=1)

        return 1. - scores
