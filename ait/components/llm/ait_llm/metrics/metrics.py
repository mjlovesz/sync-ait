import statistics
import re
import warnings
from itertools import islice

import jieba
from nltk import bleu_score

from ait_llm.common.validate import validate_parameters_by_func, validate_parameters_by_type
from ait_llm.common.log import logger
from rouge_chinese import Rouge


class Metrics(object):
    LEGAL_CHAR_PATTERN = r'^[\u4e00-\u9fa50-9a-zA-Z\s]+$'
    EXCLUDE_LIST = ['.', ',', '。', '，', ' ', '(', ')', '"', "'"]

    # >>>>>>>>>>>>>> accuracy >>>>>>>>>>>>>

    @staticmethod
    @validate_parameters_by_func(
        {
            "outs": (),
            "refs": (),
            "thr": [lambda thr: thr is None or 0 <= thr <= 1, ]
        }
    )
    def accuracy_score(outs, refs, thr, ngrams=None):
        if thr is None:
            thr = 1

        for i, (word1, word2) in enumerate(zip(outs, refs)):
            if word1 != word2:
                yield i, 0

    # >>>>>>>>>>>>>> edit distance >>>>>>>>>>>>>
    
    @staticmethod
    def _edit_distance_impl(word1, word2):
        m, n = len(word1), len(word2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if word1[i - 1] == word2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1]) + 1
        return dp[-1][-1]

    @classmethod
    @validate_parameters_by_func(
        {
            "outs": (),
            "refs": (),
            "thr": [lambda thr: thr is None or 0 <= thr]
        },
        in_class=True
    )
    def edit_distance(cls, outs, refs, thr, ngrams=None):
        if thr is None:
            # constant
            thr = 5

        for i, (word1, word2) in enumerate(zip(outs, refs)):
            score = cls._edit_distance_impl(word1, word2)
            if score <= thr:
                continue
            yield i, score
    
    # >>>>>>>>>>>>>> abnormal >>>>>>>>>>>>>
    @classmethod
    def _abnormal_string_rate_impl(cls, word):
        try:
            filtered_field = [word for word in jieba.cut(word) if word not in cls.EXCLUDE_LIST]
        except Exception as e:
            raise ValueError(f"Trying to tokenize {word}, but failed.") from e

        return 0.0 if not filtered_field else statistics.mean(not re.match(cls.LEGAL_CHAR_PATTERN, word) for word in filtered_field)

    @classmethod
    def _relative_abnormal_string_rate_impl(cls, out, ref):
        target_rate = cls._abnormal_string_rate_impl(ref)
        target_rate = 0.0001 if target_rate == 0 else target_rate

        return cls._abnormal_string_rate_impl(out) / target_rate

    @classmethod
    @validate_parameters_by_func(
        {
            "outs": [],
            "refs": [],
            "thr": [lambda thr: thr is None or 0 <= thr],
        },
        in_class=True
    )
    def relative_abnormal_string_rate(cls, outs, refs, thr, ngrams=None):
        if thr is None:
            thr = 1.2
        
        for i, (word1, word2) in enumerate(zip(outs, refs)):
            
            score = float("-inf")
            try:
                score = cls._relative_abnormal_string_rate_impl(word1, word2)
            except Exception as e:
                logger.error("An error occured when trying to calculate the relative abnormal string rate "
                             f"between `%s` and `%s` in the function `%s`. This error is caused by %s", word1, word2, "relative_abnormal_string_rate", e)
                continue

            if score <= thr:
                continue
            yield i, score
    
    # >>>>>>>>>>> Bleu >>>>>>>>>>>>>>>
    @classmethod
    def _bleu_score_impl(cls, out, ref, ngrams):
        ref_field = [word for word in jieba.cut(ref) if word not in cls.EXCLUDE_LIST]
        out_field = [word for word in jieba.cut(out) if word not in cls.EXCLUDE_LIST]

        weights = [0] * 4
        weights[ngrams - 1] = 1

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bleu = bleu_score.sentence_bleu([ref_field], out_field, weights=weights)
        return bleu

    @classmethod
    @validate_parameters_by_func(
        {
            "outs": [],
            "refs": [],
            "thr": [lambda thr: thr is None or 0 <= thr <= 1],
            "ngrams": [lambda ngrams: ngrams is None or ngrams in [1, 2, 3, 4]],
        },
        in_class=True
    )
    def bleu_score(cls, outs, refs, thr, ngrams=None):
        if thr is None:
            # constant
            thr = 0.4

        if ngrams is None:
            ngrams = 1
        
        for i, (word1, word2) in enumerate(zip(outs, refs)):
            score = float("-inf")
            try:
                score = cls._bleu_score_impl(word1, word2, int(ngrams))
            except Exception as e:
                logger.error("An error occured when trying to calculate the relative abnormal string rate "
                             f"between `%s` and `%s` in the function `%s`. This error is caused by %s", word1, word2, "bleu_score", e)
                continue

            if score <= thr:
                continue
            yield i, score

    # rouge
    @staticmethod
    def _rouge_score_impl(out, ref, ngrams):
        modified_out = " ".join(jieba.cut(out))
        modified_ref = " ".join(jieba.cut(ref))

        rouge = Rouge()

        scores = rouge.get_scores(modified_out, modified_ref)
        return scores[0][f"rouge-{ngrams}"]['f']

    @classmethod
    @validate_parameters_by_func(
        {
            "outs": [],
            "refs": [],
            "thr": [lambda thr: thr is None or 0 <= thr <= 1],
            "ngrams": [lambda ngrams: ngrams is None or ngrams in [1, 2, 3]]
        },
        in_class=True
    )
    def rouge_score(cls, outs, refs, thr, ngrams=None):
        if thr is None:
            # constant
            thr = 0.4

        if ngrams is None:
            ngrams = '1'
        
        if ngrams == 3:
            ngrams = 'l'
        
        for i, (word1, word2) in enumerate(zip(outs, refs)):
            score = float("-inf")
            try:
                score = cls._rouge_score_impl(word1, word2, ngrams)
            except Exception as e:
                logger.error("An error occured when trying to calculate the relative abnormal string rate "
                             f"between `%s` and `%s` in the function `%s`. This error is caused by %s", word1, word2, "rouge_score", e)
                continue

            if score <= thr:
                continue
            yield i, score

    @staticmethod
    def _distinct_impl(words, ngram):
        unique = set()
        count = 0
        for contiguous_item in zip(*(islice(words, i, None) for i in range(ngram))):
            unique.add(contiguous_item)
            count += 1
        
        return 0 if count == 0 else len(unique) / count

    @classmethod
    def _relative_distinct_impl(cls, out, ref, ngram):
        target_rate = cls._distinct_impl(ref, ngram)
        target_rate = 0.0001 if target_rate == 0 else target_rate

        return cls._distinct_impl(out, ngram) / target_rate

    @classmethod
    @validate_parameters_by_func(
        {
            "outs": [],
            "refs": [],
            "thr": [lambda thr: thr is None or 0 <= thr],
            "ngrams": [lambda ngrams: ngrams is None or ngrams in [1, 2, 3, 4]]
        },
        in_class=True
    )
    def relative_distinct_string_rate(cls, outs, refs, thr, ngrams=None):
        if thr is None:
            thr = 0.8
        
        if ngrams is None:
            ngrams = 2

        for i, (word1, word2) in enumerate(zip(outs, refs)):
            score = float("-inf")
            try:
                score = cls._relative_distinct_impl(word1, word2, ngrams)
            except Exception as e:
                logger.error("An error occured when trying to calculate the relative abnormal string rate "
                             f"between `%s` and `%s` in the function `%s`. This error is caused by %s", word1, word2, "relative_distinct_string_rate", e)
                continue

            if score <= thr:
                continue
            yield i, score


@validate_parameters_by_type(
    {
        "metrics_name": [str],
        "thr": (int, float, None),
        "ngrams": [int, float, None],
    }
)
def get_metric(metric_name: str, thr: int | float | None = None, ngrams: int | None = None):

    MAPPING = {
        "accuracy": Metrics.accuracy_score,
        "rouge": Metrics.rouge_score,
        "bleu": Metrics.bleu_score,
        "edit_distance": Metrics.edit_distance,
        "relative_abnormal_string_rate": Metrics.relative_abnormal_string_rate,
        "relative_distinct": Metrics.relative_distinct_string_rate,
        "END": None,
    }

    if metric_name not in MAPPING:
        raise KeyError(f"{metric_name} is not supported.")

    def wrapper(outs, refs):
        return MAPPING[metric_name](outs, refs, thr, ngrams)

    return wrapper


if __name__ == '__main__':
    # get_metric("accuracy", 2)(1, 2)
    next(Metrics.edit_distance([1], [2]))
