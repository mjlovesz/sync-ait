import unittest
from unittest import TestCase

from ait_llm.metrics.metrics import Metrics, get_metric


class TestMetrics(TestCase):

    def test_accuracy_score_positive(self):
        outs = ["1"]
        refs = ["1"]
        thr = 0
        
        generator = Metrics.accuracy_score(outs, refs, None)

        with self.assertRaises(StopIteration):
            next(generator)

    def test_accuracy_score_negative(self):
        outs = ["1"]
        refs = [""]
        thr = 0
        
        generator = Metrics.accuracy_score(outs, refs, None)

        self.assertEqual(next(generator), (0, 0))

    def test_edit_distance_impl_positive(self):
        word1 = "a"
        word2 = "a"

        self.assertEqual(Metrics._edit_distance_impl(word1, word2), 0)
    
    def test_edit_distance_impl_negative(self):
        word1 = "a"
        word2 = " "

        self.assertEqual(Metrics._edit_distance_impl(word1, word2), 1)

    def test_edit_distance_positive(self):
        outs = ["a"]
        refs = ["a"]

        generator = Metrics.edit_distance(outs, refs, None)

        with self.assertRaises(StopIteration):
            next(generator)
    
    def test_edit_distance_diff_thr(self):
        outs = ["a"]
        refs = [" "]

        generator = Metrics.edit_distance(outs, refs, 0)
        self.assertEqual(next(generator), (0, 1))

        generator = Metrics.edit_distance(outs, refs, 2)
        with self.assertRaises(StopIteration):
            next(generator)

    def test_get_metrics_should_raise_when_name_not_valid(self):
        with self.assertRaises(TypeError):
            get_metric(1)
        
        with self.assertRaises(TypeError):
            get_metric(1.5)
        
        with self.assertRaises(TypeError):
            get_metric([])

    def test_get_metrics_should_raise_when_name_not_support(self):
        with self.assertRaises(KeyError):
            get_metric("")

        with self.assertRaises(KeyError):
            get_metric("acc")        

    def test_get_metrics_should_raise_when_thr_not_valid(self):
        with self.assertRaises(TypeError):
            get_metric("accuracy", "")

    def test_abnormal_string_rate_should_0(self):
        self.assertEqual(Metrics._abnormal_string_rate_impl("我爱你中国"), 0.0)

    def test_relative_abnormal_string_rate_impl_should_0(self):
        self.assertEqual(Metrics._relative_abnormal_string_rate_impl("我爱你中国", "亲爱的母亲"), 0.0)
    
    def test_relative_abnormal_string_rate_should_0(self):
        generator = Metrics.relative_abnormal_string_rate(["我爱你中国"], ["亲爱的母亲"], None)

        with self.assertRaises(StopIteration):
            next(generator)
    
    def test_bleu(self):
        outs = ['The quick brown dog jumps on the log.', "This is a good translation."]
        refs = ['The fast black dog jumps over the log', "Can not understand this translation."]

        generator = Metrics.bleu_score(outs, refs, 0.2)
        next(generator)

        generator = Metrics.bleu_score(outs, refs, 0.625)
        with self.assertRaises(StopIteration):
            next(generator)

    def test_rouge(self):
        outs = ['The quick brown dog jumps on the log.', "This is a good translation."]
        refs = ['The fast black dog jumps over the log', "Can not understand this translation."]

        generator = Metrics.rouge_score(outs, refs, 1 / 3, 3)
        
        next(generator)

        generator = Metrics.rouge_score(outs, refs, 0.8, 2)
        with self.assertRaises(StopIteration):
            next(generator)

    def test_relative_distinct_rate(self):
        outs = ['The quick brown dog jumps on the log the log', "This is a good translation."]
        refs = ['The black dog jumps over the log the log', "Can not understand this translation."]

        generator = Metrics.relative_distinct_string_rate(outs, refs, 0.9)
    
        next(generator)
        
        generator = Metrics.relative_distinct_string_rate(outs, refs, 1.1)
        with self.assertRaises(StopIteration):
            next(generator)

    def test_relative_abnormal_string_rate(self):
        outs = ['The quick brown dog jumps on $ $ $ $', "This is a good translation."]
        refs = ['The quick brown dog jumps on the log $ $', "Can not understand this translation."]

        generator = Metrics.relative_abnormal_string_rate(outs, refs, 0.5)
        
        generator = Metrics.relative_abnormal_string_rate(outs, refs, 3.0)
        with self.assertRaises(StopIteration):
            next(generator)


if __name__ == '__main__':
    unittest.main()


    