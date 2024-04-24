import unittest
from unittest import TestCase

from ait_llm.metrics.metrics import Accuracy, EditDistance, BLEU, get_metric


class TestMetrics(TestCase):

    def test_accuracy_score_positive(self):
        outs = ["1"]
        refs = ["1"]
        thr = 0

        self.accuracy = Accuracy(thr)
        generator = self.accuracy.compare_two_lists_of_words(outs, refs)

        with self.assertRaises(StopIteration):
            next(generator)

    def test_accuracy_score_negative(self):
        outs = ["1"]
        refs = [""]
        thr = 1
        
        self.accuracy = Accuracy(thr)
        generator = self.accuracy.compare_two_lists_of_words(outs, refs)

        self.assertEqual(next(generator), (0, 0))

    def test_edit_distance_positive(self):
        outs = ["a"]
        refs = ["a"]

        edit_distance = EditDistance(0)
        generator = edit_distance.compare_two_lists_of_words(outs, refs)

        with self.assertRaises(StopIteration):
            next(generator)
    
    def test_edit_distance_diff_thr(self):
        outs = ["a"]
        refs = [" "]

        edit_distance = EditDistance(0)
        generator = edit_distance.compare_two_lists_of_words(outs, refs)
        self.assertEqual(next(generator), (0, 1))

        edit_distance = EditDistance(2)
        generator = edit_distance.compare_two_lists_of_words(outs, refs)
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


if __name__ == '__main__':
    unittest.main()


    