from unittest import TestCase
import linear_regression.main as lr


class LinearRegressionTests(TestCase):
    def test_linear_regression(self0):
        features = range(1, 8)
        labels = [155, 197, 244, 300, 356, 407, 448]
        price_per_room, base_price = lr.linear_regression(features, labels)
        print(price_per_room, base_price)
