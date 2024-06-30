import unittest
import numpy as np
from skimage import data
from sklearn.ensemble import RandomForestClassifier
from functools import partial
from skimage import feature, future


class TestTrainableSegmentation(unittest.TestCase):

    def setUp(self):
        self.full_img = data.skin()
        self.img = self.full_img[:900, :900]
        self.training_labels = np.zeros(self.img.shape[:2], dtype=np.uint8)
        self.training_labels[:130] = 1
        self.training_labels[:170, :400] = 2
        self.training_labels[600:900, 200:650] = 2
        self.training_labels[330:430, 210:320] = 3
        self.training_labels[260:340, 60:170] = 4
        self.training_labels[150:200, 720:860] = 4
        self.sigma_min = 1
        self.sigma_max = 16
        self.features_func = partial(
            feature.multiscale_basic_features,
            intensity=True,
            edges=False,
            texture=True,
            sigma_min=self.sigma_min,
            sigma_max=self.sigma_max,
            channel_axis=-1,
        )
        self.features = self.features_func(self.img)
        self.clf = RandomForestClassifier(n_estimators=50, n_jobs=-1, max_depth=10, max_samples=0.05)

    def test_fit_segmenter(self):
        clf = future.fit_segmenter(self.training_labels, self.features, self.clf)
        self.assertIsInstance(clf, RandomForestClassifier)

    def test_predict_segmenter(self):
        clf = future.fit_segmenter(self.training_labels, self.features, self.clf)
        result = future.predict_segmenter(self.features, clf)
        self.assertIsInstance(result, np.ndarray)


if __name__ == '__main__':
    unittest.main()
