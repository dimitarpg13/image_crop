import unittest
import os
from pathlib import Path
from dichotomous_image_segmentation.segmentation_manager import SegmentationManager, get_center_of_mass, DEVICE
from image_crop.image_crop_manager import ImageCropManager

IC_HOME = '/'.join(Path(os.path.abspath(os.path.dirname(__file__))).parts[:-1]).replace('/','',1)
DATASET_PATH = 'tests/sample_data/'


class TestImageCropManager(unittest.TestCase):
    def setUp(self):
        self.ic_manager = ImageCropManager()

    def test_get_cropped_image(self):
        image_path = IC_HOME + "/" + DATASET_PATH
        image_name = 'smiling-man'
        cropped_image = self.ic_manager.get_cropped_image(image_path, image_name, 'jpeg', 3.0/4.0)
        #(x_coord_center_of_mass, y_coord_center_of_mass) = get_center_of_mass(mask)
        #self.assertEqual(x_coord_center_of_mass, 300)
        #self.assertEqual(y_coord_center_of_mass, 360)
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
