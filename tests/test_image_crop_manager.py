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

        self.assertEqual(cropped_image.size[0], 308)
        self.assertEqual(cropped_image.size[1], 410)
        #cropped_image.save(image_path + image_name + "_cropped.jpeg")



if __name__ == '__main__':
    unittest.main()
