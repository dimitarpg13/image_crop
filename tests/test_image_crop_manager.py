import unittest
import os
from pathlib import Path
from dichotomous_image_segmentation.segmentation_manager import SegmentationManager, get_center_of_mass, DEVICE
from image_crop.image_crop_manager import ImageCropManager

IC_HOME = '/'.join(Path(os.path.abspath(os.path.dirname(__file__))).parts[:-1]).replace('/','',1)
DATASET_PATH = 'tests/sample_data/'

param_list = [('smiling-man', 3.0, 4.0, 308, 410),('smiling-man', 4.0, 3.0, 547, 410)]


class TestImageCropManager(unittest.TestCase):
    def setUp(self):
        self.ic_manager = ImageCropManager()

    def test_get_cropped_image(self):
        image_path = IC_HOME + "/" + DATASET_PATH
        for image_name, aspect_ratio_a, aspect_ratio_b, cropped_image_width, cropped_image_height in param_list:
            with self.subTest(image_name=image_name, aspect_ratio_a=aspect_ratio_a, aspect_ratio_b=aspect_ratio_b,
                              cropped_image_width=cropped_image_width, cropped_image_height=cropped_image_height):
                cropped_image = self.ic_manager.get_cropped_image(image_path, image_name, 'jpeg',
                                                                  cropped_image_width/cropped_image_height)

                self.assertEqual(cropped_image.size[0], cropped_image_width)
                self.assertEqual(cropped_image.size[1], cropped_image_height)
                cropped_image.save(image_path + image_name + "_" + str(round(aspect_ratio_a)) +
                                   "_" + str(round(aspect_ratio_b)) + "_cropped.jpeg")


if __name__ == '__main__':
    unittest.main()
