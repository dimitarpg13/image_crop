import os
from dichotomous_image_segmentation.segmentation_manager import\
    (SegmentationManager, get_center_of_mass, DEVICE)
from PIL import Image


class ImageCropManager:
    def __init__(self):
        pass

    def get_cropped_image(self, image_path, image_name, image_file_extension, aspect_ratio):
        """
        :param image_path: str, the URL of folder in which is the image to be cropped
        :param image_name: str, name of the image to be cropped
        :param image_file_extension: str
        :param aspect_ratio:
        :return PIL::Image instance of the cropped image
        """
        s_mgr = SegmentationManager(DEVICE)
        s_mask = s_mgr.predict_from_file(image_path, image_name, image_file_extension)
        c_mass = get_center_of_mass(s_mask)


