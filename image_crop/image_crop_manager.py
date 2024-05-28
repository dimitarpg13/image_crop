import os
from dichotomous_image_segmentation.segmentation_manager import\
    (SegmentationManager, get_center_of_mass, DEVICE)
from PIL import Image


def get_image_bounds_for_given_aspect_ratio(image_width, image_height, center_of_mass, aspect_ratio):
    left = 0
    top = 0
    right = image_width
    bottom = image_height
    return left, top, right, bottom


class ImageCropManager:
    def __init__(self):
        self.s_mgr = SegmentationManager(DEVICE)

    def get_cropped_image(self, image_path, image_name, image_file_extension, aspect_ratio):
        """
        :param image_path: str, the URL of folder in which is the image to be cropped
        :param image_name: str, name of the image to be cropped
        :param image_file_extension: str
        :param aspect_ratio:
        :return PIL::Image instance of the cropped image
        """

        s_mask = self.s_mgr.predict_from_file(image_path, image_name, image_file_extension)
        c_mass = get_center_of_mass(s_mask)

        left, top, right, bottom = get_image_bounds_for_given_aspect_ratio(
            s_mask.size[0], s_mask.size[1], c_mass, aspect_ratio)

        image_orig = Image.open(image_path + image_name + "." + image_file_extension)
        image_cropped = image_orig.crop((left, top, right, bottom))

        return image_cropped

