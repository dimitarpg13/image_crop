import os
from dichotomous_image_segmentation.segmentation_manager import \
    (SegmentationManager, get_center_of_mass, DEVICE)
from PIL import Image


def get_image_bounds_for_given_aspect_ratio(image_height, image_width, center_of_mass, aspect_ratio):
    x_coord_mass = center_of_mass[0]
    y_coord_mass = center_of_mass[1]

    if y_coord_mass >= image_height / 2.0:
        cropped_image_height = 2 * (image_height - y_coord_mass)
        if (x_coord_mass >= aspect_ratio * (image_height - y_coord_mass) and
                image_width - x_coord_mass >= aspect_ratio * (image_height - y_coord_mass)):
            cropped_image_width = 2 * aspect_ratio * (image_height - y_coord_mass)
        elif x_coord_mass < aspect_ratio * (image_height - y_coord_mass) <= image_width - x_coord_mass:
            cropped_image_width = 2 * x_coord_mass
            cropped_image_height = 2 * x_coord_mass / aspect_ratio
        elif x_coord_mass >= aspect_ratio * (image_height - y_coord_mass) > image_width - x_coord_mass:
            cropped_image_width = 2 * (image_width - x_coord_mass)
            cropped_image_height = 2 * (image_width - x_coord_mass) / aspect_ratio
        else:  # (x_coord_mass < aspect_ratio*(image_height - y_coord_mass) and
               # image_width - x_coord_mass < aspect_ratio*(image_height - y_coord_mass))
            cropped_image_width = 2 * min(x_coord_mass, image_width - x_coord_mass)
            cropped_image_height = 2 * min(x_coord_mass, image_width - x_coord_mass) / aspect_ratio
    else:  # y_coord_mass < image_height/2.0
        cropped_image_height = 2 * y_coord_mass
        if (x_coord_mass >= aspect_ratio * y_coord_mass and
                image_width - x_coord_mass >= aspect_ratio * y_coord_mass):
            cropped_image_width = 2 * aspect_ratio * y_coord_mass
        elif x_coord_mass < aspect_ratio * y_coord_mass <= image_width - x_coord_mass:
            cropped_image_width = 2 * x_coord_mass
            cropped_image_height = 2 * x_coord_mass / aspect_ratio
        elif x_coord_mass >= aspect_ratio * y_coord_mass > image_width - x_coord_mass:
            cropped_image_width = 2 * (image_width - x_coord_mass)
            cropped_image_height = 2 * (image_width - x_coord_mass) / aspect_ratio
        else:  # x_coord_mass < aspect_ratio * y_coord_mass and image_width - x_coord_mass < aspect_ratio * y_coord_mass
            cropped_image_width = 2 * min(x_coord_mass, image_width - x_coord_mass)
            cropped_image_height = 2 * min(x_coord_mass, image_width - x_coord_mass) / aspect_ratio

    left = round(x_coord_mass - cropped_image_width / 2.0)
    #top = round(image_height - cropped_image_height)#(y_coord_mass + cropped_image_height / 2.0))
    top = round(y_coord_mass - cropped_image_height / 2.0)
    right = left + round(cropped_image_width) # round(x_coord_mass + cropped_image_width / 2.0)
    bottom = top+ round(cropped_image_height) #round(y_coord_mass + cropped_image_height / 2.0)

    return left, top, right, bottom


class ImageCropManager:
    def __init__(self):
        self.s_mgr = SegmentationManager(DEVICE)
        self.c_mass = None
        self.s_mask = None
        self.im_mask = None

    def get_cropped_image(self, image_path, image_name, image_file_extension, aspect_ratio):
        """
        :param image_path: str, the URL of folder in which is the image to be cropped
        :param image_name: str, name of the image to be cropped
        :param image_file_extension: str
        :param aspect_ratio:
        :return PIL::Image instance of the cropped image
        """

        self.s_mask = self.s_mgr.predict_from_file(image_path, image_name, image_file_extension)
        self.c_mass = get_center_of_mass(self.s_mask)
        self.seg_image = Image.fromarray(self.s_mask)

        left, top, right, bottom = get_image_bounds_for_given_aspect_ratio(
            self.s_mask.shape[0], self.s_mask.shape[1], self.c_mass, aspect_ratio)

        image_orig = Image.open(image_path + image_name + "." + image_file_extension)
        image_cropped = image_orig.crop((left, top, right, bottom))
        return image_cropped
