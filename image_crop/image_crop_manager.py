import os
import numpy as np
import torch

from dichotomous_image_segmentation.segmentation_manager import SegmentationManager
from image_crop.pixel_segmentation.pixel_segmentation import PixelSegmentationManager
from PIL import Image
from enum import Enum

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    

class VerticalCentering(Enum):
    DO_NOT_CENTER = 1
    CENTER_ON_MASS_CENTER = 2
    CENTER_ON_MASS_CENTER_WITH_OFFSET = 3
    CENTER_ON_MASS_CENTER_WITH_ZOOM_OUT = 4


def get_center_of_mass(mask):
    """
    Computes the center of mass of a mask
    :param mask:
    :return: tuple with x and y coordinates of the center of mass of the mask

    Note: the mask is an 2D array of pixels and it comes with swapped horizontal and vertical axes.
     Therefore, the center of mass coordinates swapped as well.
    """
    rows = mask.shape[0]
    cols = mask.shape[1]
    center_of_mass = (mask * np.mgrid[0:rows, 0:cols]).sum(1).sum(1) / mask.sum()
    return tuple([round(center_of_mass[1]), round(center_of_mass[0])])


def get_cropped_image_height_and_width(image_height, image_width, center_of_mass, aspect_ratio):
    """
    Compute the height and width of the image to be cropped
    :param image_height: int, height of the image
    :param image_width: int, width of the image
    :param center_of_mass: tuple, center of mass of the image
    :param aspect_ratio: float, aspect ratio of the image
    :return: tuple with the height and width of the image to be cropped

    Details: we have computed the center of mass of the gray-scale image already.
    The image is cropped with the given aspect ratio. The center of mass is used to center the image vertically and
    horizontally. The image is cropped symmetrically around the center of mass.
    First, crop the image vertically. For example if the center of mass is in the upper half of the image, crop
    the lower half. If by doing so we cannot guarantee the given aspect ratio - that is cropped image is still too wide,
     then crop the image width so that it fit and crop additionally the image hight so that the aspect ratio is
     guaranteed. We do implement similar logic if the center of mass is in the lower half of the image.
    """
    above_mass_center = image_height - center_of_mass[1]
    below_mass_center = center_of_mass[1]
    mass_center_right = image_width - center_of_mass[0]
    mass_center_left = center_of_mass[0]
    if below_mass_center >= image_height / 2.0: # mass center in the upper half of the image
        cropped_image_height = 2 * above_mass_center
        if (mass_center_left >= aspect_ratio * above_mass_center and
                mass_center_right >= aspect_ratio * above_mass_center):
            cropped_image_width = 2 * aspect_ratio * above_mass_center
        elif mass_center_left < aspect_ratio * above_mass_center <= mass_center_right:
            cropped_image_width = 2 * mass_center_left
            cropped_image_height = 2 * mass_center_left / aspect_ratio
        elif mass_center_left >= aspect_ratio * above_mass_center > mass_center_right:
            cropped_image_width = 2 * mass_center_right
            cropped_image_height = 2 * mass_center_right / aspect_ratio
        else:  # (mass_center_left < aspect_ratio * above_mass_center and
               # mass_center_right < aspect_ratio * above_mass_center)
            cropped_image_width = 2 * min(mass_center_left, mass_center_right)
            cropped_image_height = 2 * min(mass_center_left, mass_center_right) / aspect_ratio
    else:  # below_mass_center < image_height / 2.0 - mass center in the lower half of the image
        cropped_image_height = 2 * below_mass_center
        if (mass_center_left >= aspect_ratio * below_mass_center and
                mass_center_right >= aspect_ratio * below_mass_center):
            cropped_image_width = 2 * aspect_ratio * below_mass_center
        elif mass_center_left < aspect_ratio * below_mass_center <= mass_center_right:
            cropped_image_width = 2 * mass_center_left
            cropped_image_height = 2 * mass_center_left / aspect_ratio
        elif mass_center_left >= aspect_ratio * below_mass_center > mass_center_right:
            cropped_image_width = 2 * mass_center_right
            cropped_image_height = 2 * mass_center_right / aspect_ratio
        else:  # (mass_center_left < aspect_ratio * below_mass_center and
               # mass_center_right < aspect_ratio * below_mass_center)
            cropped_image_width = 2 * min(mass_center_left, mass_center_right)
            cropped_image_height = 2 * min(mass_center_left, mass_center_right) / aspect_ratio

    return cropped_image_height, cropped_image_width


def get_image_bounds_no_vertical_centering(image_height, image_width, center_of_mass, aspect_ratio):
    """
    Compute the bounds of the image to be cropped with given aspect ratio. Do not center the image vertically
      by the mass center but offset it by the vertical offset to avoid cropping of extremities in the image
    :param image_height: int, height of the image
    :param image_width: int, width of the image
    :param center_of_mass: tuple, center of mass of the image
    :param aspect_ratio: float, aspect ratio of the image
    :return: tuple with the left, top, right, bottom coordinates of the image to be cropped
    """
    x_coord_mass = center_of_mass[0]

    cropped_image_height, cropped_image_width = get_cropped_image_height_and_width(image_height, image_width,
                                                                                   center_of_mass, aspect_ratio)

    left = round(x_coord_mass - cropped_image_width / 2.0)
    top = (image_height - cropped_image_height) / 2.0
    right = left + round(cropped_image_width)
    bottom = top + round(cropped_image_height)

    return left, top, right, bottom


def get_image_bounds_vertically_mass_centered_no_offset(image_height, image_width, center_of_mass, aspect_ratio):
    """
    Compute the bounds of the image to be cropped with given aspect ratio. Center the image vertically
      by the mass center but offset it by the vertical offset to avoid cropping of extremities in the image
    :param image_height: int, height of the image
    :param image_width: int, width of the image
    :param center_of_mass: tuple, center of mass of the image
    :param aspect_ratio: float, aspect ratio of the image
    :return: tuple with the left, top, right, bottom coordinates of the image to be cropped
    """
    x_coord_mass = center_of_mass[0]
    y_coord_mass = center_of_mass[1]

    cropped_image_height, cropped_image_width = get_cropped_image_height_and_width(image_height, image_width,
                                                                                   center_of_mass, aspect_ratio)

    left = round(x_coord_mass - cropped_image_width / 2.0)
    right = left + round(cropped_image_width)
    if y_coord_mass >= image_height / 2.0:
        top = round(y_coord_mass - cropped_image_height / 2.0)
        bottom = top + round(cropped_image_height)

    else:
        bottom = 2 * y_coord_mass
        top = bottom - cropped_image_height

    return left, top, right, bottom


def get_image_bounds_vertically_mass_centered_with_offset(image_height, image_width, center_of_mass, aspect_ratio,
                                                          pixel_segmentation_mgr=None):
    """
    Compute the bounds of the image to be cropped with given aspect ratio. Center the image vertically
      by the mass center but offset it by the vertical offset to avoid cropping of extremities in the image
    :param image_height: int, height of the image
    :param image_width: int, width of the image
    :param center_of_mass: tuple, center of mass of the image
    :param aspect_ratio: float, aspect ratio of the image
    :param pixel_segmentation_mgr: PixelSegmentationManager instance
    :return: tuple with the left, top, right, bottom coordinates of the image to be cropped

    Details: the cropping algorithm with vertical offset (cr. CENTER_ON_MASS_CENTER_WITH_OFFSET) proceeds as follows:

    First, we crop the image vertically and horizontally centering symmetrically around the center of mass.

    Second, we compute two vertical offsets - vertical_offset_top and vertical_offset_bottom. vertical_offset_top
    gives the minimal distance between the segment of interest and the top of the original image frame.
    vertical_offset_bottom gives the minimal distance between the segment of interest and the bottom of the original
     image frame.

    Third, we check if the segment of interest has been cropped out. We do this by computing the vertical cutoff as:

    vertical_cutoff = image_height - vertical_offset_top - cropped_image_height - vertical_offset_bottom

    If the vertical cutoff is greater than 0, we will be cutting off the top or the bottom of the segment of interest.
    Thus, if the vertical cutoff is positive we offset the center of mass vertically in the opposite direction of
    the closest horizontal edge. For instance, if the mass center is closer to the top edge vertically
    (given by the condition y_coord_mass >= image_height / 2.0) and vertical_cutoff > 0 then we move the cropped
    top edge (given with  round(y_coord_mass - cropped_image_height / 2.0) ) up by the vertical_cutoff.
    Thus, the offsetting of the cropped top will reduce the amount of cropping of the segment of interest in
     vertical direction.

    More details on this algorithm can be found in
    https://confluence.nike.com/display/EDAAAML/Documents+on+Image+Cropping+Algorithms
    """
    x_coord_mass = center_of_mass[0]
    y_coord_mass = center_of_mass[1]

    cropped_image_height, cropped_image_width = get_cropped_image_height_and_width(image_height, image_width,
                                                                                   center_of_mass, aspect_ratio)

    vertical_offset_top = 0
    vertical_offset_bottom = 0

    if pixel_segmentation_mgr is not None:
        vertical_offset_top, vertical_offset_bottom = (
            pixel_segmentation_mgr.compute_vertical_offset(y_coord_mass, cropped_image_height))
    left = round(x_coord_mass - cropped_image_width / 2.0)
    right = left + round(cropped_image_width)
    if y_coord_mass >= image_height / 2.0:
        top = round(y_coord_mass - cropped_image_height / 2.0)
        top_with_vertical_offset = top
        vertical_cutoff = image_height - vertical_offset_top - cropped_image_height - vertical_offset_bottom
        if vertical_cutoff > 0: # we will be cutting off the top of the segment of interest
            if vertical_cutoff < top:
                top_with_vertical_offset = top - vertical_cutoff
            else:
                top_with_vertical_offset = 0

        bottom_with_vertical_offset = top_with_vertical_offset + round(cropped_image_height)

    else:
        bottom = 2 * y_coord_mass
        bottom_with_vertical_offset = bottom
        vertical_cutoff = image_height - cropped_image_height - vertical_offset_bottom - vertical_offset_top

        if vertical_cutoff > 0: # we will be cutting off the bottom of the segment of interest
            bottom_with_vertical_offset = bottom + vertical_cutoff

        top_with_vertical_offset = bottom_with_vertical_offset - cropped_image_height

    return left, top_with_vertical_offset, right, bottom_with_vertical_offset


class ImageCropManager:
    def __init__(self, vertical_centering=VerticalCentering.CENTER_ON_MASS_CENTER_WITH_OFFSET):
        self.s_mgr = SegmentationManager(DEVICE)
        self.vertical_centering = vertical_centering
        self.c_mass = None
        self.s_mask = None
        self.im_mask = None
        self.seg_image = None

    def get_cropped_image(self, image_path, image_name, image_file_extension, aspect_ratio):
        """
        :param image_path: str, the URL of folder in which is the image to be cropped
        :param image_name: str, name of the image to be cropped
        :param image_file_extension: str
        :param aspect_ratio:
        :return PIL::Image instance of the cropped image
        """

        self.s_mask = self.s_mgr.predict_from_file(image_path, image_name, image_file_extension)
        ps_mgr = PixelSegmentationManager(DEVICE, self.s_mask)
        self.c_mass = get_center_of_mass(self.s_mask)
        self.seg_image = Image.fromarray(self.s_mask)

        left = None
        top = None
        right = None
        bottom = None
        if self.vertical_centering == VerticalCentering.CENTER_ON_MASS_CENTER_WITH_OFFSET:
            left, top, right, bottom = get_image_bounds_vertically_mass_centered_with_offset(
                self.s_mask.shape[0], self.s_mask.shape[1], self.c_mass, aspect_ratio, pixel_segmentation_mgr=ps_mgr)
        elif self.vertical_centering == VerticalCentering.CENTER_ON_MASS_CENTER:
            left, top, right, bottom = get_image_bounds_vertically_mass_centered_no_offset(
                self.s_mask.shape[0], self.s_mask.shape[1], self.c_mass, aspect_ratio)
        elif self.vertical_centering == VerticalCentering.DO_NOT_CENTER:
            left, top, right, bottom = get_image_bounds_no_vertical_centering(
                self.s_mask.shape[0], self.s_mask.shape[1], self.c_mass, aspect_ratio)
        elif self.vertical_centering == VerticalCentering.CENTER_ON_MASS_CENTER_WITH_ZOOM_OUT:
            raise NotImplementedError("VerticalCentering.CENTER_ON_MASS_CENTER_WITH_ZOOM_OUT is not implemented yet")

        image_orig = Image.open(image_path + image_name + "." + image_file_extension)
        image_cropped = image_orig.crop((left, top, right, bottom))
        return image_cropped
