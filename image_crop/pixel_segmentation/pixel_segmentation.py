import numpy as np
from skimage import data, segmentation, feature, future
from sklearn.ensemble import RandomForestClassifier as sklRFC
from functools import partial
from statistics import mean
from enum import Enum


class OffsetColorInclusionSet(Enum):
    ONLY_DARKEST = 1
    WITH_MEDIUM_GREY = 2
    WITH_MOST_POPULAR_SHADE_OF_GRAY = 3


def get_frequencies(gray_scale_image):
    """
    Compute the frequencies of the different pixel values in the image
    """
    v, c = np.unique(gray_scale_image, return_counts=True)
    frequencies = dict(zip(v, c))
    return frequencies


def compute_statistics(frequencies):
    """
    Compute the range, mean, median, mode of the pixel values in the image
    """
    sorted_by_frequency = sorted(frequencies.items(), key=lambda x: x[1])
    sorted_by_value = sorted(frequencies.items(), key=lambda x: x[0])
    max_val = sorted_by_value[-1][0]
    min_val = sorted_by_value[0][0]
    pixel_range = sorted_by_value[-1][0] - sorted_by_value[0][0]
    pixel_mean = mean([x[0] for x in sorted_by_value])
    pixel_median = sorted_by_value[len(sorted_by_value) // 2][0]
    pixel_mode = sorted_by_frequency[-1][0]

    statistics = dict()
    statistics["range"] = pixel_range
    statistics["mean"] = pixel_mean
    statistics["median"] = pixel_median
    statistics["mode"] = pixel_mode
    statistics["max"] = max_val
    statistics["min"] = min_val
    statistics["values_and_frequencies"] = sorted_by_value
    return statistics


class PixelSegmentationManager:
    """
    Class to manage the segmentation of the grayscale image mask.
    This class contains two sub-algorithms both of which compute two vertical offsets - vertical_offset_top and
    vertical_offset_bottom. vertical_offset_top represents the minimal distance between the top of the original image
     frame and the segment of interest in the image identified by the supplied segmentation mask.
    vertical_offset_bottom represents the minimal distance between the bottom of the original image frame and
      the segment of interest.

    1) _compute_vertical_offset_using_fine_estimate uses Random Forest Classifier to segment the grayscale image mask
    into tri states (dark, gray, light) and then uses the segmentation to compute the vertical offsets.
    2) _compute_vertical_offset_using_coarse_estimate uses fast heuristics to compute the vertical offsets.

    """
    DARK_THRESHOLD = 200
    DARK_REGION_SEGMENT = 1
    MEDIAN_GRAY_REGION_SEGMENT = 2
    MOST_POPULAR_SHADE_OF_GRAY_SEGMENT = 3
    LIGHT_REGION_SEGMENT = 4

    def __init__(self, device, segmentation_mask, offset_type=OffsetColorInclusionSet.ONLY_DARKEST,
                 use_fine_segmentation_estimate=True):
        """
        Initialize the pixel segmentation manager and set the segmentation mask
        :param device: str
        :param segmentation_mask: numpy array
        :param offset_type: OffsetColorInclusionSet Enum
        :param use_fine_segmentation_estimate: Boolean Enables the use of the algorithm delivering fine segmentation
         estimate; if set to False, the coarse estimate heuristics is used.
        """
        self.device = device
        self.segmentation_mask = segmentation_mask
        self.use_fine_segmentation_estimate = use_fine_segmentation_estimate
        self.offset_type = offset_type
        self.clf = None
        self.features = None
        self.features_func = None
        self.sigma_min = 1
        self.sigma_max = 16
        self.center_of_mass_y = None
        self.cropped_image_height = None
        self.training_labels = None
        self.frequencies = None
        self.statistics = None
        self.segmentation_result = None

    def _estimate_and_label_gray_area(self):
        """
        Estimate the gray area of the image. This method is part of the fine estimate algorithm for obtaining the
        top and bottom vertical offsets.
        :return: None
        """
        if len(self.statistics["values_and_frequencies"]) > 1:
            darkest_pixel = self.statistics["values_and_frequencies"][0][0]
            lightest_pixel = self.statistics["values_and_frequencies"][-1][0]
            median_pixel = self.statistics["median"]
            mode_pixel = self.statistics["mode"]
            if darkest_pixel < median_pixel < lightest_pixel:
                # we want to estimate the median gray area of the image
                # and label all elements of the set of median gray pixels with 2
                x, y = np.where(self.segmentation_mask == median_pixel)
                self.training_labels[x, y] = self.MEDIAN_GRAY_REGION_SEGMENT

            if mode_pixel not in (darkest_pixel, median_pixel, lightest_pixel):
                x, y = np.where(self.segmentation_mask == mode_pixel)
                self.training_labels[x, y] = self.MOST_POPULAR_SHADE_OF_GRAY_SEGMENT

    def _estimate_and_label_image_segment(self):
        """
        Estimate and label the segment of interest in the image.
        This method is part of the fine estimate algorithm for obtaining the top and bottom vertical offsets.
        :return: None
        """
        lightest_pixel = self.statistics["values_and_frequencies"][-1][0]
        x, y = np.where(self.segmentation_mask == lightest_pixel)
        self.training_labels[x, y] = self.LIGHT_REGION_SEGMENT

    def _estimate_and_label_black_region(self):
        """
        Compute the minimum distance between the segment of interest and the horizontal edge opposite to the edge
        closest to the mass center of the image.
        This method is part of the fine estimate algorithm for obtaining the top and bottom vertical offsets.
        :return: Boolean
        """
        black_height = 0
        if len(self.statistics["values_and_frequencies"]) > 1:
            lightest_pixel = self.statistics["values_and_frequencies"][-1][0]
            dark_pixel_index = 0
            dark_region_on_top = True
            dark_pixel = self.statistics["values_and_frequencies"][0][0]
            darkest_pixels = [dark_pixel]

            if self.center_of_mass_y >= self.segmentation_mask.shape[0] / 2.0:
                # we want to estimate the height of the dark region of the top
                # and label all rows from top to that height with 1
                mask_array = self.segmentation_mask
            else:
                # we want to estimate the height of the black region of the bottom
                # and label all rows from bottom to that height with 1
                dark_region_on_top = False
                mask_array = self.segmentation_mask[::-1]

            while lightest_pixel - dark_pixel > self.DARK_THRESHOLD:

                for row in mask_array:
                    if all(pixel in darkest_pixels for pixel in row):
                        black_height += 1
                    else:
                        break

                if black_height == 0:
                    dark_pixel_index += 1
                else:
                    break
                dark_pixel = self.statistics["values_and_frequencies"][dark_pixel_index][0]
                darkest_pixels.append(dark_pixel)

            if black_height > 0:
                if dark_region_on_top:
                    # label all rows from top to black_height with 1
                    self.training_labels[:black_height] = self.DARK_REGION_SEGMENT
                else:
                    # label all rows from bottom to black_height with 1
                    self.training_labels[-black_height:] = self.DARK_REGION_SEGMENT

        return black_height

    def _set_training_labels(self):
        """
        Set the training labels for the segmentation algorithm
         This method is part of the fine estimate algorithm for obtaining the top and bottom vertical offsets.
        :return: Boolean
        """
        (n_x, n_y) = self.segmentation_mask.shape[:2]

        self.training_labels = np.zeros((n_x, n_y), dtype=np.uint8)

        self.frequencies = get_frequencies(self.segmentation_mask)
        self.statistics = compute_statistics(self.frequencies)
        black_height = self._estimate_and_label_black_region()

        if black_height > 0:
            self._estimate_and_label_gray_area()
            self._estimate_and_label_image_segment()
            return True
        else:
            return False

    def _train_segmentation(self, segmentation_mask, center_of_mass_y, cropped_image_height):
        """
        Train the segmentation algorithm
         This method is part of the fine estimate algorithm for obtaining the top and bottom vertical offsets.
        :return: Boolean
        """
        self.segmentation_mask = segmentation_mask
        self.center_of_mass_y = center_of_mass_y
        self.cropped_image_height = cropped_image_height

        if self._set_training_labels():
            self.features_func = partial(
                feature.multiscale_basic_features,
                intensity=True,
                edges=False,
                texture=True,
                sigma_min=self.sigma_min,
                sigma_max=self.sigma_max,
                #channel_axis=-1,
            )
            self.features = self.features_func(self.segmentation_mask)
            if self.device == "cuda":
                from cuml import RandomForestClassifier as cumlRFC
                self.clf = self.clf = cumlRFC(n_estimators=25, max_depth=13, n_bins=15, n_streams=8)
            else:
                self.clf = sklRFC(n_estimators=50, n_jobs=-1, max_depth=10, max_samples=0.05)
            self.clf = future.fit_segmenter(self.training_labels, self.features, self.clf)
            return True
        else:
            return False

    def _predict_segmentation(self, segmentation_mask, center_of_mass_y, cropped_image_height):
        """
        Infer the segmentation of the grayscale segmentation mask
         This method is part of the fine estimate algorithm for obtaining the top and bottom vertical offsets.
        :return: Boolean
        """
        if self._train_segmentation(segmentation_mask, center_of_mass_y, cropped_image_height):
            self.segmentation_result = future.predict_segmenter(self.features, self.clf)
            return True
        else:
            return False

    def _compute_vertical_offset_using_fine_estimate(self, center_of_mass_y, cropped_image_height):
        """
        Compute the top and bottom vertical offsets of the segment of interest.
        This is the main method of the fine estimate algorithm for obtaining the top and bottom vertical offsets.
        """
        vertical_offset_top = 0
        vertical_offset_bottom = 0

        if self._predict_segmentation(self.segmentation_mask, center_of_mass_y, cropped_image_height):
            reversed_segmentation_result = self.segmentation_result[::-1]
            if self.offset_type == OffsetColorInclusionSet.ONLY_DARKEST:
                for row in self.segmentation_result:
                    if not all(pixel_seg_type == self.DARK_REGION_SEGMENT for pixel_seg_type in row):
                        break
                    vertical_offset_top += 1

                for row in reversed_segmentation_result:
                    if not all(pixel_seg_type == self.DARK_REGION_SEGMENT for pixel_seg_type in row):
                        break
                    vertical_offset_bottom += 1
            elif self.offset_type == OffsetColorInclusionSet.WITH_MEDIUM_GREY:
                for row in self.segmentation_result:
                    if not all(pixel_seg_type == self.DARK_REGION_SEGMENT or
                               pixel_seg_type == self.MEDIAN_GRAY_REGION_SEGMENT for pixel_seg_type in row):
                        break
                    vertical_offset_top += 1

                for row in reversed_segmentation_result:
                    if not all(pixel_seg_type == self.DARK_REGION_SEGMENT or
                               pixel_seg_type == self.MEDIAN_GRAY_REGION_SEGMENT for pixel_seg_type in row):
                        break
                vertical_offset_bottom += 1
            elif self.offset_type == OffsetColorInclusionSet.WITH_MOST_POPULAR_SHADE_OF_GRAY:
                for row in self.segmentation_result:
                    if not all(pixel_seg_type == self.DARK_REGION_SEGMENT or
                               pixel_seg_type == self.MEDIAN_GRAY_REGION_SEGMENT or
                               pixel_seg_type == self.MOST_POPULAR_SHADE_OF_GRAY_SEGMENT for pixel_seg_type in row):
                        break
                    vertical_offset_top += 1

                for row in reversed_segmentation_result:
                    if not all(pixel_seg_type == self.DARK_REGION_SEGMENT or
                               pixel_seg_type == self.MEDIAN_GRAY_REGION_SEGMENT or
                               pixel_seg_type == self.MOST_POPULAR_SHADE_OF_GRAY_SEGMENT for pixel_seg_type in row):
                        break
                    vertical_offset_bottom += 1

        return vertical_offset_top, vertical_offset_bottom

    def _compute_vertical_offset(self, threshold, mask_array):
        """
         Compute the vertical offset of the segmentation mask given the threshold value
         This method is part of the coarse estimate algorithm for obtaining the top and bottom vertical offsets.
        """
        vertical_offset = 0
        dark_pixel_index = 0
        dark_pixel = self.statistics["values_and_frequencies"][dark_pixel_index][0]
        lightest_pixel = self.statistics["values_and_frequencies"][-1][0]
        darkest_pixels = [dark_pixel]
        while lightest_pixel - dark_pixel > threshold:
            for row in mask_array:
                if all(pixel in darkest_pixels for pixel in row):
                    vertical_offset += 1
                else:
                    break

            if vertical_offset == 0:
                dark_pixel_index += 1
            else:
                break

            dark_pixel = self.statistics["values_and_frequencies"][dark_pixel_index][0]
            darkest_pixels.append(dark_pixel)

        return vertical_offset

    def _estimate_black_and_grey_region_height(self):
        """
        Estimate the height of the black and gray regions of the image
        This method is part of the coarse estimate algorithm for obtaining the top and bottom vertical offsets.
        """
        vertical_offset_top = 0
        vertical_offset_bottom = 0
        if len(self.statistics["values_and_frequencies"]) > 1:

            reversed_segmentation_mask = self.segmentation_mask[::-1]
            lightest_pixel = self.statistics["values_and_frequencies"][-1][0]
            darkest_pixel = self.statistics["values_and_frequencies"][0][0]
            threshold = None
            if self.offset_type == OffsetColorInclusionSet.ONLY_DARKEST:
                threshold = self.DARK_THRESHOLD
            elif self.offset_type == OffsetColorInclusionSet.WITH_MEDIUM_GREY:
                median = self.statistics["median"]
                if darkest_pixel < median < lightest_pixel:
                    threshold = lightest_pixel - self.statistics["median"]
                else:
                    threshold = lightest_pixel - darkest_pixel
            elif self.offset_type == OffsetColorInclusionSet.WITH_MOST_POPULAR_SHADE_OF_GRAY:
                mode = self.statistics["mode"]
                if darkest_pixel < mode < lightest_pixel:
                    threshold = lightest_pixel - self.statistics["mode"]
                else:
                    threshold = lightest_pixel - darkest_pixel

            vertical_offset_top = self._compute_vertical_offset(threshold, self.segmentation_mask)

            vertical_offset_bottom = self._compute_vertical_offset(threshold, reversed_segmentation_mask)

        return vertical_offset_top, vertical_offset_bottom

    def _compute_vertical_offset_using_coarse_estimate(self, center_of_mass_y, cropped_image_height):
        """
        Compute the top and bottom vertical offsets of the segment of interest.
         This is the main method of the coarse estimate algorithm for obtaining the top and bottom vertical offsets.
        :param center_of_mass_y: int
        :param cropped_image_height: int
        """
        self.center_of_mass_y = center_of_mass_y
        self.cropped_image_height = cropped_image_height
        self.frequencies = get_frequencies(self.segmentation_mask)
        self.statistics = compute_statistics(self.frequencies)
        return self._estimate_black_and_grey_region_height()

    def compute_vertical_offset(self, center_of_mass_y, cropped_image_height):
        """
        Compute the vertical offset of the segmentation mask
        :return: tuple of integers - darkest_region_offset, dark_and_grey_region_offset
        """
        if self.use_fine_segmentation_estimate:
            return self._compute_vertical_offset_using_fine_estimate(center_of_mass_y, cropped_image_height)
        else:
            return self._compute_vertical_offset_using_coarse_estimate(center_of_mass_y, cropped_image_height)

