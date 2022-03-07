import unittest

from src.data_utils.utils import get_image_files, get_attributes


class TestGetImageFiles(unittest.TestCase):
    def test_length(self):
        file_list, labels = get_image_files()
        self.assertEqual(len(file_list), 203,
                         "Unexpected number of file names")
        self.assertEqual(len(labels), len(file_list),
                         "file count and label count don't match")


class TestGetAttributes(unittest.TestCase):
    def test_shape(self):
        attr = get_attributes()
        self.assertEqual(attr.shape, (359, 8))