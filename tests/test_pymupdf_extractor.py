import os
import unittest
from src.arxiv2report.pymupdf_extractor import PyMuPDFExtractor

class TestPyMuPDFExtractor(unittest.TestCase):

    def setUp(self):
        self.extractor = PyMuPDFExtractor(output_dir="tests/test_output")
        os.makedirs(self.extractor.output_dir, exist_ok=True)

    def tearDown(self):
        # Clean up the output directory
        # for f in os.listdir(self.extractor.output_dir):
            # os.remove(os.path.join(self.extractor.output_dir, f))
        # os.rmdir(self.extractor.output_dir)
        pass

    def test_extract_images_directly(self):
        """
        Test case for 2506.04980.pdf, which has 3 images that can be extracted directly.
        The extractor now returns a dict mapping keys like 'figure_1' to (caption, path, (w,h)).
        """
        pdf_path = "data/2506.04980.pdf"
        results = self.extractor.extract_images(pdf_path)
        self.assertIsInstance(results, dict)
        self.assertEqual(len(results), 3)
        for key, value in results.items():
            # value should be a tuple (caption, path, (w,h))
            self.assertIsInstance(value, tuple)
            self.assertEqual(len(value), 3)
            _, path, size = value
            self.assertTrue(os.path.exists(path))
            self.assertIsInstance(size, tuple)
            self.assertEqual(len(size), 2)

    def test_extract_images_by_rendering(self):
        """
        Test case for 2502.13681.pdf, which has 12 images and requires rendering.
        """
        pdf_path = "data/2502.13681.pdf"
        results = self.extractor.extract_images(pdf_path)
        self.assertIsInstance(results, dict)
        self.assertEqual(len(results), 12)
        for key, value in results.items():
            _, path, size = value
            self.assertTrue(os.path.exists(path))
            self.assertIsInstance(size, tuple)
            self.assertEqual(len(size), 2)

if __name__ == "__main__":
    unittest.main()
