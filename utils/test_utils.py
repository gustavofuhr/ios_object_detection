import unittest

import cv2
import numpy as np

from image_proc import resize_keep_ratio

class TestUtils(unittest.TestCase):
    
    def test_resize_keep_ratio(self):
        # height > width
        im = cv2.imread("../sample_images/demo_portrait.jpg")
        self.assertEqual(im.shape, (382, 303, 3))
        w_expected = int((800/382)*303)
        im_r = resize_keep_ratio(im, (800, 800))
        self.assertEqual(im_r.shape, (800, w_expected, 3))

        im_rp = resize_keep_ratio(im, (800, 800), padding = True)
        self.assertEqual(im_rp.shape, (800, 800, 3))
        self.assertTrue(np.all(im_rp[:,w_expected:,:] == 114))

        im_rlb, offset_xy = resize_keep_ratio(im, (800, 800), padding = True, letter_box=True)
        self.assertEqual(im_rp.shape, (800, 800, 3))
        self.assertEqual(offset_xy[0], (800-w_expected)//2)
        self.assertEqual(offset_xy[1], 0)
        self.assertTrue(np.all(im_rlb[:,:offset_xy[0],:] == 114))
        self.assertTrue(np.all(im_rlb[:,-offset_xy[0],:] == 114))
        
        # width > height
        im = cv2.imread("../sample_images/demo.jpg")
        self.assertEqual(im.shape, (427, 640, 3))
        h_expected = int((800/640)*427)
        im_r = resize_keep_ratio(im, (800, 800))
        self.assertEqual(im_r.shape, (h_expected, 800, 3))

        im_rp = resize_keep_ratio(im, (800, 800), padding = True)
        self.assertEqual(im_rp.shape, (800, 800, 3))
        self.assertTrue(np.all(im_rp[h_expected:,:,:] == 114))

        im_rlb, offset_xy = resize_keep_ratio(im, (800, 800), padding = True, letter_box=True)
        self.assertEqual(im_rp.shape, (800, 800, 3))
        self.assertEqual(offset_xy[1], (800-h_expected)//2)
        self.assertEqual(offset_xy[0], 0)
        self.assertTrue(np.all(im_rlb[:offset_xy[1],:,:] == 114))
        self.assertTrue(np.all(im_rlb[-offset_xy[1]:,:,:] == 114))


if __name__ == '__main__':  
    unittest.main()