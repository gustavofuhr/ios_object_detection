import cv2
import numpy as np

def resize_keep_ratio(image, 
                      new_shape, 
                      padding = False, 
                      padding_color = (114, 114, 114),
                      letter_box = False):
    """"
    Resize image keeping the aspect ratio.
    
    Args:
        image: np.array, input image
        new_shape: tuple, (height, width) of the new image
        padding: bool, add padding to the image
        padding_color: tuple, color of the padding
        letter_box: bool, use the LetterBox padding style

    Returns:
        np.array, resized image
        offset_xy if letter_box is True, tuple with the offset in x and y
    """
    h, w = image.shape[:2]
    ratio = min(new_shape[0] / h, new_shape[1] / w)
    new_h = int(h * ratio)
    new_w = int(w * ratio)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    if padding or letter_box:
        pad_h = (new_shape[0] - new_h)
        pad_w = (new_shape[1] - new_w)
        if letter_box:
            # import pdb; pdb.set_trace()
            top, bottom, left, right = pad_h//2, pad_h//2, pad_w//2, pad_w//2
            padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padding_color)
            offset = (0, pad_h//2) if pad_h > pad_w else (pad_w//2, 0)
            return padded, offset
        
        # no letterbox
        top, bottom, left, right = 0, pad_h, 0, pad_w
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padding_color)
        return padded
    else:
        return resized