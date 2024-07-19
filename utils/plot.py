import cv2
import numpy as np

def draw_text(img, text,
          font=cv2.FONT_HERSHEY_PLAIN,
          pos=(0, 0),
          font_scale=3,
          font_thickness=2,
          text_color=(0, 255, 0),
          text_color_bg=(0, 0, 0)
          ):

    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, pos, (x + text_w, y + text_h), text_color_bg, -1)
    cv2.putText(img, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)

    return text_size

def plot_detections(im, detections, detector_size = None):
    # im = cv2.imread(image_path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    for detection in detections:
        box = detection["box"]
        
        if detector_size is not None:
            # scale box according to detector_size
            scale_x = im.shape[1] / detector_size[0]
            scale_y = im.shape[0] / detector_size[1]
            box = [box[0] * scale_x, box[1] * scale_y, box[2] * scale_x, box[3] * scale_y]
        
        label = detection["label"]
        score = detection["score"]
        cv2.rectangle(im, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)

        label_position = (int(box[0]), int(box[1]) - 10)

        draw_text(im, f"{label} {score:.2f}", font_scale=1, font_thickness=1, pos=label_position, text_color_bg=(255, 0, 0), 
                  text_color=(255, 255, 255))
    return im