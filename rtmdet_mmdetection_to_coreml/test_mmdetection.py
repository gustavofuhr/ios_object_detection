import os, random

from mmdet.apis import init_detector, inference_detector

# Specify the path to model config and checkpoint file
config_file = 'third_party/mmdetection/configs/rtmdet/rtmdet_tiny_8xb32-300e_coco.py'
checkpoint_file = 'rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth'
mm_model = init_detector(config_file, checkpoint_file, device='cpu')
mmdetection_model = lambda image_file: inference_detector(mm_model, image_file)

# image_files = [os.path.join("../sample_images/",f) for f in os.listdir("../sample_images/") \
                                # if f.endswith(".jpg") or f.endswith(".jpeg") or f.endswith(".png")]
# r_image_file = random.choice(image_files)
r_image_file = "../sample_images/2008_000012.jpg"
print(f"Comparing models for {r_image_file}")

# get inference from mmdetection
pred_mmdetection = mmdetection_model(r_image_file)

