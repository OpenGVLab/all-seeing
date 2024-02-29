import os
from PIL import Image
from tqdm import tqdm

DEFAULT_BOX_SCALE = 999
box_start_tag = '<box>'
box_end_tag = '</box>'

img_path = 'playground/data/coco/train2014'
flip_img_path = 'playground/data/coco_flip/train2014'
os.makedirs(flip_img_path, exist_ok=True)

for filename in tqdm(os.listdir(img_path)):
    image = os.path.join(img_path, filename)
    image_flip = os.path.join(flip_img_path, os.path.basename(image))

    if not os.path.exists(image_flip):
        image_pil = Image.open(image).transpose(Image.FLIP_LEFT_RIGHT)
        image_pil.save(image_flip)
