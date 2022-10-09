from PIL import Image
import os

image_dir = r'C:\Users\Neo\Desktop\1\images2'
save_dir = r'C:\Users\Neo\Desktop\1\images'
target_resolution = 1080
image_list = os.listdir(image_dir)
image_list = [os.path.join(image_dir, name) for name in image_list]

for image_path in image_list:
    im = Image.open(image_path)
    w, h = im.size

    if w < h:
        scale = w / target_resolution
    else:
        scale = h / target_resolution

    im = im.resize((int(w / scale), int(h / scale)))
    im.save(os.path.join(save_dir, os.path.basename(image_path)))
