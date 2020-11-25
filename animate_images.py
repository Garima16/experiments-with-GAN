from pathlib import Path
import imageio

img_dir = Path('../dcgan-mnist-generated-images')
images = list(img_dir.glob('*.png'))
image_list = []
for file_name in images:
    image_list.append(imageio.imread(file_name))
imageio.mimwrite('dcgan_animation.gif', image_list)
