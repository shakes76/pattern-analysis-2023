'''
Find wally in a scene
'''
import imageio
import skimage.color as color
import numpy as np

face = imageio.imread('wally/images/wally_template_castle_edit.png')
face = color.rgb2gray(face)
tx, ty = face.shape
print("Face Shape:", face.shape)
scene = imageio.imread('wally/images/Wally_castle.jpg')
scene = color.rgb2gray(scene)
sx, sy = scene.shape
print("Scene Shape:", scene.shape)

def apply_template(center, template, image):
    '''
    Applies NCC between template and image patch at center
    '''
    r, c = center
    image_shape = template.shape
    r_pad, c_pad = [(size - 1) // 2 for size in image_shape]
    patch = image[[slice(r-r_pad, r+r_pad+1), slice(c-c_pad, c+c_pad+1)]]
    patch = patch / np.linalg.norm(patch)
    template = template / np.linalg.norm(template)
    return np.sum(patch * template)

filteredImage = np.zeros_like(scene)
for xIndex, row in enumerate(scene):
    if xIndex <= int(tx/2) or xIndex > (sx-int(tx/2)-1):
        continue
    for yIndex, coloum in enumerate(row):
        if yIndex <= int(ty/2) or yIndex > (sy-int(ty/2)-1):
            continue
        center = (xIndex, yIndex)
        #print(center)
        filteredImage[center] = apply_template(center, face, scene)

#plot
import matplotlib.pyplot as plt

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 10))

plt.gray()
plt.tight_layout()

ax[0].imshow(scene, interpolation="nearest")
ax[0].axis('off')
ax[0].set_title('Image')
ax[1].imshow(filteredImage, interpolation="nearest")
ax[1].axis('off')
ax[1].set_title('Cross Correlations')

plt.show()