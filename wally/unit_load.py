'''
Test the load loading
'''
import imageio
import skimage.color as color

face = imageio.imread('wally/images/wally_template_castle_edit.png')
face = color.rgb2gray(face)
tx, ty = face.shape
print("Face Shape:", face.shape)
scene = imageio.imread('wally/images/Wally_castle.jpg')
scene = color.rgb2gray(scene)
sx, sy = scene.shape
print("Scene Shape:", scene.shape)

#plot
import matplotlib.pyplot as plt

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 10))

plt.gray()
plt.tight_layout()

ax.imshow(scene, interpolation="nearest")
ax.axis('off')
ax.set_title('Image')

plt.show()
