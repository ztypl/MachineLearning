import random
import io
import tkinter
from PIL import Image, ImageChops
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from sklearn import datasets, svm, metrics

# The digits dataset
digits = datasets.load_digits()

# The data that we are interested in is made of 8x8 images of digits, let's
# have a look at the first 4 images, stored in the `images` attribute of the
# dataset.  If we were working from image files, we could load them using
# matplotlib.pyplot.imread.  Note that each image must have the same size. For these
# images, we know which digit they represent: it is given in the 'target' of
# the dataset.

images_and_labels = list(zip(digits.images, digits.target))

# for index, (image, label) in enumerate(images_and_labels[:4]):
#     plt.subplot(2, 4, index + 1)
#     plt.axis('off')
#     plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
#     plt.title('Training: %i' % label)

# To apply a classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

# Create a classifier: a support vector classifier
classifier = svm.SVC(gamma=0.001)

# We learn the digits on the first half of the digits
classifier.fit(data[:n_samples // 2], digits.target[:n_samples // 2])

# Now predict the value of the digit on the second half:
expected = digits.target[n_samples // 2:]
predicted = classifier.predict(data[n_samples // 2:])



print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

images_and_predictions = list(zip(digits.images[n_samples // 2:], predicted, expected))

random.shuffle(images_and_predictions)

for index, (image, prediction, expectation) in enumerate(images_and_predictions[:12]):
    plt.subplot(3, 4, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Prediction: %i\nExpectation: %i' % (prediction, expectation))


plt.show()

#
# class Demo:
#     def __init__(self):
#         self.window = tkinter.Tk()
#         self.cv = tkinter.Canvas(self.window, bg='white')
#         self.cv.pack()
#         self.bt_clear = tkinter.Button(self.window, text="Clear", command=self.clear_canvas)
#         self.bt_clear.pack()
#
#         self.to_post = tkinter.Button(self.window, text="Post", command=self.post)
#         self.to_post.pack()
#
#         self.width = 6
#         self.old_x = 0
#         self.old_y = 0
#
#         self.cv.bind("<Button-1>", self.mouse_down_event)
#         self.cv.bind("<B1-Motion>", self.mouse_move_event)
#         self.cv.focus_set()
#
#         self.window.mainloop()
#
#     def mouse_down_event(self, event):
#         self.old_x = event.x
#         self.old_y = event.y
#
#     def mouse_move_event(self, event):
#         # self.cv.create_oval(event.x-self.width, event.y-self.width,
#         #              event.x+self.width, event.y+self.width, fill='black')
#         x = event.x
#         y = event.y
#         if x != self.old_x and y != self.old_y:
#             self.cv.create_line(self.old_x, self.old_y, x, y, fill='black', width=self.width)
#             self.cv.create_oval(x - self.width / 2, y - self.width / 2,
#                                 x + self.width / 2, y + self.width / 2, fill='black')
#             self.old_x = x
#             self.old_y = y
#
#     def clear_canvas(self):
#         self.cv.delete(tkinter.ALL)
#
#     def post(self):
#         ps = self.cv.postscript()
#         im = Image.open(io.BytesIO(ps.encode('utf-8')))
#         im = autoCrop(im, 'white')
#         im.convert('L')
#         x = max(im.size)
#         im = im.resize((x, x))
#         im.thumbnail((8, 8))
#
#         im2 = Image.new('L', (8, 8), color='white')
#         im2.paste(im)
#
#         data = (255 - np.asarray(im2).astype('float64'))//16
#         data = data.reshape((1, data.size))
#         y = classifier.predict(data)
#         print(y)
#
#         #print(im2.size)
#         # im2.show()
#
#
# def autoCrop(image, backgroundColor=None):
#     '''Intelligent automatic image cropping.
#        This functions removes the usless "white" space around an image.
#
#        If the image has an alpha (tranparency) channel, it will be used
#        to choose what to crop.
#
#        Otherwise, this function will try to find the most popular color
#        on the edges of the image and consider this color "whitespace".
#        (You can override this color with the backgroundColor parameter)
#
#        Input:
#             image (a PIL Image object): The image to crop.
#             backgroundColor (3 integers tuple): eg. (0,0,255)
#                  The color to consider "background to crop".
#                  If the image is transparent, this parameters will be ignored.
#                  If the image is not transparent and this parameter is not
#                  provided, it will be automatically calculated.
#
#        Output:
#             a PIL Image object : The cropped image.
#     '''
#
#     def mostPopularEdgeColor(image):
#         ''' Compute who's the most popular color on the edges of an image.
#             (left,right,top,bottom)
#
#             Input:
#                 image: a PIL Image object
#
#             Ouput:
#                 The most popular color (A tuple of integers (R,G,B))
#         '''
#         im = image
#         if im.mode != 'RGB':
#             im = image.convert("RGB")
#
#         # Get pixels from the edges of the image:
#         width, height = im.size
#         left = im.crop((0, 1, 1, height - 1))
#         right = im.crop((width - 1, 1, width, height - 1))
#         top = im.crop((0, 0, width, 1))
#         bottom = im.crop((0, height - 1, width, height))
#         pixels = left.tostring() + right.tostring() + top.tostring() + bottom.tostring()
#
#         # Compute who's the most popular RGB triplet
#         counts = {}
#         for i in range(0, len(pixels), 3):
#             RGB = pixels[i] + pixels[i + 1] + pixels[i + 2]
#             if RGB in counts:
#                 counts[RGB] += 1
#             else:
#                 counts[RGB] = 1
#
#         # Get the colour which is the most popular:
#         mostPopularColor = sorted([(count, rgba) for (rgba, count) in counts.items()], reverse=True)[0][1]
#         return ord(mostPopularColor[0]), ord(mostPopularColor[1]), ord(mostPopularColor[2])
#
#     bbox = None
#
#     # If the image has an alpha (tranparency) layer, we use it to crop the image.
#     # Otherwise, we look at the pixels around the image (top, left, bottom and right)
#     # and use the most used color as the color to crop.
#
#     # --- For transparent images -----------------------------------------------
#     if 'A' in image.getbands():  # If the image has a transparency layer, use it.
#         # This works for all modes which have transparency layer
#         bbox = image.split()[list(image.getbands()).index('A')].getbbox()
#     # --- For non-transparent images -------------------------------------------
#     elif image.mode == 'RGB':
#         if not backgroundColor:
#             backgroundColor = mostPopularEdgeColor(image)
#         # Crop a non-transparent image.
#         # .getbbox() always crops the black color.
#         # So we need to substract the "background" color from our image.
#         bg = Image.new("RGB", image.size, backgroundColor)
#         diff = ImageChops.difference(image, bg)  # Substract background color from image
#         bbox = diff.getbbox()  # Try to find the real bounding box of the image.
#     else:
#         raise NotImplementedError("Sorry, this function is not implemented yet for images in mode '%s'." % image.mode)
#
#     if bbox:
#         image = image.crop(bbox)
#
#     return image
#
#
# Demo()
