# coding : utf-8
# create by ztypl on 2017/9/5

import io
import pickle
import tkinter
from PIL import Image, ImageChops
import numpy as np

from sklearn import datasets, svm, metrics

class Demo:
    def __init__(self):
        self.filename = 'data.pkl'
        self.classifier = None

        try:
            x = open(self.filename, 'rb')
            self.train_dataset = pickle.load(x)
        except FileNotFoundError:
            self.train_dataset = {'data':[],'target':[]}

        self.window = tkinter.Tk()
        self.cv = tkinter.Canvas(self.window, bg='white')
        self.cv.pack()
        self.bt_clear = tkinter.Button(self.window, text="Clear", command=self.clear_canvas)
        self.bt_clear.pack()

        self.text = tkinter.Entry(self.window)
        self.text.pack()

        self.bt_train = tkinter.Button(self.window, text="Train", command=self.train)
        self.bt_train.pack()

        self.bt_train_all = tkinter.Button(self.window, text="Train ALL", command=self.train_all)
        self.bt_train_all.pack()

        self.to_post = tkinter.Button(self.window, text="Post", command=self.post)
        self.to_post.pack()

        self.width = 6
        self.old_x = 0
        self.old_y = 0

        self.cv.bind("<Button-1>", self.mouse_down_event)
        self.cv.bind("<B1-Motion>", self.mouse_move_event)
        self.cv.focus_set()

        self.window.mainloop()

        pickle.dump(self.train_dataset, open(self.filename,'wb'))

    def mouse_down_event(self, event):
        self.old_x = event.x
        self.old_y = event.y

    def mouse_move_event(self, event):
        # self.cv.create_oval(event.x-self.width, event.y-self.width,
        #              event.x+self.width, event.y+self.width, fill='black')
        x = event.x
        y = event.y
        if x != self.old_x and y != self.old_y:
            self.cv.create_line(self.old_x, self.old_y, x, y, fill='black', width=self.width)
            self.cv.create_oval(x - self.width / 2, y - self.width / 2,
                                x + self.width / 2, y + self.width / 2, fill='black')
            self.old_x = x
            self.old_y = y

    def clear_canvas(self):
        self.cv.delete(tkinter.ALL)

    def train(self):
        num = int(self.text.get())
        ps = self.cv.postscript()
        im = Image.open(io.BytesIO(ps.encode('utf-8')))
        im = autoCrop(im, 'white')
        im.convert('L')
        x = max(im.size)
        im = im.resize((x, x))
        im.thumbnail((20, 20))

        data = 255 - np.asarray(im).astype('float64')
        data = list(data.reshape((1, data.size)))[0]

        self.train_dataset['data'].append(data)
        self.train_dataset['target'].append(num)

        self.clear_canvas()
        self.text.delete(0,tkinter.END)


    def train_all(self):
        self.classifier = svm.SVC(gamma=0.001)
        self.classifier.fit(self.train_dataset['data'], self.train_dataset['target'])

    def post(self):
        if self.classifier is None:
            self.classifier = svm.SVC(gamma=0.001)

        ps = self.cv.postscript()
        im = Image.open(io.BytesIO(ps.encode('utf-8')))
        im = autoCrop(im, 'white')
        im.convert('L')
        x = max(im.size)
        im = im.resize((x, x))
        im.thumbnail((20, 20))


        data = 255 - np.asarray(im).astype('float64')
        data = data.reshape((1, data.size))
        y = self.classifier.predict(data)
        print(y)

        #print(im2.size)
        # im2.show()


def autoCrop(image, backgroundColor=None):
    '''Intelligent automatic image cropping.
       This functions removes the usless "white" space around an image.

       If the image has an alpha (tranparency) channel, it will be used
       to choose what to crop.

       Otherwise, this function will try to find the most popular color
       on the edges of the image and consider this color "whitespace".
       (You can override this color with the backgroundColor parameter)

       Input:
            image (a PIL Image object): The image to crop.
            backgroundColor (3 integers tuple): eg. (0,0,255)
                 The color to consider "background to crop".
                 If the image is transparent, this parameters will be ignored.
                 If the image is not transparent and this parameter is not
                 provided, it will be automatically calculated.

       Output:
            a PIL Image object : The cropped image.
    '''

    def mostPopularEdgeColor(image):
        ''' Compute who's the most popular color on the edges of an image.
            (left,right,top,bottom)

            Input:
                image: a PIL Image object

            Ouput:
                The most popular color (A tuple of integers (R,G,B))
        '''
        im = image
        if im.mode != 'RGB':
            im = image.convert("RGB")

        # Get pixels from the edges of the image:
        width, height = im.size
        left = im.crop((0, 1, 1, height - 1))
        right = im.crop((width - 1, 1, width, height - 1))
        top = im.crop((0, 0, width, 1))
        bottom = im.crop((0, height - 1, width, height))
        pixels = left.tostring() + right.tostring() + top.tostring() + bottom.tostring()

        # Compute who's the most popular RGB triplet
        counts = {}
        for i in range(0, len(pixels), 3):
            RGB = pixels[i] + pixels[i + 1] + pixels[i + 2]
            if RGB in counts:
                counts[RGB] += 1
            else:
                counts[RGB] = 1

        # Get the colour which is the most popular:
        mostPopularColor = sorted([(count, rgba) for (rgba, count) in counts.items()], reverse=True)[0][1]
        return ord(mostPopularColor[0]), ord(mostPopularColor[1]), ord(mostPopularColor[2])

    bbox = None

    # If the image has an alpha (tranparency) layer, we use it to crop the image.
    # Otherwise, we look at the pixels around the image (top, left, bottom and right)
    # and use the most used color as the color to crop.

    # --- For transparent images -----------------------------------------------
    if 'A' in image.getbands():  # If the image has a transparency layer, use it.
        # This works for all modes which have transparency layer
        bbox = image.split()[list(image.getbands()).index('A')].getbbox()
    # --- For non-transparent images -------------------------------------------
    elif image.mode == 'RGB':
        if not backgroundColor:
            backgroundColor = mostPopularEdgeColor(image)
        # Crop a non-transparent image.
        # .getbbox() always crops the black color.
        # So we need to substract the "background" color from our image.
        bg = Image.new("RGB", image.size, backgroundColor)
        diff = ImageChops.difference(image, bg)  # Substract background color from image
        bbox = diff.getbbox()  # Try to find the real bounding box of the image.
    else:
        raise NotImplementedError("Sorry, this function is not implemented yet for images in mode '%s'." % image.mode)

    if bbox:
        image = image.crop(bbox)

    return image


Demo()
