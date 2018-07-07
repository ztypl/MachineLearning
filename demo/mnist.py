# coding : utf-8
# create by ztypl on 2018/6/19

import io
import tkinter
import numpy as np
from PIL import Image, ImageChops

class Demo:
    def __init__(self):
        self.window = tkinter.Tk()
        self.cv = tkinter.Canvas(self.window, bg='white')
        self.cv.pack()
        self.bt_clear = tkinter.Button(self.window, text="Clear", command=self.clear_canvas)
        self.bt_clear.pack()

        self.to_post = tkinter.Button(self.window, text="Post", command=self.post)
        self.to_post.pack()

        self.width = 6
        self.old_x = 0
        self.old_y = 0

        self.cv.bind("<Button-1>", self.mouse_down_event)
        self.cv.bind("<B1-Motion>", self.mouse_move_event)
        self.cv.focus_set()

        self.window.mainloop()

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

    def post(self):
        ps = self.cv.postscript()
        im = Image.open(io.BytesIO(ps.encode('utf-8')))
        im.convert('L')
        x = max(im.size)
        im = im.resize((x, x))
        im.thumbnail((28, 28))

        im2 = Image.new('L', (28, 28), color='white')
        im2.paste(im)

        data = np.asarray(im2)
        # data = (255 - np.asarray(im2).astype('float64'))//16
        # data = data.reshape((1, data.size))
        print(data)

        #print(im2.size)
        # im2.show()



Demo()