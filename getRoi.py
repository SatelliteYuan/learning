import tkinter as tk
from PIL import Image, ImageTk
import cv2

root = tk.Tk()

root = tk.Tk()
root.title('应用程序窗口')        #窗口标题
root.resizable(False, False)    #固定窗口大小
windowWidth = 800               #获得当前窗口宽
windowHeight = 500              #获得当前窗口高
screenWidth,screenHeight = root.maxsize()     #获得屏幕宽和高
geometryParam = '%dx%d+%d+%d'%(windowWidth, windowHeight, (screenWidth-windowWidth)/2, (screenHeight - windowHeight)/2)
root.geometry(geometryParam)    #设置窗口大小及偏移坐标
root.wm_attributes('-topmost',1)#窗口置顶

#label文本
label_text = tk.Label(root, text = '文本');
label_text.pack();
 
#label图片
img_gif = tk.PhotoImage(file = '1.jpg')
label_img = tk.Label(root, image = img_gif)
label_img.pack()

 
#不带图button
button = tk.Button(root, text = '不带图按钮')
button.pack()
 
#带图button，image
# button_img_gif = tk.PhotoImage(file = 'button_gif.gif')
# button_img = tk.Button(root, image = button_img_gif, text = '带图按钮')
# button_img.pack()
 
#带图button，bitmap
button_bitmap = tk.Button(root, bitmap = 'error', text = '带图按钮')
button_bitmap.pack()
 
root.mainloop()


# img_open = Image.open('./frames0.tif')
# img_png = ImageTk.PhotoImage(img_open)
# label_img = tk.Label(root, image = img_png)
# label_img.pack()


# def motion(event):
#     x, y = event.x, event.y
#     print('{}, {}'.format(x, y))

# root.bind('<Motion>', motion)
root.mainloop()