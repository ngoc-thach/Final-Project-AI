from tkinter import *
from tkinter import filedialog
import cv2
from PIL import ImageTk, Image
from tkinter import Tk, Label, Button, filedialog
from keras.models import load_model
from keras.utils import load_img, img_to_array
import numpy as np

# Load model từ drive
model_1 = load_model('khuonmat.h5')
class_names = ['Lionel Messi', 'Truong Giang']
class_age=['36','40']
class_job=['Footballer','Artist']
class_nn=['Argentina','Vietnam']
# Build GUI
th = Tk()
th.title("Face Recognition")
th.geometry('450x450')

lbl = Label(th, text="Final Project AI", fg='red', font=("Arial", 25))
lbl.place(x=120, y=0)
lbl1 = Label(th, text="SVTH: ", fg='black')
lbl1.place(x=40, y=50)
lbl3 = Label(th, text="Dinh Tran Ngoc Thach", fg='black')
lbl3.place(x=120, y=50)
lbl2 = Label(th, text="MSSV:", fg='black')
lbl2.place(x=40, y=75)
lbl4 = Label(th, text="20146529", fg='black')
lbl4.place(x=150, y=75)
lbl5 = Label(th, text="Instructors: ", fg='black')
lbl5.place(x=40, y=100)
lbl6 = Label(th, text="PGT.TS Nguyen Truong Thinh", fg='black')
lbl6.place(x=120, y=100)

# Chức năng chọn ảnh
def select_image():
    # Chọn file ảnh từ máy tính
    file_path = filedialog.askopenfilename()
    
    # Load ảnh
    img = load_img(file_path, target_size=(300, 300))
    img = img_to_array(img)
    img = img.astype('float32')
    img = img / 255
    img = np.expand_dims(img, axis=0)

    # Dự đoán kết quả
    result = model_1.predict(img).argmax()
    person_name = class_names[result]
    person_job=class_job[result]
    person_nn=class_nn[result]
    person_age=class_age[result]
    # Thay đổi ảnh hiển thị
    image = Image.open(file_path)
    image = image.resize((200, 200), Image.ANTIALIAS)
    image_tk = ImageTk.PhotoImage(image)
    label_img.config(image=image_tk)
    label_img.image = image_tk
    
    # Cập nhật tên người
    label_name.config(text=person_name)
    label_name1.config(text='Name:')
    label_age.config(text=person_age)
    label_age1.config(text='Age:')
    label_job.config(text=person_job)
    label_job1.config(text='Job:')
    label_nn.config(text=person_nn)
    label_nn1.config(text='Nationality:')
# Nút "Chọn ảnh"
button_select = Button(th, text="Select Picture", command=select_image)
button_select.place(x=90, y=175)

# Hiển thị ảnh mặc định
default_image_path = 'GetArticleImage.jpg'
default_image = Image.open(default_image_path)
default_image = default_image.resize((200, 200), Image.ANTIALIAS)
default_image_tk = ImageTk.PhotoImage(default_image)
label_img = Label(th, image=default_image_tk)
label_img.place(x=25, y=200)

# Hiển thị tên người
label_name = Label(th, text="", font=("Arial", 12))
label_name.place(x=310, y=225)
label_name1 = Label(th, text="", font=("Arial", 12))
label_name1.place(x=250, y=225)
label_age = Label(th, text="", font=("Arial", 12))
label_age.place(x=350, y=250)
label_age1 = Label(th, text="", font=("Arial", 12))
label_age1.place(x=250, y=250)
label_job = Label(th, text="", font=("Arial", 12))
label_job.place(x=330, y=275)
label_job1 = Label(th, text="", font=("Arial", 12))
label_job1.place(x=250, y=275)
label_nn = Label(th, text="", font=("Arial", 12))
label_nn.place(x=350, y=300)
label_nn1 = Label(th, text="", font=("Arial", 12))
label_nn1.place(x=250, y=300)
th.mainloop()
