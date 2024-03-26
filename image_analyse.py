import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageFilter
import cv2
import numpy as np




# Вывод изображения в окно приложения
def show_img(image, root):
    global img_label
    pil_img = Image.fromarray(image)
    img_tk = ImageTk.PhotoImage(image=pil_img)
    if 'img_label' in globals():
        img_label.config(image=img_tk)
    else:
        img_label = tk.Label(root, image=img_tk, anchor='se')
        img_label.pack()
    img_label.image = img_tk
    return img_label


# Изменение размера изображение под размер окна
def resize_img(image, max_height, max_width):
    global win
    height, width = image.shape[:2]
    width_ratio = max_width / width
    height_ratio = max_height / height
    ratio = min(height_ratio, width_ratio)
    new_height = int(height * ratio)
    new_width = int(width * ratio)
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)


# добавление кнопок после открытия изображения
def buttons(button_frame):
    global kernel_size_entry, threshold_entry_1, threshold_entry_2, cont_button, kernel_size_entry_m, q_entry, edge_button

    # Кнопка для двустороннего фильтра
    bilateral_blur_button = tk.Button(button_frame, command=bilat_blur, text='Двусторонний фильтр')
    bilateral_blur_button.grid(row=0, column=0, padx=5, pady=5)

    # Кнопки для задания размера ядра
    kernel_label = tk.Label(button_frame, text='Ядро для медианного фильтра')
    kernel_label.grid(row=1, column=0, padx=5, pady=5)
    kernel_size_entry = tk.Entry(button_frame)
    kernel_size_entry.grid(row=2, column=0, padx=5, pady=5)

    # Кнопка для медианного фильтра
    med_blur_button = tk.Button(button_frame, command=med_blur, text='Медианный фильтр')
    med_blur_button.grid(row=3, column=0, padx=5, pady=5)

    # Кнопки для бинаризации 1
    threshold_label = tk.Label(button_frame, text='Нижний порог (от 0 до 255)')
    threshold_label.grid(row=4, column=0, padx=5, pady=5)
    threshold_entry_1 = tk.Entry(button_frame)
    threshold_entry_1.grid(row=5, column=0, padx=5, pady=5)

    threshold_label = tk.Label(button_frame, text='Верхний порог (от 0 до 255)')
    threshold_label.grid(row=6, column=0, padx=5, pady=5)
    threshold_entry_2 = tk.Entry(button_frame)
    threshold_entry_2.grid(row=7, column=0, padx=5, pady=5)

    binar_button = tk.Button(button_frame, command=binar_thr, text='Бинаризация по порогам')
    binar_button.grid(row=8, column=0, padx=5, pady=5)

    binar_button_f = tk.Button(button_frame, command=adaptiv_bin, text='Бинаризация по формуле')
    binar_button_f.grid(row=9, column=0, padx=5, pady=5)


    # Кнопки для задания размера ядра
    kernel_label = tk.Label(button_frame, text='Ядро для морфологических операций')
    kernel_label.grid(row=10, column=0, padx=5, pady=5)
    kernel_size_entry_m = tk.Entry(button_frame)
    kernel_size_entry_m.grid(row=11, column=0, padx=5, pady=5)

    # Кнопка для закрытия
    closing_button = tk.Button(button_frame, text='Закрытие', command=closing)
    closing_button.grid(row=12, column=0, padx=5, pady=5)

    # Кнопка для дилатации
    delation_button = tk.Button(button_frame, text='Дилатация', command=delation)
    delation_button.grid(row=13, column=0, padx=5, pady=5)

    # Кнопка для открытия
    opening_button = tk.Button(button_frame, text='Открытие', command=opening)
    opening_button.grid(row=14, column=0, padx=5, pady=5)

    # Кнопка для эрозии
    erode_button = tk.Button(button_frame, text='Эрозия', command=erode)
    erode_button.grid(row=15, column=0, padx=5, pady=5)

    # Кнопка для отображения края
    edge_button = tk.Button(button_frame, text='Выделить объекты', command=distance_edge, state=tk.DISABLED)
    edge_button.grid(row=16, column=0, padx=5, pady=5)

    # Кваниль для площади и отображения
    q_label = tk.Label(button_frame, text='Доля наименьших площадей')
    q_label.grid(row=17, column=0, padx=5, pady=5)
    q_entry = tk.Entry(button_frame)
    q_entry.grid(row=18, column=0, padx=5, pady=5)


    # Кнопка для сегментации
    cont_button = tk.Button(button_frame, text='Сегментация', command=cont, state=tk.DISABLED)
    cont_button.grid(row=19, column=0, padx=5, pady=5)



# Функция для отмены последнего изменения
def prev_img_show():
    global win, img, img_label, kernel_size_entry, prev_img
    img = prev_img
    show_img(img, win)

# Функция для медианного фильтра
def med_blur():
    global win, img, img_label, format_flg, left_buttons, kernel_size_entry, prev_img
    prev_img = img.copy()
    kernel_size = kernel_size_entry.get()
    if kernel_size == "":
        kernel = 3
    else:
        kernel = int(kernel_size)
    if kernel % 2 == 0:
        kernel = kernel_size + 1
    img = cv2.medianBlur(img, kernel)
    show_img(img, win)

# Закрытие
def closing():
    global win, img, img_label, format_flg, left_buttons, kernel_size_entry, prev_img
    prev_img = img.copy()
    kernel_size = kernel_size_entry_m.get()
    if kernel_size == "":
        kernel = 3
    else:
        kernel = int(kernel_size)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel=np.ones((kernel, kernel), np.uint8), iterations=1)
    show_img(img, win)

# Дилатация
def delation():
    global win, img, img_label, format_flg, left_buttons, kernel_size_entry_m, prev_img
    prev_img = img.copy()
    kernel_size = kernel_size_entry_m.get()
    if kernel_size == "":
        kernel = 4
    else:
        kernel = int(kernel_size)
    img = cv2.dilate(img, kernel=np.ones((kernel, kernel), np.uint8), iterations=1)
    show_img(img, win)

# Открытие
def opening():
    global win, img, img_label, format_flg, left_buttons, kernel_size_entry_m, prev_img
    prev_img = img.copy()
    kernel_size = kernel_size_entry_m.get()
    if kernel_size == "":
        kernel = 4
    else:
        kernel = int(kernel_size)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, np.ones((kernel, kernel), np.uint8), iterations=1)
    show_img(img, win)

# Закрытие
def erode():
    global win, img, img_label, format_flg, left_buttons, kernel_size_entry_m, prev_img
    prev_img = img.copy()
    kernel_size = kernel_size_entry_m.get()
    if kernel_size == "":
        kernel = 3
    else:
        kernel = int(kernel_size)
    img = cv2.erode(img, np.ones((kernel, kernel), np.uint8), iterations=1)
    show_img(img, win)

# Функция для сегментации
def cont():
    global fst_img, win, img, img_label, format_flg, left_buttons, prev_img, num_plains, was_unknown, q_entry
    was_unknown = False
    prev_img = img.copy()
    q_num = q_entry.get()
    if q_num == "":
        q = 0.1
    else:
        q = float(q_num)

    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(cnt) for cnt in contours]
    quantile_value = np.quantile(areas, q)
    filtered_contours = [cnt for cnt, area in zip(contours, areas) if area >= quantile_value]
    cv2.drawContours(fst_img, filtered_contours, -1, (255, 0, 255), 2)
    num_plains.config(text=f'Обнаружено {len(filtered_contours)} самолетов')
    img = fst_img
    show_img(img, win)


# Функция для бинаризации
def binar_thr():
    global win, img, img_label, format_flg, left_buttons, threshold_entry, prev_img, cont_button, edge_button
    prev_img = img.copy()
    threshold1 = threshold_entry_1.get()
    threshold2 = threshold_entry_2.get()
    if threshold1 == "":
        th0 = 80
    else:
        th0 = int(threshold1)
    if threshold2 == "":
        th1 = 225
    else:
        th1 = int(threshold2)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_min = np.array((th0, 0, 0), np.uint8)
    h_max = np.array((th1, 255, 255), np.uint8)
    cur_image = cv2.inRange(hsv, h_min, h_max)

    _, img = cv2.threshold(cur_image, 0, 255, cv2.THRESH_BINARY_INV)
    cont_button.config(state=tk.NORMAL)
    edge_button.config(state=tk.NORMAL)
    show_img(img, win)

def adaptiv_bin():
    global win, img, img_label, format_flg, left_buttons, threshold_entry, prev_img, cont_button, edge_button
    segmented_image = img.copy()
    b = img[:, :, 0]
    g = img[:, :, 1]
    r = img[:, :, 2]
    br = 0.299 * r + 0.587 * g + 0.114 * b
    mask = (br > 50) & (r < b) | (r + 5 < b) & (b < g)
    segmented_image[mask] = [0, 0, 0]  # Черный цвет (фон)

    gray = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)
    threshold = np.percentile(gray, 90)
    _, img = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    cont_button.config(state=tk.NORMAL)
    edge_button.config(state=tk.NORMAL)
    show_img(img, win)

def distance_edge():
    global win, img, img_label, format_flg, left_buttons, threshold_entry, prev_img, cont_button, unknown, was_unknown
    dist_transform = cv2.distanceTransform(img, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.1 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(img, sure_fg)
    was_unknown = True
    show_img(unknown, win)
# Функция для  фильтра
def bilat_blur():
    global win, img, img_label, format_flg, left_buttons, kernel_size_entry, prev_img
    prev_img = img.copy()
    img = cv2.bilateralFilter(img, d=4, sigmaColor=200, sigmaSpace=300)
    show_img(img, win)


# Функция для открытия изображения в rgb
def open_img():
    global fst_img, win, img, img_label, format_flg, left_buttons, prev_img, prev_img_button, save_button
    try:
        img_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    except:
        label = tk.Label(win)
        label.config(text='!Не удалось открыть файл или такого файла не существует')
        raise
    format_flg = 1
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = resize_img(img, win.winfo_height() - 50, win.winfo_width() - 200)
    fst_img = img.copy()
    img_label = show_img(img, win)
    prev_img_button.config(state=tk.NORMAL)
    save_button.config(state=tk.NORMAL)
    buttons(left_buttons)
def save_image():
    global img, unknown
    save_path = filedialog.asksaveasfilename(defaultextension=".pdf")
    if was_unknown:
        pil_img = Image.fromarray(unknown)
    else:
        pil_img = Image.fromarray(img)
    pil_img.save(save_path)

win_height = 1250
win_width = 600

# Создания диалогового окна
win = tk.Tk()
win.title('Лабораторная работа 1. Смирнова Виктория')
win.geometry(f'{win_height}x{win_width}+100+100')

# Создание фрейма кнопок открытия изображений
top_buttons = tk.Frame(win, pady=30)
top_buttons.pack(side=tk.BOTTOM)

# Создание фрейма кнопок обработки изображений
left_buttons = tk.Frame(win)
left_buttons.pack(side=tk.LEFT)

# Кнопка для открытия изображений
open_img_cbtn = tk.Button(top_buttons, text='Загрузить изображение', command=open_img)
open_img_cbtn.grid(row=0, column=0, padx=10)

save_button = tk.Button(top_buttons, text="Сохранить изображение(.pdf)", command=save_image, state=tk.DISABLED)
save_button.grid(row=0, column=1,padx=10)

prev_img_button = tk.Button(top_buttons, text='Отменить последнее изменение', command=prev_img_show, state=tk.DISABLED)
prev_img_button.grid(row=0, column=2, padx=5, pady=5)
was_unknown = False
# Отображение кол-ва самолетов после сегментации
num_plains = tk.Label(top_buttons)
num_plains.grid(row=1, column=1, padx=10, pady=10)
win.mainloop()