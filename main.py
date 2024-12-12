import os
import numpy as np
from PIL import Image
import cv2
import torch
import torch.nn as nn

# Загрузка каскадного классификатора для обнаружения номерных знаков
plate_cascade = cv2.CascadeClassifier('russian_plate.xml')


def GetnExpID():
    with open("id.txt", "r") as file:
        id = file.read()
    id = str(int(id) + 1)
    with open("id.txt", "w") as file:
        file.write(id)
    return id


id = GetnExpID()
print(f"Current ID: {id}")

# Загрузка изображения из файла для тестирования
image_path = '1.jpg'  # Замените на путь к вашему изображению
image = cv2.imread(image_path)
cv2.imshow('1', image)
cv2.waitKey(0)
def detect_plate(img):
    plate_img = img.copy()
    plate_rects = plate_cascade.detectMultiScale(plate_img, scaleFactor=1.2, minNeighbors=7)

    if len(plate_rects) == 0:
        return plate_img, None  # Если номерные знаки не найдены

    # Находим самый большой прямоугольник (номерной знак)
    largest_plate = max(plate_rects, key=lambda r: r[2] * r[3])
    (x, y, w, h) = largest_plate

    # Ограничиваем координаты рамки в пределах изображения
    x = max(x, 0)
    y = max(y, 0)
    w = min(w, img.shape[1] - x)
    h = min(h, img.shape[0] - y)

    # Извлечение области интереса (номерного знака)
    plate = img[y:y + h, x:x + w]

    # Нормализация размера до 333x75 перед сохранением
    plate_normalized = cv2.resize(plate, (333, 75))

    # Сохранение нормализованного изображения номерного знака
    cv2.imwrite(f'plates/readyplates/contour{id}.png', plate_normalized)

    # Рисуем рамку вокруг номерного знака
    cv2.rectangle(plate_img, (x + 2, y), (x + w - 3, y + h - 5), (51, 181, 155), 3)

    return plate_img, plate_normalized


def find_contours(dimensions, img):
    # Находим контуры в изображении
    contours, _ = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    valid_contours = []

    for contour in contours:
        intX, intY, intWidth, intHeight = cv2.boundingRect(contour)
        if (dimensions[0] < intWidth < dimensions[1]) and (dimensions[2] < intHeight < dimensions[3]):
            valid_contours.append((intX, intY, intWidth, intHeight))

    char_images = []

    for (intX, intY, intWidth, intHeight) in valid_contours:
        char_image = img[intY:intY + intHeight, intX:intX + intWidth]
        char_image_resized = cv2.resize(char_image, (20, 40))
        char_images.append(char_image_resized)

    return char_images


def segment_characters(image):
    img_lp = cv2.resize(image, (333, 75))
    img_gray_lp = cv2.cvtColor(img_lp, cv2.COLOR_BGR2GRAY)

    img_binary_lp = cv2.adaptiveThreshold(img_gray_lp, 255,
                                          cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY_INV,
                                          11,
                                          2)

    kernel = np.ones((2, 2), np.uint8)
    img_binary_lp = cv2.erode(img_binary_lp, kernel)
    img_binary_lp = cv2.dilate(img_binary_lp, kernel)

    dimensions = [img_binary_lp.shape[0] / 6,
                  img_binary_lp.shape[0] / 2,
                  img_binary_lp.shape[1] / 10,
                  2 * img_binary_lp.shape[1] / 3]

    return find_contours(dimensions, img_binary_lp)


# Обработка изображения
output_img_with_plate_detection, normalized_plate_image = detect_plate(image)

if normalized_plate_image is not None:
    char_images = segment_characters(normalized_plate_image)

# Загрузка изображения из файла для тестирования
# Замените 'path_to_your_image.png' на путь к вашему изображению
# image_path = 'plates/readyplates/contour80.png'
image_path = '3.jpg'
image = Image.open(image_path)



# Преобразование изображения PIL в массив NumPy
image_np = np.array(image)

# Убедитесь, что изображение в правильном формате (BGR)
if image_np.ndim == 3 and image_np.shape[2] == 4:  # Если RGBA
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2BGR)
elif image_np.ndim == 3 and image_np.shape[2] == 3:  # Если RGB
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

# Определение размеров изображения и фрагментов
image_width, image_height = image.size
fragment_width = 35
fragment_height = 75
overlap = 15  # Сдвиг вправо

# Папка для сохранения фрагментов
output_dir = 'plates/cracked'
os.makedirs(output_dir, exist_ok=True)

fragments = []
for x in range(0, image_width - fragment_width + 1, overlap):
    for y in range(0, image_height - fragment_height + 1):
        box = (x, y, x + fragment_width, y + fragment_height)
        fragment = image.crop(box)

        # Обрезаем по 2 пикселя сверху и снизу
        fragment = fragment.crop((0, 2, fragment_width, fragment_height - 2))

        # Нормализуем размер до 28x28 пикселей
        fragment = fragment.resize((28, 28))

        fragments.append(fragment)

        # Сохранение фрагмента с уникальным именем
        fragment.save(os.path.join(output_dir, f'fragment_{len(fragments)}.png'))

print(f'Сохранено {len(fragments)} фрагментов в папку {output_dir}.')

# # Выводим результаты
# plt.figure(figsize=(10, 6))
# plt.imshow(cv2.cvtColor(output_img_with_plate_detection, cv2.COLOR_BGR2RGB))
# plt.axis('off')
# plt.title('Распознанный номер')
# plt.show()

cv2.imshow("2",output_img_with_plate_detection)
cv2.waitKey(0)
class ConvNet(nn.Module):  # Класс модели
    def __init__(self):  # Конструктор
        super(ConvNet,self).__init__()
        self.act = nn.LeakyReLU()
        self.maxpool = nn.MaxPool2d(2, 2)

        # Свёрточные слои
        self.conv0 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=0)  # Выход: [32, 26, 26]
        self.conv1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0) # Выход: [32, 24, 24]
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0) # Выход: [32, 22, 22]

        # После двух операций MaxPool размер уменьшается:
        # После первого MaxPool: [32, 13, 13]
        # После второго MaxPool: [32, 6, 6]

        # Адаптивное усреднение для приведения к размеру [1, 1]
        self.adaptive = nn.AdaptiveAvgPool2d((1, 1))  # Приводит к размеру [batch_size, channels, 1, 1]

        # Полносвязные слои
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(32 * 1 * 1, 10)  # Вход: [32 * 1 * 1], Выход: [10]
        self.linear2 = nn.Linear(10, 23)            # Вход: [10], Выход: [23]

    def forward(self, x):
        out = self.conv0(x)
        out = self.act(out)
        out = self.maxpool(out)   # Выход: [32, 13, 13]

        out = self.conv1(out)
        out = self.act(out)
        out = self.maxpool(out)   # Выход: [32, 6, 6]

        out = self.conv2(out)
        out = self.act(out)

        out = self.adaptive(out)   # Приводим к размеру [batch_size, channels=32, height=1, width=1]

        out = self.flatten(out)     # Преобразуем в вектор
        out = self.linear1(out)     # Первый полносвязный слой
        out = self.act(out)
        out = self.linear2(out)     # Второй полносвязный слой

        return out


# Инициализируем модель
model = ConvNet()
model.load_state_dict(torch.load('model_letters.pth', weights_only=True))


# ______________________________
# Укажите путь к директории с изображениями
directory_path = 'craсked/'

# Пример списка классов
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
               'A', 'B', 'C', 'E', 'H', 'K', 'M', ' ', 'P', 'T',
               'X', 'Y', 'false']  # Замените на свои классы

# Список для хранения предсказанных классов
predicted_sequence = []

# Получаем список изображений и сортируем их по числовому значению
image_files = [f for f in os.listdir(directory_path) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
image_files.sort(key=lambda x: int(os.path.splitext(x)[0]))  # Сортируем по номеру файла

# Проход по всем изображениям в директории
for img_name in image_files:
    image_path = os.path.join(directory_path, img_name)

    # Загрузка изображения
    image = cv2.imread(image_path)
    if image is None:
        print(f"Image at {image_path} could not be loaded.")
        continue

    # Преобразование цвета из BGR в RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Изменение размера изображения до 28x28
    image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_AREA)

    # Нормализация значений пикселей
    image_tensor = torch.tensor(image).permute(2, 0, 1).float() / 255.0  # Теперь размерность (C, H, W)

    # Добавление размерности батча
    image_tensor = image_tensor.unsqueeze(0)  # Теперь размерность (1, C, H, W)

    # Применение модели (если требуется предварительная обработка)
    with torch.no_grad():  # Отключаем градиенты для оценки
        pred = model(image_tensor)

    # Получение предсказанных классов
    predicted_classes = torch.argmax(pred, dim=1)

    # Получение названий предсказанных классов
    predicted_labels = [class_names[i] for i in predicted_classes]

    # Добавляем предсказанные символы в последовательность
    predicted_sequence.append(predicted_labels[0])  # Берем только первый элемент для каждого изображения

# Формируем итоговую строку из предсказанных символов
result_string = ''.join(predicted_sequence)
print("Предсказанная последовательность символов:", result_string)