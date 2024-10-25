import pandas as pd
import random
import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
def load_random_images_from_folder(folder, num_images=1000):
        images = []
        labels = []
        filenames = os.listdir(folder)
        random.shuffle(filenames)
        selected_filenames = filenames[:num_images]
        for filename in selected_filenames:
            filepath = os.path.join(folder, filename)
            if os.path.isfile(filepath):
                img = cv2.imread(filepath)
                if img is not None:
                    img = cv2.resize(img, (64, 64))
                    images.append(img)
                    label = 1 if 'dog' in filename else 0
                    labels.append(label)
        return np.array(images), np.array(labels)

train_path = "C:/Users/Nvidia/Downloads/dogs-vs-cats/train/train"
test_path = "C:/Users/Nvidia/Downloads/dogs-vs-cats/test1/test1"
train_images, train_labels = load_random_images_from_folder(train_path)
test_images, test_labels = load_random_images_from_folder(test_path)
images = np.concatenate((train_images, test_images), axis=0)
labels = np.concatenate((train_labels, test_labels), axis=0)

images = images / 255.0

n_samples, height, width, depth = images.shape
X = images.reshape((n_samples, height * width * depth))
y = labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

