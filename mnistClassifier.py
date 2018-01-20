from mnist import MNIST
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

def showPlot(image, label):
    image = np.array(image, dtype='uint8')
    image = image.reshape(28,28)
    plt.title('Label is {label}'.format(label=label))
    plt.imshow(image, cmap='Blues')
    plt.show()

mndata = MNIST('samples')

images, labels = mndata.load_training()


accuracies = []
print("Fitting models")
for k in range(1,10):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(np.array(images[0:20000]), np.array(labels[0:20000]))
    imagesTest, labelsTest = mndata.load_testing()
    score = model.score(np.array(images[20000:22000]), np.array(labels[20000:22000]))
    print("k=%d, accuracy=%.2f%%" % (k, score * 100))
    accuracies.append(score)
    predictions = model.predict(np.array(imagesTest[:1000]))
    print("EVALUATION ON TESTING DATA")
    print(classification_report(np.array(labelsTest[:1000]), predictions))


