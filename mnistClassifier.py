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
imagesTest, labelsTest = mndata.load_testing()

accuracies = []
print("Fitting models")
for k in range(1, 2):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(np.array(images[:60000]), np.array(labels[:60000]))
    # score = model.score(np.array(images[20000:22000]), np.array(labels[20000:22000]))
    # print("k=%d, accuracy=%.2f%%" % (k, score * 100))
    # accuracies.append(score)



    neighDist, neighIndex = model.kneighbors(np.array(imagesTest[5342:5343]), n_neighbors=4)


    print(neighIndex)
    neighIndex = neighIndex.reshape((1, -1))
    print(neighIndex)
    print("Trained")

    showPlot(imagesTest[5342], labelsTest[5342])
    showPlot(images[neighIndex[0][0]], labels[neighIndex[0][0]])
    showPlot(images[neighIndex[0][1]], labels[neighIndex[0][1]])
    showPlot(images[neighIndex[0][2]], labels[neighIndex[0][2]])
    showPlot(images[neighIndex[0][3]], labels[neighIndex[0][3]])


    # predictions = model.predict(np.array(imagesTest[:10000]))
    # print("EVALUATION ON TESTING DATA")
    # print(classification_report(np.array(labelsTest[:10000]), predictions))



