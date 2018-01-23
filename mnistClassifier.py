from mnist import MNIST
import numpy as np
# import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier, DistanceMetric
from sklearn.metrics import classification_report

def showPlot(image, label):
    image = np.array(image, dtype='uint8')
    image = image.reshape(28, 28)
    # plt.title('Label is {label}'.format(label=label))
    # plt.imshow(image, cmap='Blues')
    # plt.show()

mndata = MNIST('samples')

images, labels = mndata.load_training()
imagesTest, labelsTest = mndata.load_testing()


accuracies = []
print("Fitting models")
for k in range(1, 2):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(np.array(images[:6000]), np.array(labels[:6000]))
    # score = model.score(np.array(images[20000:22000]), np.array(labels[20000:22000]))
    # print("k=%d, accuracy=%.2f%%" % (k, score * 100))
    # accuracies.append(score)


# def ques1c():
    # neighDist, neighIndex = model.kneighbors(np.array(imagesTest[0:10]), n_neighbors=5)
    #
    # fig, ax = plt.subplots(nrows=10, ncols=6, squeeze=True)
    #
    # j = 0
    # for row in ax:
    #     i = 0
    #     for col in row:
    #         if i == 0:
    #             image = np.array(imagesTest[j], dtype='uint8')
    #         else:
    #             image = np.array(images[neighIndex[j][i-1]], dtype='uint8')
    #         image = image.reshape(28, 28)
    #         col.imshow(image, cmap='Blues')
    #         i += 1
    #     j += 1
    #
    # plt.show()
    #
    # print(neighIndex)
    # print("Trained")

predictions = model.predict(np.array(imagesTest[:100]))
print("EVALUATION ON TESTING DATA")
print(classification_report(np.array(labelsTest[:100]), predictions))