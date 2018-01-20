from mnist import MNIST
import numpy as np
import matplotlib.pyplot as plt

def showNum(image):
    for i in range(0, 28):
        line = image[28 * i: 28 * (i + 1)]
        strLine = ""
        for j in line:
            if j < 10:
                strLine += "."
            elif j < 100:
                strLine += "x"
            else:
                strLine += "#"
        print(strLine)

def showPlot(image, label):
    image = np.array(image, dtype='uint8')
    image = image.reshape(28,28)
    plt.title('Label is {label}'.format(label=label))
    plt.imshow(image, cmap='Blues')
    plt.show()

mndata = MNIST('samples')
images, labels = mndata.load_training()
# or
#images, labels = mndata.load_testing()
for i in range(0,1):
    showNum(images[i])
    print(labels[i])
showPlot(images[3], labels[3])
