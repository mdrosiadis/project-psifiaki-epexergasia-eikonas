import matplotlib.pyplot as plt
from skimage.feature import hog
from torchvision import datasets
from torchvision.transforms import ToTensor
from sklearn import svm
from joblib import dump, load
import os

from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from utilities import create_confusion_matrix

DATASET_FILENAME = 'data/hog_dataset.gz'
MODEL_FILENAME = 'model_hog.data'

CELL_PIXELS = 8
CELLS_PER_BLOCK = 2

if not os.path.isfile(DATASET_FILENAME):
    print('Computing HOGs for MNIST dataset')
    # create the dataset
    # download MNIST dataset
    train_data = datasets.MNIST(
        root = 'data',
        train = True,
        transform = ToTensor(),
        download = True,
    )
    test_data = datasets.MNIST(
        root = 'data',
        train = False,
        transform = ToTensor()
    )

    # transform images to HOGs
    train_hogs = [
        hog(image, orientations=9, pixels_per_cell=(CELL_PIXELS, CELL_PIXELS), cells_per_block=(CELLS_PER_BLOCK, CELLS_PER_BLOCK))
        for image in train_data.data
    ]
    train_targets = train_data.targets

    test_hogs = [
        hog(image, orientations=9, pixels_per_cell=(CELL_PIXELS, CELL_PIXELS), cells_per_block=(CELLS_PER_BLOCK, CELLS_PER_BLOCK))
        for image in test_data.data
    ]
    test_targets = test_data.targets

    hog_dataset = (train_hogs, train_data.targets, test_hogs, test_data.targets)
    # save dataset to file
    dump(hog_dataset, DATASET_FILENAME)

else:
    # load dataset 
    print('Loading cached HOGs')
    train_hogs, train_targets, test_hogs, test_targets = load(DATASET_FILENAME)


if not os.path.isfile(MODEL_FILENAME):
    # create the SVM
    print("Fitting SVM")
    clf = svm.SVC(decision_function_shape='ovo') # one versus one (ovo)
    # fit to train data
    clf.fit(train_hogs, train_targets)
    # save trained model to file
    print('Saving fitted SVM')
    dump(clf, MODEL_FILENAME)

else:
    print("Using pretrained SVM model")
    # load the model
    clf = load(MODEL_FILENAME)


# test against testing data
print("Testing model against test data")
test_predictions = clf.predict(test_hogs)
comp = [tp == tt for tp, tt in zip(test_predictions, test_targets)]

correct = sum(comp).item()
total = len(comp)
accuracy = correct / total
print(f'Correct: {correct} / {total} (accuracy: {accuracy})')

cm = create_confusion_matrix(test_targets, test_predictions)
print('Confusion matrix')
print(cm)
# plot confusion matrix

ConfusionMatrixDisplay.from_predictions(test_targets, test_predictions)
plt.title(f'Confusion matrix HOG (cell pixels: {CELL_PIXELS}, cells per block: {CELLS_PER_BLOCK})')
plt.savefig(f'plots/confusion_hog_{CELL_PIXELS}_{CELLS_PER_BLOCK}.png')

# plt.show()

