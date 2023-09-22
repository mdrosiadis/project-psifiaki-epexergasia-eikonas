import os
from lz4.frame import create_compression_context
from matplotlib.cbook import print_cycles
import torch
import torch.nn as nn

from torchvision import datasets
from torchvision.transforms import ToTensor

from torch.utils.data import DataLoader
from torch.autograd import Variable

from torch.optim import SGD

from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from utilities import create_confusion_matrix


MODEL_FILENAME = 'model_cnn.data'
BATCH_SIZE = 100
NUM_EPOCHS = 6

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

loaders = {
    'train' : DataLoader(train_data,
                         batch_size=BATCH_SIZE,
                         shuffle=True,
                         num_workers=1),

    'test'  :DataLoader(test_data,
                        batch_size=BATCH_SIZE,
                        shuffle=True,
                        num_workers=1),
}

# display 1 image from each class
train_targets_list = [t.item() for t in train_data.targets]
fig = plt.figure()
for label in range(10):
    i = train_targets_list.index(label)
    plt.subplot(2, 5, label+1)
    plt.axis('off')
    plt.imshow(train_data.data[i], cmap='gray', vmin=0, vmax=255)
    plt.title(str(label))

plt.gcf().suptitle('Training data sample')
# plt.show()
plt.savefig('plots/dataset_preview.png')

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# network class
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=6,
                kernel_size=3,
                stride=1,
                padding=0,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(6),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=6,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=0,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.full1 = nn.Sequential(
            nn.Linear(16*5*5, 120),
            nn.ReLU()
        )
        self.full2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU()
        )
        self.out = nn.Sequential(
            nn.Linear(84, 10),
            # nn.Softmax(dim=1)
        )


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.full1(x)
        x = self.full2(x)

        return self.out(x)

    def predict_(self, images):
        test_output = model(images)
        pred_y = torch.max(test_output, 1)[1].data.squeeze()
        return pred_y

    def predict_one(self, img):
        return self.predict_(img.resize(1, 1, 28, 28))



def train(num_epochs, model, loaders):
    loss_func = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9)
    model.train()
    # Train the model
    total_step = len(loaders['train'])
    epoch_ticks = [0]
    ac, l = test(model, loss_func=loss_func)
    accuracy_val = [ac]
    loss_val = [l]
    for epoch in range(num_epochs):
        model.train()
        for i, (images, labels) in enumerate(loaders['train']):

            b_x = Variable(images)   # batch x
            b_y = Variable(labels)   # batch y

            output = model(b_x)
            loss = loss_func(output, b_y)

            # clear gradients for this training step   
            optimizer.zero_grad()
            # backpropagation, compute gradients 
            loss.backward()                # apply gradients             
            optimizer.step()
            if (i+1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

        epoch_ticks.append(epoch+1)
        ac, l = test(model, loss_func=loss_func)
        accuracy_val.append(ac)
        loss_val.append(l)

    plt.subplot(2, 1, 1)
    plt.plot(epoch_ticks, loss_val)
    plt.grid()
    plt.xlabel('Epoch')
    plt.ylabel('loss')

    plt.subplot(2, 1, 2)
    plt.plot(epoch_ticks, accuracy_val)
    plt.grid()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    plt.gcf().suptitle(f'CNN training over {num_epochs} epochs')
    plt.savefig(f'plots/cnn_training_{NUM_EPOCHS}.png')
    # plt.show()

# b_y = Variable(lab)   # batch y
def test(model, loss_func=None,print_output=False):
    # Test the model
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        ls = []
        for images, labels in loaders['test']:
            # test_output = model(images)
            # pred_y = torch.max(test_output, 1)[1].data.squeeze()
            pred_y = model.predict_(images)
            correct += (pred_y == labels).sum().item()
            total += labels.size(0)
            if loss_func is not None:
                ls.append(loss_func(model(Variable(images)), Variable(labels)).item())
            # accuracy = (pred_y == labels).sum().item() / float(labels.size(0))
            # print(pred_y, labels, accuracy)

        avg_loss = None
        if loss_func is not None:
            avg_loss = sum(ls) / len(ls)

        acc = correct / total
        if print_output:
            print('Test Accuracy of the model on the 10000 test images: %.2f' % acc)
            print(f'{correct=} {total=} {len(loaders["test"])=}')


    return acc, avg_loss


model = CNN()
if not os.path.isfile(MODEL_FILENAME):
    print('Training model')
    train(NUM_EPOCHS, model, loaders)
    torch.save(model.state_dict(), MODEL_FILENAME)
else:
    print('Using pretrained model')
    model.load_state_dict(torch.load(MODEL_FILENAME))
    model.eval()

print('Trained model')
test(model, print_output=True)

print("Testing model against test data")
model.eval()
with torch.no_grad():
    actual = []
    predictions = []
    for images, labels in loaders['test']:
        # test_output = model(images)
        # pred_y = torch.max(test_output, 1)[1].data.squeeze()
        pred_y = model.predict_(images)
        actual.extend([l.item() for l in labels])
        predictions.extend([p.item() for p in pred_y])


cm = create_confusion_matrix(actual, predictions)
print('Confusion matrix')
print(cm)
# plot confusion matrix
ConfusionMatrixDisplay.from_predictions(actual, predictions)
plt.title('Confusion matrix CNN')
plt.savefig(f'plots/confusion_cnn_{NUM_EPOCHS}.png')
# plt.show()

