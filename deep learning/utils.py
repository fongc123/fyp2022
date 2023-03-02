"""
File: functions.py

This file contains utility functions for the age classification of dried orange peels.
"""
from orange_peels import Constants as c
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, random_split
from sklearn import metrics
import cv2
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import json

def get_device():
    """
    Get the device to use for training.

    Returns:
    - device: a string containing the device to use
    """
    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"
    device = torch.device(device)
    print(f"Device: {device}")

    return device

def view_image(image, label, decoder):
    """
    Show a Tensor with `matplotlib.pyplot.imshow`.

    Parameters:
    - image: a Tensor of shape (3, H, W)
    - label: a Tensor of shape (1)
    - decoder: a dictionary for decoding the label
    """
    image = image.numpy().astype(np.int32).transpose(1, 2, 0)
    label = decoder[label]
    plt.imshow(image)
    plt.title(f"Label: {label}")
    plt.show()

def create_loader(data, prop=None, train_idx=None, test_idx=None):
    """
    Create train and test data loaders.

    Parameters:
    - prop: proportion of data to use for training
    - train_idx: indices of data to use for training
    - test_idx: indices of data to use for testing
    """
    if prop is not None:
        size = int(prop*len(data))
        train, test = random_split(data, [size, len(data)-size])
    elif train_idx is not None and test_idx is not None:
        train = Subset(data, train_idx)
        test = Subset(data, test_idx)
    else:
        raise ValueError("Invalid arguments.")

    return DataLoader(train, batch_size=c.BATCH_SIZE, shuffle=True), DataLoader(test, batch_size=c.BATCH_SIZE, shuffle=True)

def create_model(base, optimizer, num_classes=4, output_layers=2048, remove_layers=1, device=None, learning_rate=c.LR):
    """
    Create a model from a base model.

    Parameters:
    - base: starting model (e.g., ResNet50)
    - optimizer: optimizer to use (e.g., Adam)
    - num_classes: number of classes to classify
    - device: device to use for training
    - learning_rate: learning rate to use for training
    """
    feature_extractor = nn.Sequential(*list(deepcopy(base).children())[:-remove_layers]) # remove the last layer
    for param in feature_extractor.parameters():
        param.requires_grad = False # freeze the parameters

    model = nn.Sequential(
        feature_extractor,
        nn.Flatten(),
        nn.Linear(output_layers, num_classes) # new classifier
    )

    if device is not None:
        model = model.to(device)
    
    optimizer = optimizer(model.parameters(), lr=learning_rate)

    return model, optimizer

def create_model_inception(base, optimizer, num_classes=4, output_layers=2048, device=None, learning_rate=c.LR):
    model = deepcopy(base)
    for param in model.parameters():
        param.requires_grad = False
    
    model.aux_logits = False
    model.fc = nn.Linear(output_layers, num_classes)

    if device is not None:
        model = model.to(device)
    
    optimizer = optimizer(model.parameters(), lr=learning_rate)

    return model, optimizer

def train_model(model, train_loader, test_loader, optimizer, criterion, device=None, epochs=c.EPOCHS, stats=False):
    train_loss = []
    train_acc = []
    test_acc = []
    for epoch in range(epochs):
        for images, labels in train_loader:
            model.train()
            if device is not None:
                images = images.to(device)
                labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # training loss and accuracy
            train_loss.append(loss.item())
            train_acc.append((outputs.argmax(1) == labels).float().mean().item()) # running accuracy

            # testing accuracy   
            model.eval()
            predictions, true = test_model(model, test_loader, device)
            test_acc.append((predictions == true).float().mean().item())

        if stats:
            print(f"[E{epoch+1}]\tLoss {train_loss[-1]:.4f}, Train {train_acc[-1]:.4f}, Test {test_acc[-1]:.4f}")
    
    return train_loss, train_acc, test_acc

def test_model(model, loader, device=None, save=False):
    predictions = torch.tensor([], dtype = torch.long).to(device)
    true = torch.tensor([], dtype = torch.long).to(device)

    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            if device is not None:
                images = images.to(device)
                labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            predictions = torch.cat((predictions, predicted))
            true = torch.cat((true, labels))

    if save:
        return predictions.cpu().numpy(), true.cpu().numpy()
    else:
        return predictions, true

def generate_metrics(true, predictions, dl=[6,10,15,20]):
    report = metrics.classification_report(true, predictions, output_dict=True)
    cm = metrics.ConfusionMatrixDisplay.from_predictions(true, predictions, display_labels=dl)

    return cm, report

def plot_loss_acc(axes, train_loss, train_acc, test_acc, label=None, color=None, max_loss=2):
    if label is not None and color is not None:
        axes[0].plot(train_loss, label=label, color=color, alpha=0.75)
        axes[1].plot(train_acc, label=label, color=color, alpha=0.75)
        axes[2].plot(test_acc, label=label, color=color, alpha=0.75)
    else:
        axes[0].plot(train_loss, alpha=0.75)
        axes[1].plot(train_acc, alpha=0.75)
        axes[2].plot(test_acc, alpha=0.75)
    for i in range(3):
        axes[i].set_xlabel("Iterations")
    axes[0].set_ylim([0, max_loss])
    axes[0].set_title("Training Loss")
    axes[0].set_ylabel("Loss")
    axes[1].set_ylim([0, 1])
    axes[1].set_title("Training Accuracy")
    axes[1].set_ylabel("Accuracy")
    axes[2].set_ylim([0, 1])
    axes[2].set_title("Test Accuracy")
    axes[2].set_ylabel("Accuracy")

def results_to_json(filename, train_loss, train_acc, test_acc, true, predictions):
    results = {
        "train_loss": train_loss,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "true": true.tolist(),
        "predictions": predictions.tolist()
    }

    with open(filename, "w") as f:
        json.dump(results, f)

def load_json(filename):
    with open(filename, "r") as f:
        results = json.load(f)

    return results

def find_object_boundary_canny(image, canny_min=25, canny_max=75, thresh_min=0, thresh_val=255, dilate_size=5):
    """
    Find the boundary of an object in an image with the Canny edge algorithm.

    It returns the contour with the maximum area.

    Parameters:
    - image: the image to process (grayscale)
    - canny_min: the minimum hysterisis threshold for the Canny edge algorithm
    - canny_max: the maximum hysterisis threshold for the Canny edge algorithm
    - thresh_min: the thresholding value
    - thresh_val: the value to assign to pixels above the threshold
    - dilate_size: the size of the structuring element for dilation
    """
    canny = cv2.Canny(image, canny_min, canny_max)
    _, thresh = cv2.threshold(canny, thresh_min, thresh_val, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = cv2.dilate(thresh, np.ones((dilate_size, dilate_size), np.uint8))
    cnts, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(cnts, key=cv2.contourArea)

    return cnt

def draw_masked_image(image, cnt, color=(255, 255, 255)):
    """
    Returns the masked image based on a contour.

    Parameters:
    - image: the image to mask (colored)
    - cnt: the contour to use for masking
    """
    mask = np.zeros(image.shape, np.uint8)
    cv2.drawContours(mask, [cnt], 0, color, -1)
    
    return cv2.bitwise_and(image, mask)

def split_image(image, nsplits=2):
    """
    Splits image into nsplits x nsplits subimages.
    """
    h, w = image.shape[:2]
    subimages = []
    for i in range(nsplits):
        for j in range(nsplits):
            subimages.append(image[i*h//nsplits:(i+1)*h//nsplits, j*w//nsplits:(j+1)*w//nsplits])

    return subimages

def filter_subimages(images, thresh=0.5):
    """
    Filters out subimages that are completely black.
    """
    filtered = []
    for image in images:
        if np.sum(image == 0) < np.product(image.shape)*thresh:
            filtered.append(image)

    return filtered