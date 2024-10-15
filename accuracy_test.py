import os
import torch
from PIL import Image
import torch.nn as nn  # This was missing
import torch.nn.functional as F  # This was missing
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score
from tqdm import tqdm  # for showing progress bar

# Use the same class definitions and transformation as in your Streamlit app
class_names = ['Benign', 'Early', 'Pre', 'Pro']
class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}

# Transformations (same as during inference)
transform = transforms.Compose([
    transforms.Resize(24),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Load the trained model (same ResNeXt architecture)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Model definition (ResNeXt architecture from your previous code)
class Block(nn.Module):
    expansion = 2

    def __init__(self, in_planes, cardinality=32, bottleneck_width=4, stride=1):
        super(Block, self).__init__()
        group_width = cardinality * bottleneck_width
        self.conv1 = nn.Conv2d(in_planes, group_width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(group_width)
        self.conv2 = nn.Conv2d(group_width, group_width, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(group_width)
        self.conv3 = nn.Conv2d(group_width, self.expansion * group_width, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * group_width)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * group_width:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * group_width, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * group_width)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNeXt(nn.Module):
    def __init__(self, num_blocks, cardinality, bottleneck_width, num_classes=4):
        super(ResNeXt, self).__init__()
        self.cardinality = cardinality
        self.bottleneck_width = bottleneck_width
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(num_blocks[0], 1)
        self.layer2 = self._make_layer(num_blocks[1], 2)
        self.layer3 = self._make_layer(num_blocks[2], 2)
        self.linear = nn.Linear(cardinality * bottleneck_width * 8, num_classes)

    def _make_layer(self, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(Block(self.in_planes, self.cardinality, self.bottleneck_width, stride))
            self.in_planes = Block.expansion * self.cardinality * self.bottleneck_width
        self.bottleneck_width *= 2
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 6)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

# Load the trained model
model = ResNeXt(num_blocks=[3, 3, 3], cardinality=32, bottleneck_width=4)
model.load_state_dict(torch.load('D:\\Leukemia project\\Source_Code\\model.pth', map_location=device)['net'])
model.to(device)
model.eval()

# Function to evaluate accuracy on test dataset
def evaluate_model(test_dir):
    all_preds = []
    all_labels = []

    # Iterate over the classes in the test folder
    for class_name in class_names:
        class_dir = os.path.join(test_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        
        # Get all images in the class directory
        for img_name in tqdm(os.listdir(class_dir), desc=f"Processing {class_name}"):
            img_path = os.path.join(class_dir, img_name)
            if not img_path.endswith(('png', 'jpg', 'jpeg')):
                continue

            # Load and preprocess the image
            image = Image.open(img_path)
            image = transform(image).unsqueeze(0).to(device)

            # Make prediction
            with torch.no_grad():
                outputs = model(image)
                _, predicted = outputs.max(1)
                predicted_class = predicted.item()

            # Store prediction and actual label
            all_preds.append(predicted_class)
            all_labels.append(class_to_idx[class_name])

    # Calculate accuracy
    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy

# Path to the test dataset
test_dir = 'D:\\Leukemia project\\Source_Code\\dataset\\dataset\\test'

# Run the evaluation
accuracy = evaluate_model(test_dir)
print(f" Accuracy: {accuracy * 100:.2f}%")
from collections import defaultdict
import numpy as np

# Function to evaluate weighted accuracy on test dataset
def evaluate_model_weighted(test_dir):
    all_preds = []
    all_labels = []
    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    # Iterate over the classes in the test folder
    for class_name in class_names:
        class_dir = os.path.join(test_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        # Get all images in the class directory
        for img_name in tqdm(os.listdir(class_dir), desc=f"Processing {class_name}"):
            img_path = os.path.join(class_dir, img_name)
            if not img_path.endswith(('png', 'jpg', 'jpeg')):
                continue

            # Load and preprocess the image
            image = Image.open(img_path)
            image = transform(image).unsqueeze(0).to(device)

            # Make prediction
            with torch.no_grad():
                outputs = model(image)
                _, predicted = outputs.max(1)
                predicted_class = predicted.item()

            # Store prediction and actual label
            true_label = class_to_idx[class_name]
            all_preds.append(predicted_class)
            all_labels.append(true_label)

            # Update class-wise correct predictions and total samples
            class_total[true_label] += 1
            if predicted_class == true_label:
                class_correct[true_label] += 1

    # Calculate class-wise accuracy and weighted accuracy
    class_accuracies = {}
    total_samples = sum(class_total.values())
    weighted_acc = 0

    for class_name, idx in class_to_idx.items():
        if class_total[idx] > 0:
            class_accuracies[class_name] = class_correct[idx] / class_total[idx]
            weighted_acc += class_accuracies[class_name] * (class_total[idx] / total_samples)

    # Print class-wise accuracies
    print("\nClass-wise accuracies:")
    for class_name, acc in class_accuracies.items():
        print(f"{class_name}: {acc * 100:.2f}%")

    return weighted_acc

# Path to the test dataset
test_dir = 'D:\\Leukemia project\\Source_Code\\dataset\\dataset\\test'

# Run the evaluation
weighted_accuracy = evaluate_model_weighted(test_dir)
print(f"Weighted test Accuracy: {weighted_accuracy * 100:.2f}%")
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.preprocessing import label_binarize
import numpy as np

# Function to evaluate all performance metrics on the test dataset
def evaluate_performance_metrics(test_dir):
    all_preds = []
    all_labels = []

    # Iterate over the classes in the test folder
    for class_name in class_names:
        class_dir = os.path.join(test_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        # Get all images in the class directory
        for img_name in tqdm(os.listdir(class_dir), desc=f"Processing {class_name}"):
            img_path = os.path.join(class_dir, img_name)
            if not img_path.endswith(('png', 'jpg', 'jpeg')):
                continue

            # Load and preprocess the image
            image = Image.open(img_path)
            image = transform(image).unsqueeze(0).to(device)

            # Make prediction
            with torch.no_grad():
                outputs = model(image)
                _, predicted = outputs.max(1)
                predicted_class = predicted.item()

            # Store prediction and actual label
            true_label = class_to_idx[class_name]
            all_preds.append(predicted_class)
            all_labels.append(true_label)

    # Calculate precision, recall, F1-score, and weighted F1-score
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    # Generate classification report for more detailed metrics
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    # AUC-ROC calculation (one-vs-rest approach)
    all_labels_bin = label_binarize(all_labels, classes=list(range(len(class_names))))
    all_preds_bin = label_binarize(all_preds, classes=list(range(len(class_names))))

    if len(class_names) > 2:
        roc_auc = roc_auc_score(all_labels_bin, all_preds_bin, average='weighted', multi_class='ovr')
    else:
        roc_auc = roc_auc_score(all_labels_bin, all_preds_bin, average='weighted')

    # Output performance metrics
    print(f"\nWeighted Precision: {precision * 100:.2f}%")
    print(f"Weighted Recall: {recall * 100:.2f}%")
    print(f"Weighted F1-Score: {f1 * 100:.2f}%")
    print(f"AUC-ROC: {roc_auc * 100:.2f}%")

# Path to the test dataset
test_dir = 'D:\\Leukemia project\\Source_Code\\dataset\\dataset\\test'

# Run the evaluation
evaluate_performance_metrics(test_dir)

