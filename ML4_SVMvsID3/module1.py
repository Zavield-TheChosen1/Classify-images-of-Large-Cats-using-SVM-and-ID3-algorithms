# Import necessary libraries
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import os
import matplotlib.pyplot as plt

from tabulate import tabulate
import numpy as np

# Start of training time
start_time = time.time()

# Paths to JSON files
json_files = [
    "E:/Code/ML4_SVMvsID3/anh_nhan_dan_1.json",
    "E:/Code/ML4_SVMvsID3/anh_dan_nhan_2.json",
    "E:/Code/ML4_SVMvsID3/anh_dan_nhan_3.json",
    "E:/Code/ML4_SVMvsID3/anh_dan_nhan_6.json",
    "E:/Code/ML4_SVMvsID3/anh_dan_nhan_4.json"
]

# Initialize lists to store loss values for SVM and Decision Tree
loss_values_svm = []
loss_values_decision_tree = []

# Initialize an empty list to store image data from all files
all_image_data = []

# Loop through JSON files
for json_file in json_files:
    if os.path.exists(json_file):
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if "_via_img_metadata" in data and isinstance(data["_via_img_metadata"], dict):
            print(f"Image data in file {json_file} is valid.")
        else:
            print(f"Image data in file {json_file} is invalid.")
    else:
        print(f"File {json_file} does not exist.")

# Initialize an empty list to store image data from all files
all_image_data = []

for json_file in json_files:
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    if "_via_img_metadata" in data and isinstance(data["_via_img_metadata"], dict):
        for image_id, image_info in data["_via_img_metadata"].items():
            if "regions" in image_info and isinstance(image_info["regions"], list):
                print(f"Found valid region information in file {json_file}.")
            else:
                print(f"No valid region information found in file {json_file}.")

    image_data = []

    for image_id, image_info in data["_via_img_metadata"].items():
        filename = image_info["filename"]
        size = image_info["size"]

        if "regions" in image_info:
            regions = image_info["regions"]
            if len(regions) > 0:
                region_attributes = regions[0].get("region_attributes", {})
                name = region_attributes.get("Name", "Undefined")
                type_ = region_attributes.get("Species", "Undefined")
                identifying = region_attributes.get("Identifying", "Undefined")
            else:
                name = "Undefined"
                type_ = "Undefined"
                identifying = "Undefined"
        else:
            name = "Undefined"
            type_ = "Undefined"
            identifying = "Undefined"

        image_data.append({
            "filename": filename,
            "size": size,
            "name": name,
            "type": type_,
            "identifying": identifying
        })

    df = pd.DataFrame(image_data)
    label_encoder = LabelEncoder()
    string_columns = ["name", "type", "identifying"]

    for col in string_columns:
        df[col] = label_encoder.fit_transform(df[col])

    df.drop("filename", axis=1, inplace=True)
    all_image_data.append(df)

combined_data = pd.concat(all_image_data, ignore_index=True)
x_train, x_test, y_train, y_test = train_test_split(combined_data.drop("type", axis=1), combined_data["type"], test_size=0.2)

train_size = len(x_train)
test_size = len(x_test)

# SVM Model

svm_model = SVC()
svm_model.fit(x_train, y_train)

actual_test_ratio = test_size / (train_size + test_size)

if actual_test_ratio == 0.2:
    print("Training and test set split performed correctly with a test ratio of 0.2.")
else:
    print("There is an error in the training and test set split, or the test ratio is not appropriate.")

y_pred_svm = svm_model.predict(x_test)

current_loss_svm = svm_model.score(x_test, y_test)
loss_values_svm.append(current_loss_svm)

accuracy_svm = accuracy_score(y_test, y_pred_svm)
print("Accuracy of SVM model:", accuracy_svm)

precision_svm = precision_score(y_test, y_pred_svm, average='macro')
recall_svm = recall_score(y_test, y_pred_svm, average='macro')
print("Recall of SVM model:", recall_svm)
print("Precision of SVM model:", precision_svm)

confusion_svm = confusion_matrix(y_test, y_pred_svm)
print("Confusion matrix of SVM model:\n", confusion_svm)

model_svm_path = 'trained_svm_model.joblib'
joblib.dump(svm_model, model_svm_path)

# Decision Tree Model

decision_tree_model = DecisionTreeClassifier()
decision_tree_model.fit(x_train, y_train)

actual_test_ratio = test_size / (train_size + test_size)

if actual_test_ratio == 0.2:
    print("Training and test set split performed correctly with a test ratio of 0.2.")
else:
    print("There is an error in the training and test set split, or the test ratio is not appropriate.")

y_pred_decision_tree = decision_tree_model.predict(x_test)

current_loss_decision_tree = decision_tree_model.score(x_test, y_test)
loss_values_decision_tree.append(current_loss_decision_tree)

accuracy_decision_tree = accuracy_score(y_test, y_pred_decision_tree)
print("Accuracy of Decision Tree model:", accuracy_decision_tree)

precision_decision_tree = precision_score(y_test, y_pred_decision_tree, average='macro')
recall_decision_tree = recall_score(y_test, y_pred_decision_tree, average='macro')
print("Recall of Decision Tree model:", recall_decision_tree)
print("Precision of Decision Tree model:", precision_decision_tree)

confusion_decision_tree = confusion_matrix(y_test, y_pred_decision_tree)
print("Confusion matrix of Decision Tree model:\n", confusion_decision_tree)

model_decision_tree_path = 'trained_decision_tree_model.joblib'
joblib.dump(decision_tree_model, model_decision_tree_path)

end_time = time.time()

training_time_svm = end_time - start_time
training_time_decision_tree = end_time - start_time

# SVM Model Evaluation (creating a table)
svm_accuracy = accuracy_score(y_test, y_pred_svm)
svm_precision = precision_score(y_test, y_pred_svm, average='macro')
svm_recall = recall_score(y_test, y_pred_svm, average='macro')
svm_confusion = confusion_matrix(y_test, y_pred_svm)

# Decision Tree Model Evaluation (creating a table)
decision_tree_accuracy = accuracy_score(y_test, y_pred_decision_tree)
decision_tree_precision = precision_score(y_test, y_pred_decision_tree, average='macro')
decision_tree_recall = recall_score(y_test, y_pred_decision_tree, average='macro')
decision_tree_confusion = confusion_matrix(y_test, y_pred_decision_tree)

# Create a table for SVM model
def plot_evaluation(title, accuracy, precision, recall, confusion):
    plt.figure(figsize=(12, 4))
    
    # Evaluate accuracy, precision, recall
    plt.subplot(131)
    plt.bar(['Accuracy', 'Precision', 'Recall'], [accuracy, precision, recall])
    plt.title(f'{title} Evaluation Metrics')
    plt.ylim([0, 1])
    
    # Display confusion matrix
    plt.subplot(132)
    plt.imshow(confusion, interpolation='nearest', cmap=plt.get_cmap('Blues'))
    plt.title(f'{title} Confusion Matrix')
    plt.colorbar()
    
    # Display labels on the confusion matrix
    num_classes = len(confusion)
    plt.xticks(np.arange(num_classes), np.arange(1, num_classes + 1), rotation=45)
    plt.yticks(np.arange(num_classes), np.arange(1, num_classes + 1))
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    plt.tight_layout()
    plt.show()

# Plot evaluation for SVM
plot_evaluation('SVM', svm_accuracy, svm_precision, svm_recall, svm_confusion)

# Plot evaluation for Decision Tree
plot_evaluation('Decision Tree', decision_tree_accuracy, decision_tree_precision, decision_tree_recall, decision_tree_confusion)

# Evaluate SVM model
svm_score = svm_model.score(x_test, y_test)

svm_info = [
    ["Accuracy", f"{svm_score:.2%}"],
    ["Recall", f"{recall_svm:.2f}"],
    ["Precision", f"{precision_svm:.2f}"],
    ["Training Time", f"{training_time_svm:.2f} seconds"],
]
print("\nSVM Model Information:")
print(tabulate(svm_info, headers=["Feature", "Value"], tablefmt="fancy_grid"))

# Evaluate Decision Tree model
decision_tree_score = decision_tree_model.score(x_test, y_test)
decision_tree_info = [
    ["Accuracy", f"{decision_tree_score:.2%}"],
    ["Recall", f"{recall_decision_tree:.2f}"],
    ["Precision", f"{precision_decision_tree:.2f}"],
    ["Training Time", f"{training_time_decision_tree:.2f} seconds"],
]
print("\nDecision Tree Model Information:")
print(tabulate(decision_tree_info, headers=["Feature", "Value"], tablefmt="fancy_grid"))

# Print the number of data samples
sample_count = len(combined_data)
print(f"Number of data samples: {sample_count}")

print(f"Training time for SVM: {training_time_svm:.2f} seconds")
print(f"Training time for Decision Tree: {training_time_decision_tree:.2f} seconds")

# Plot loss values for SVM
plt.figure(figsize=(8, 6))
plt.plot(list(range(1, len(loss_values_svm) + 1)), loss_values_svm, marker='o', linestyle='-', color='b')
plt.title('Loss Function Values for SVM over Training Iterations')
plt.xlabel('Training Iterations')
plt.ylabel('Loss Value')
plt.grid(True)
plt.show()

# Plot loss values for Decision Tree
plt.figure(figsize=(8, 6))
plt.plot(list(range(1, len(loss_values_decision_tree) + 1)), loss_values_decision_tree, marker='o', linestyle='-', color='g')
plt.title('Loss Function Values for Decision Tree over Training Iterations')
plt.xlabel('Training Iterations')
plt.ylabel('Loss Value')
plt.grid(True)
plt.show()

# Training completion message
print(f"Training has completed. The trained SVM model has been saved at: {model_svm_path}")
print(f"The trained Decision Tree model has been saved at: {model_decision_tree_path}")
