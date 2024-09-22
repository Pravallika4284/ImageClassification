# Import required libraries
from sklearn import datasets, svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load the digits dataset (MNIST)
digits = datasets.load_digits()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2, random_state=42)

# Initialize the SVM classifier
clf = svm.SVC(gamma=0.001)

# Train the classifier with the training data
clf.fit(X_train, y_train)

# Predict labels for the test set
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Display some sample images from the dataset
plt.gray()  # Use grayscale color map
plt.matshow(digits.images[0])  # Show the first image in the dataset
plt.show()
