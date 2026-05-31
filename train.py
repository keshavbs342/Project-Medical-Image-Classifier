# %%
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16
import matplotlib.pyplot as plt
import tensorflow as tf
!pip install tensorflow matplotlib pandas numpy opencv-python

# %%

BATCH_SIZE = 32
IMAGE_SIZE = (224, 224)

print("Loading Training Data...")

train_dataset = tf.keras.utils.image_dataset_from_directory(
    'data/train',
    shuffle=True,
    batch_size=BATCH_SIZE,
    image_size=IMAGE_SIZE
)

print("\nLoading Validation Data...")
val_dataset = tf.keras.utils.image_dataset_from_directory(
    'data/val',
    shuffle=True,
    batch_size=BATCH_SIZE,
    image_size=IMAGE_SIZE
)

class_names = train_dataset.class_names
print(f"\nClasses found: {class_names}")

# %%
plt.figure(figsize=(10, 10))
for images, labels in train_dataset.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        # Convert pixel values back to standard integers for displaying
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
plt.show()

# %%

print("Step 1: Building the Deep Learning Model...")

# Load the pre-trained VGG16 model
base_model = VGG16(weights='imagenet', include_top=False,
                   input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze the base model

# Build our custom Medical Classifier on top
model = models.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
print("Model built successfully! Moving to training...\n")

# Train the model for 5 passes (epochs) over the dataset
print("Step 2: Starting Model Training...")
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=5
)

# Save the trained model to your computer
model.save("pneumonia_classifier.keras")
print("Model saved successfully as pneumonia_classifier.keras!")

# %%
print("Loading Test Data for Final Evaluation...")

# It is crucial that shuffle=False here so the labels match up perfectly
test_dataset = tf.keras.utils.image_dataset_from_directory(
    'data/test',
    shuffle=False,
    batch_size=BATCH_SIZE,
    image_size=IMAGE_SIZE
)

print("\nGrading the Model on Unseen Data...")
test_loss, test_acc = model.evaluate(test_dataset)
print(f"\n🏆 Final Test Accuracy: {test_acc * 100:.2f}%")

# %%
!pip install scikit-learn seaborn

# %%

print("Loading the saved model...")
# Load the pre-trained brain from your hard drive
model = tf.keras.models.load_model("pneumonia_classifier.keras")
print("Model loaded successfully! You can now skip the training cell.")

# %%
%pip install seaborn scikit-learn

# %%

print("Generating predictions on the test data...")
# Have the model predict the diagnosis for all 624 test images
predictions = model.predict(test_dataset)
# Convert percentage probabilities into strict 0 (Normal) or 1 (Pneumonia) predictions
predicted_classes = np.where(predictions > 0.5, 1, 0)

# Get the actual, true labels from the dataset to compare
true_classes = np.concatenate([y for x, y in test_dataset], axis=0)

# Calculate the Confusion Matrix
cm = confusion_matrix(true_classes, predicted_classes)

# Draw the colorful Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal', 'Pneumonia'],
            yticklabels=['Normal', 'Pneumonia'])
plt.title('Confusion Matrix - Pneumonia Detection')
plt.ylabel('True Diagnosis (What the doctor said)')
plt.xlabel('Predicted Diagnosis (What the AI said)')
plt.show()

# Print a detailed statistical report
print("\n--- Detailed Diagnostic Report ---")
print(classification_report(true_classes, predicted_classes,
      target_names=['Normal', 'Pneumonia']))
