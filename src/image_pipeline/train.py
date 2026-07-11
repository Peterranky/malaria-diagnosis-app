import os
import json
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

def build_model():
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False
    
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
    return model

def preprocess(features):
    image = tf.image.resize(features['image'], IMG_SIZE)
    # MobileNetV2 expects inputs in [-1, 1]
    image = (image / 127.5) - 1.0
    label = features['label']
    return image, label

def main():
    print("Loading NIH Malaria dataset...")
    # Load with info to get splits
    ds, info = tfds.load('malaria', with_info=True)
    
    # Split manually: 70% train, 15% val, 15% test
    # Total = 27558. Train = 19290, Val = 4134, Test = 4134
    
    train_ds = tfds.load('malaria', split='train[:70%]')
    val_ds = tfds.load('malaria', split='train[70%:85%]')
    test_ds = tfds.load('malaria', split='train[85%:]')
    
    train_ds = train_ds.map(preprocess).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.map(preprocess).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    test_ds_batched = test_ds.map(preprocess).batch(BATCH_SIZE)
    
    model = build_model()
    print("Training model...")
    # Train for 3 epochs for prototype demonstration
    model.fit(train_ds, validation_data=val_ds, epochs=3)
    
    # Save model
    models_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
    os.makedirs(models_dir, exist_ok=True)
    metrics_dir = os.path.join(models_dir, 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)
    
    model_path = os.path.join(models_dir, 'image_model.keras')
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    # Evaluation on Test set
    print("Evaluating on test set...")
    y_true = []
    y_pred_probs = []
    
    for images, labels in test_ds_batched:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred_probs.extend(preds.flatten())
        
    y_true = np.array(y_true)
    y_pred = (np.array(y_pred_probs) > 0.5).astype(int)
    
    # Label 0: uninfected, Label 1: parasitized (based on tfds malaria dataset definition)
    # Actually, in tfds malaria: 0 = parasitized, 1 = uninfected. Let's verify.
    # tfds info: features=FeaturesDict({'image': Image(shape=(None, None, 3), dtype=uint8), 'label': ClassLabel(shape=(), dtype=int64, num_classes=2)})
    # tfds info says: 0 = parasitized, 1 = uninfected.
    # We will map 0->Parasitized, 1->Uninfected.
    
    acc = accuracy_score(y_true, y_pred)
    # Treating 0 as the positive class for sensitivity/specificity calculations if we want to predict "Parasitized"
    # But let's just use sklearn metrics natively
    cm = confusion_matrix(y_true, y_pred)
    tp = cm[0, 0]   # parasitized correctly detected
    fn = cm[0, 1]   # parasitized missed -> false negatives
    fp = cm[1, 0]   # uninfected flagged as parasitized
    tn = cm[1, 1]   # uninfected correctly cleared

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision   = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall      = sensitivity
    f1          = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy    = (tp + tn) / (tp + tn + fp + fn)

    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity)
    }
    metrics_path = os.path.join(metrics_dir, 'image_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
        
    print(f"Metrics saved to {metrics_path}")
    
    # Plot Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Parasitized', 'Uninfected'], yticklabels=['Parasitized', 'Uninfected'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Image Diagnosis Confusion Matrix')
    cm_path = os.path.join(metrics_dir, 'image_confusion_matrix.png')
    plt.savefig(cm_path)
    plt.close()
    print(f"Confusion matrix saved to {cm_path}")

if __name__ == "__main__":
    main()
