import os
import cv2
import numpy as np
from collections import defaultdict
from PIL import Image
import imagehash
from hashlib import md5
import pennylane as qml
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import imgaug.augmenters as iaa
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb


def extract_frames(video_path, output_folder, frame_rate=1):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return
    os.makedirs(output_folder, exist_ok=True)
    frame_count = 0
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_interval = max(1, fps // frame_rate)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_folder, f"frame_{frame_count}.jpg")
            cv2.imwrite(frame_filename, frame)
        frame_count += 1
    cap.release()
    print(f"Frames extracted and saved in {output_folder}")


def compute_phash(image_path):
    """Compute perceptual hash for an image."""
    img = Image.open(image_path).convert('L')
    return imagehash.phash(img)
def detect_duplicate_frames(folder_path, threshold=1):
    """Detect duplicate frames based on perceptual hash similarity."""
    frame_hashes = {}
    duplicates = defaultdict(list)
    for filename in sorted(os.listdir(folder_path)):
        frame_path = os.path.join(folder_path, filename)
        if os.path.isfile(frame_path):
            try:
                phash = compute_phash(frame_path)
                for existing_hash in frame_hashes:
                    if abs(phash - existing_hash) <= threshold:
                        duplicates[frame_hashes[existing_hash]].append(filename)
                        break
                else:
                    frame_hashes[phash] = filename
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    duplicate_frames = dict(duplicates)
    if duplicate_frames:
        print("Duplicate frames detected:")
        for original, dups in duplicate_frames.items():
            print(f"{original} -> {dups}")
    else:
        print("No duplicate frames found.")



def compute_hash(image):
    """Compute perceptual hash of an image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (16, 16))
    avg = resized.mean()
    hash_val = ''.join('1' if pixel > avg else '0' for pixel in resized.flatten())
    return md5(hash_val.encode()).hexdigest()
def detect_tampering(folder_path):
    """Detect frame tampering based on image hash comparisons."""
    hashes = {}
    tampered_frames = []
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(('.jpg', '.png', '.bmp')):
            frame_path = os.path.join(folder_path, filename)
            frame = cv2.imread(frame_path)
            if frame is None:
                continue
            frame_hash = compute_hash(frame)
            if frame_hash in hashes.values():
                continue
            if hashes and frame_hash not in hashes.values():
                tampered_frames.append(filename)
            hashes[filename] = frame_hash
    if tampered_frames:
        print("Tampered frames detected:", tampered_frames)
    else:
        print("No tampering detected.")




def detect_watermark(input_folder, output_folder, low_threshold=50, high_threshold=150, blur_ksize=(5,5)):
    watermark_features = {}

    for filename in sorted(os.listdir(input_folder)):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(input_folder, filename)
            img = cv2.imread(img_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, blur_ksize, 0)
            thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            edges = cv2.Canny(thresh, low_threshold, high_threshold)

            if np.any(edges):
                watermark_features[filename] = 1
            else:
                watermark_features[filename] = 0
    return watermark_features



num_qubits = 4
dev = qml.device("default.qubit", wires=num_qubits)
def quantum_conv_layer(inputs, q_weights):
    qml.AngleEmbedding(inputs, wires=range(num_qubits))
    qml.BasicEntanglerLayers(q_weights, wires=range(num_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]
weight_shapes = {"q_weights": (3, num_qubits)}
qnode = qml.QNode(quantum_conv_layer, dev, interface="numpy")
class QuantumLayer(Layer):
    def __init__(self, num_qubits, **kwargs):
        super(QuantumLayer, self).__init__(**kwargs)
        self.num_qubits = num_qubits
        self.dev = qml.device("default.qubit", wires=num_qubits)
    def build(self, input_shape):
        self.q_weights = self.add_weight(
            name="q_weights",
            shape=(3, self.num_qubits),
            initializer="random_normal",
            trainable=True
        )
    def call(self, inputs):
        inputs_np = tf.keras.backend.eval(inputs) if isinstance(inputs, tf.Tensor) else np.array(inputs)
        def process_batch(batch):
            @qml.qnode(self.dev)
            def circuit(x):
                for i in range(self.num_qubits):
                    qml.RY(x[i], wires=i)
                return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]
            return np.array([circuit(x) for x in batch])
        outputs = tf.numpy_function(process_batch, [inputs_np], tf.float32)
        return outputs
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.num_qubits)
def create_qcnn():
    model = keras.Sequential([
        layers.Input(shape=(4,)),
        QuantumLayer(num_qubits, name="quantum_layer"),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(16, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"], run_eagerly=True)
    return model
def augment_image(img):
    if img is None:
        raise ValueError("Invalid image: Check file paths or corrupt images.")
    img = img.astype(np.uint8)
    seq = iaa.Sequential([
        iaa.GaussianBlur(sigma=(0, 1.0)),
        iaa.Affine(rotate=(-10, 10)),
        iaa.AdditiveGaussianNoise(scale=(0, 0.05 * 255))
    ])
    return seq.augment_image(img)
def load_frames(folder):
    images, labels = [], []
    for file in os.listdir(folder):
        if file.startswith('.'):
            print(f"Skipping invalid image: {file}")
            continue
        path = os.path.join(folder, file)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Skipping corrupted or unreadable image: {file}")
            continue
        img = augment_image(img)
        img = cv2.resize(img, (64, 64))
        img = img.flatten() / 127.5 - 1
        images.append(img)
        labels.append(0 if "pirated" in file.lower() else 1)
    images = np.array(images)
    labels = np.array(labels)
    if images.shape[0] >= 4:
        pca = PCA(n_components=4)
        images = pca.fit_transform(images)
    else:
        print("Warning: Not enough samples for PCA. Padding with zeros.")
        while images.shape[0] < 4:
            images = np.vstack([images, np.zeros(images.shape[1])])
    return images, labels
def train_qcnn(folder):
    x_data, y_data = load_frames(folder)
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)
    model = create_qcnn()
    model.fit(x_train, y_train, epochs=10, batch_size=16, validation_data=(x_test, y_test))
    loss, accuracy = model.evaluate(x_test, y_test)
    print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")
    return model, x_train, x_test, y_train, y_test
def extract_qcnn_features(model, x_data):
    x_data = np.array(x_data) if not isinstance(x_data, np.ndarray) else x_data
    quantum_layer = next(layer for layer in model.layers if isinstance(layer, QuantumLayer))
    features = quantum_layer(x_data).numpy()
    return features
def plot_qcnn_features(features, y_data):
    plt.figure(figsize=(8, 6))
    for i in range(features.shape[1]):
        plt.scatter(range(features.shape[0]), features[:, i], label=f'Feature {i}')
    plt.xlabel('Sample Index')
    plt.ylabel('Feature Value')
    plt.title('Extracted QCNN Features')
    plt.legend()
    plt.show()
dataset = "/content/output2"
model, x_train, x_test, y_train, y_test = train_qcnn(dataset)
_ = model.predict(x_test[:1])
quantum_layer = model.get_layer(name="quantum_layer")
model.summary()


def main():
    video_folder = input("Enter the path of the file: ")
    frame_folder = input("Enter the frame folder: ")
    watermark_folder=input("Enter the watermark folder:")
    os.makedirs(frame_folder, exist_ok=True)
    os.makedirs(watermark_folder, exist_ok=True)
    extract_frames(video_folder, frame_folder)
    detect_duplicate_frames(frame_folder)
    detect_tampering(frame_folder)
    detect_watermark(frame_folder,watermark_folder)
    video_frames_folder2=frame_folder
    x_data, y_data = load_frames(video_frames_folder2)
    features = extract_qcnn_features(model, x_data)
    plot_qcnn_features(features, y_data)


    def extract_qcnn_feature(model, data):
         return np.random.rand(len(data), 10)
    def train_xgboost(X_train, y_train):
        model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss')
        model.fit(X_train, y_train)
        return model
    data_size = 100
    y_data = np.random.randint(0, 2, data_size)
    X_data = np.random.rand(data_size, 64, 64)
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

    features_train = extract_qcnn_feature(None, X_train)
    features_test = extract_qcnn_feature(None, X_test)

    xgb_model = train_xgboost(features_train, y_train)

    predictions = xgb_model.predict(features_test)
    print("Accuracy:", accuracy_score(y_test, predictions))
    def check_piracy(video_frames):
        video_frames = video_frames.reshape(video_frames.shape[0], -1)
        prediction = model.predict(video_frames)
        for pred in prediction:
            if pred == 1:
                print("Video is pirated")
                break
            else:
                print("Video is not pirated")
                break

    check_piracy(X_test)

    print("Classification Report:\n", classification_report(y_test, predictions))

    plt.figure(figsize=(6, 4))
    sns.heatmap(confusion_matrix(y_test, predictions), annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
    xgb.plot_importance(xgb_model)
    plt.title("XGBoost Feature Importance")
    plt.show()


if __name__ == "__main__":
    main()

