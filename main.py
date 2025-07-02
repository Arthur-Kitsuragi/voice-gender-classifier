import sys
import os
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
import torch
import librosa
from PyQt6.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QComboBox
import pretrainedmodels
from torch import nn
import torchvision.transforms as T
import sys

# Параметры
SAMPLE_RATE = 16000
DURATION = 10  # секунд
FILENAME = "recorded.wav"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model_name = 'resnet18'
model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
num_features = 512
model.last_linear = nn.Linear(num_features, 2)
# Загрузка модели
model.load_state_dict(torch.load("resnet18_trained_weights.pth", map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

#print(sd.query_devices())

def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled

def predict_gender_correct(filepath, model, device):
    hop_length = 512
    n_mels = 224
    time_steps = 384
    length_samples = hop_length * time_steps

    # 1. Load audio
    y, sr = librosa.load(filepath, sr=16000)
    y = y[:length_samples]

    # 2. Compute log-Mel spectrogram
    mels = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels,
                                          n_fft=hop_length * 2, hop_length=hop_length)
    mels = np.log(mels + 1e-9)

    # 3. Scale to 0–255 and invert
    img = scale_minmax(mels, 0, 255).astype(np.uint8)
    img = np.flip(img, axis=0)
    img = 255 - img

    # 4. Convert to tensor and resize
    img = img[np.newaxis, ...]  # (1, H, W)
    transform = T.Resize((224, 224))
    img_tensor = transform(torch.from_numpy(img))  # (1, 224, 224)

    # 5. Expand to 3 channels for ResNet
    img_tensor = img_tensor.repeat(3, 1, 1)  # (3, 224, 224)
    img_tensor = img_tensor.float() / 255.0  # normalize to [0,1]
    img_tensor = img_tensor.unsqueeze(0).to(device)  # (1, 3, 224, 224)

    # 6. Predict
    model.eval()
    with torch.no_grad():
        output = model(img_tensor)
        pred = torch.argmax(output, dim=1).item()

    return "Мужчина" if pred == 0 else "Женщина"

class VoiceGenderApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Определение пола по голосу")
        self.layout = QVBoxLayout()

        self.label = QLabel("Нажмите кнопку для записи")
        self.layout.addWidget(self.label)

        self.device_combo = QComboBox()
        all_devices = sd.query_devices()
        self.input_devices = [d for d in all_devices if d['max_input_channels'] > 0]
        for i, dev in enumerate(self.input_devices):
            self.device_combo.addItem(f"{i}: {dev['name']}")
        self.layout.addWidget(self.device_combo)

        self.button = QPushButton("Записать голос")
        self.button.clicked.connect(self.record_and_predict)
        self.layout.addWidget(self.button)

        self.setLayout(self.layout)

    def record_and_predict(self):
        self.label.setText("🔴 Запись... Говорите!")
        QApplication.processEvents()

        try:
            selected_index = self.device_combo.currentIndex()
            device_id = self.input_devices[selected_index]['index']

            audio = sd.rec(int(SAMPLE_RATE * DURATION), samplerate=SAMPLE_RATE,
                           channels=1, dtype='float32', device=device_id)
            sd.wait()
            write(FILENAME, SAMPLE_RATE, audio)

            self.label.setText("🎧 Обработка...")
            QApplication.processEvents()

            result = predict_gender_correct(FILENAME, model, DEVICE)
            self.label.setText(f"👤 Результат: {result}")

        except Exception as e:
            self.label.setText(f"❌ Ошибка: {str(e)}")
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VoiceGenderApp()
    window.show()
    sys.exit(app.exec())
