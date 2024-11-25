import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, f1_score
import tensorflow as tf
import matplotlib.pyplot as plt

# بارگذاری داده‌ها از CSV
dataset_path = "I:\\IOT\\HW3\\For F\\Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
data = pd.read_csv(dataset_path)

# تنظیم نام ستون‌ها
columns = ['Destination Port', 'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
           'Total Length of Fwd Packets', 'Total Length of Bwd Packets', 'Fwd Packet Length Max',
           'Fwd Packet Length Min', 'Fwd Packet Length Mean', 'Fwd Packet Length Std', 'Bwd Packet Length Max',
           'Bwd Packet Length Min', 'Bwd Packet Length Mean', 'Bwd Packet Length Std', 'Flow Bytes/s',
           'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Total',
           'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Total', 'Bwd IAT Mean',
           'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags',
           'Bwd URG Flags', 'Fwd Header Length', 'Bwd Header Length', 'Fwd Packets/s', 'Bwd Packets/s',
           'Min Packet Length', 'Max Packet Length', 'Packet Length Mean', 'Packet Length Std',
           'Packet Length Variance', 'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count',
           'ACK Flag Count', 'URG Flag Count', 'CWE Flag Count', 'ECE Flag Count', 'Down/Up Ratio',
           'Average Packet Size', 'Avg Fwd Segment Size', 'Avg Bwd Segment Size', 'Fwd Header Length.1',
           'Fwd Avg Bytes/Bulk', 'Fwd Avg Packets/Bulk', 'Fwd Avg Bulk Rate', 'Bwd Avg Bytes/Bulk',
           'Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate', 'Subflow Fwd Packets', 'Subflow Fwd Bytes',
           'Subflow Bwd Packets', 'Subflow Bwd Bytes', 'Init_Win_bytes_forward', 'Init_Win_bytes_backward',
           'act_data_pkt_fwd', 'min_seg_size_forward', 'Active Mean', 'Active Std', 'Active Max', 'Active Min',
           'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min', 'Label']
data.columns = columns

# پیش‌پردازش داده‌ها
data = data.replace([np.inf, -np.inf], np.nan)
data = data.dropna()
features = data.drop('Label', axis=1)
labels = data['Label']

# کدگذاری ویژگی‌های متنی به اعداد
le = LabelEncoder()
labels = le.fit_transform(labels)

# مقیاس‌بندی ویژگی‌ها
scaler = StandardScaler()
features = scaler.fit_transform(features)

# تقسیم داده‌ها
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

# ایجاد داده‌های زمانی
timesteps = 100
sequences_train, labels_train = [], []
for i in range(len(X_train) - timesteps):
    sequences_train.append(X_train[i:i+timesteps])
    labels_train.append(y_train[i+timesteps-1])
X_train_ts, y_train_ts = np.array(sequences_train), np.array(labels_train)

sequences_test, labels_test = [], []
for i in range(len(X_test) - timesteps):
    sequences_test.append(X_test[i:i+timesteps])
    labels_test.append(y_test[i+timesteps-1])
X_test_ts, y_test_ts = np.array(sequences_test), np.array(labels_test)

# تعریف و آموزش مدل LSTM
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, input_shape=(X_train_ts.shape[1], X_train_ts.shape[2])),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train_ts, y_train_ts, epochs=10, batch_size=64, validation_data=(X_test_ts, y_test_ts))

# ارزیابی مدل
y_pred = (model.predict(X_test_ts) > 0.5).astype("int32")
print("Accuracy:", accuracy_score(y_test_ts, y_pred))
print("Classification Report:\n", classification_report(y_test_ts, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test_ts, y_pred))
print("F1 Score:", f1_score(y_test_ts, y_pred))

# رسم نمودار دقت
plt.plot(history.history['accuracy'], label='Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()
