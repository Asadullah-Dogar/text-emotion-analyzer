# 🚀 Emotion Analyzer

**Real-time Emotion Detection with Deep Learning and Streamlit**

Analyze emotions in text with this interactive Streamlit application! This project leverages a robust neural network built with TensorFlow/Keras, comprehensive preprocessing, and an intuitive UI for training, evaluation, and real-time emotion predictions.

---

## ✨ Key Features

### 📊 Data Pipeline

* **Automatic Loading**: Seamlessly imports training, validation, and test sets from `train.txt`, `val.txt`, and `test.txt`.
* **Text Preprocessing**: Cleans, tokenizes, and pads input data for optimal model performance.
* **Label Encoding**: Converts emotion categories into machine-readable labels.
* **Visual Insights**: Explore class distributions via real-time plots and charts.

### 🧠 Model Architecture

* **Bidirectional LSTMs**: Captures context from both directions in text sequences.
* **Embedding Layer**: Learns semantic representations of words.
* **Regularization**: Dropout layers prevent overfitting.
* **Dense Layers**: ReLU-activated layers for deep pattern learning.
* **Softmax Output**: Predicts emotion class probabilities.

### 🎯 Training & Evaluation

* **Early Stopping**: Stops training based on validation loss improvements.
* **Learning Rate Scheduler**: Dynamically adjusts the learning rate for faster convergence.
* **Live Monitoring**: Watch progress through Streamlit progress bars.
* **Model Metrics**: View accuracy, loss curves, confusion matrix, and full classification report.

### 🔮 Interactive Prediction

* **Flexible Input**: Type directly, upload files, or use example inputs.
* **Confidence Scores**: See predicted emotions with probability bars and relevant emojis.
* **Visual Feedback**: Dynamic confidence charts and detailed breakdowns for every prediction.

---

## 🛠️ How to Use

### 1. Install Dependencies

```bash
pip install streamlit tensorflow pandas numpy matplotlib seaborn plotly scikit-learn
```

### 2. Project Structure

```
emotion-analyzer/
├── emotion_analyzer.py     # Main Streamlit application
├── train.txt               # Training data (e.g., 16,000+ samples)
├── test.txt                # Testing data
└── val.txt                 # Validation data
```

> **Note:** Place all `.txt` files in the same directory as `emotion_analyzer.py`.

### 3. Launch the App

```bash
streamlit run emotion_analyzer.py
```

### 4. Train the Model

1. App auto-detects data files on startup.
2. Click **🎯 Train Model** in the sidebar.
3. Monitor training with live metrics.
4. Model is auto-saved after training.

### 5. Predict Emotions

* Type your text or upload a file.
* View instant predictions with confidence breakdowns.
* Explore rich visuals and emoji-enhanced insights.

---

## 📊 Expected Performance

| Metric          | Expected Value |
| --------------- | -------------- |
| Accuracy        | 85–95%         |
| Generalization  | High           |
| Inference Speed | Real-time      |

> With a substantial dataset (e.g., 16k+ samples), the model delivers high accuracy and handles diverse inputs robustly.

---

## 📸 Demo Preview

*Coming Soon: GIF or screenshots showcasing real-time emotion prediction interface.*

---

## 📁 Dataset Format

Each line in the dataset should follow this structure:

```
<emotion_label>\t<text>
```

**Example:**

```
joy	I just got a promotion at work!
sadness	I miss my family so much right now.
```

---

## 🤝 Contributing

Feel free to open issues or submit pull requests. Feedback and collaboration are always welcome!

