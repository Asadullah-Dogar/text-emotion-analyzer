import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import re
import pickle
import os
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="🤖 Emotion-in-Text Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .metric-container {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        border-left: 5px solid #667eea;
    }
    
    .emotion-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        margin: 10px 0;
    }
    
    .success-message {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    
    .error-message {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

class EmotionAnalyzer:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.label_encoder = None
        self.max_length = 100
        self.vocab_size = 10000
        self.embedding_dim = 128
        self.lstm_units = 64
        self.emotions = []
        self.emotion_emojis = {
            'sadness': '😢',
            'joy': '😊',
            'love': '❤️',
            'anger': '😠',
            'fear': '😨',
            'surprise': '😲'
        }
    
    def load_data(self, file_path):
        """Load and parse emotion data from text files"""
        texts = []
        emotions = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                for line in file:
                    line = line.strip()
                    if line and ';' in line:
                        # Split by the last semicolon to get text and emotion
                        parts = line.rsplit(';', 1)
                        if len(parts) == 2:
                            text, emotion = parts
                            texts.append(text.strip())
                            emotions.append(emotion.strip())
            
            return texts, emotions
        except Exception as e:
            st.error(f"Error loading {file_path}: {str(e)}")
            return [], []
    
    def preprocess_text(self, text):
        """Clean and preprocess text data"""
        # Convert to lowercase
        text = text.lower()
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Remove extra whitespaces
        text = ' '.join(text.split())
        return text
    
    def prepare_data(self, train_file, test_file, val_file):
        """Load and prepare all datasets"""
        # Load data
        train_texts, train_emotions = self.load_data(train_file)
        test_texts, test_emotions = self.load_data(test_file)
        val_texts, val_emotions = self.load_data(val_file)
        
        if not train_texts:
            return None, None, None, None, None, None
        
        # Preprocess texts
        train_texts = [self.preprocess_text(text) for text in train_texts]
        test_texts = [self.preprocess_text(text) for text in test_texts]
        val_texts = [self.preprocess_text(text) for text in val_texts]
        
        # Combine all texts for tokenizer
        all_texts = train_texts + test_texts + val_texts
        
        # Initialize and fit tokenizer
        self.tokenizer = Tokenizer(num_words=self.vocab_size, oov_token='<OOV>')
        self.tokenizer.fit_on_texts(all_texts)
        
        # Convert texts to sequences
        train_sequences = self.tokenizer.texts_to_sequences(train_texts)
        test_sequences = self.tokenizer.texts_to_sequences(test_texts)
        val_sequences = self.tokenizer.texts_to_sequences(val_texts)
        
        # Pad sequences
        train_padded = pad_sequences(train_sequences, maxlen=self.max_length, padding='post')
        test_padded = pad_sequences(test_sequences, maxlen=self.max_length, padding='post')
        val_padded = pad_sequences(val_sequences, maxlen=self.max_length, padding='post')
        
        # Encode labels
        all_emotions = train_emotions + test_emotions + val_emotions
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(all_emotions)
        self.emotions = list(self.label_encoder.classes_)
        
        train_labels = to_categorical(self.label_encoder.transform(train_emotions))
        test_labels = to_categorical(self.label_encoder.transform(test_emotions))
        val_labels = to_categorical(self.label_encoder.transform(val_emotions))
        
        return (train_padded, train_labels), (test_padded, test_labels), (val_padded, val_labels)
    
    def build_model(self):
        """Build the neural network model"""
        model = Sequential([
            Embedding(self.vocab_size, self.embedding_dim, input_length=self.max_length),
            Bidirectional(LSTM(self.lstm_units, dropout=0.3, recurrent_dropout=0.3, return_sequences=True)),
            Bidirectional(LSTM(self.lstm_units//2, dropout=0.3, recurrent_dropout=0.3)),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(32, activation='relu'),
            Dropout(0.3),
            Dense(len(self.emotions), activation='softmax')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def train_model(self, train_data, val_data, epochs=50, batch_size=32):
        """Train the model"""
        train_X, train_y = train_data
        val_X, val_y = val_data
        
        # Callbacks
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True
        )
        
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=0.0001
        )
        
        # Train the model
        history = self.model.fit(
            train_X, train_y,
            validation_data=(val_X, val_y),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        return history
    
    def evaluate_model(self, test_data):
        """Evaluate the model on test data"""
        test_X, test_y = test_data
        
        # Predictions
        y_pred = self.model.predict(test_X)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(test_y, axis=1)
        
        # Classification report
        report = classification_report(
            y_true_classes, y_pred_classes,
            target_names=self.emotions,
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_true_classes, y_pred_classes)
        
        return report, cm, y_pred
    
    def predict_emotion(self, text):
        """Predict emotion for a single text"""
        if not self.model or not self.tokenizer:
            return None
        
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        # Convert to sequence
        sequence = self.tokenizer.texts_to_sequences([processed_text])
        padded = pad_sequences(sequence, maxlen=self.max_length, padding='post')
        
        # Predict
        prediction = self.model.predict(padded)[0]
        
        # Get results
        results = []
        for i, emotion in enumerate(self.emotions):
            results.append({
                'emotion': emotion,
                'confidence': float(prediction[i]),
                'emoji': self.emotion_emojis.get(emotion, '🤔')
            })
        
        # Sort by confidence
        results.sort(key=lambda x: x['confidence'], reverse=True)
        
        return results
    
    def save_model(self, model_path='emotion_model'):
        """Save the trained model and tokenizer"""
        if self.model:
            self.model.save(f'{model_path}.h5')
            
            # Save tokenizer and label encoder
            with open(f'{model_path}_tokenizer.pkl', 'wb') as f:
                pickle.dump(self.tokenizer, f)
            
            with open(f'{model_path}_label_encoder.pkl', 'wb') as f:
                pickle.dump(self.label_encoder, f)
            
            # Save metadata
            metadata = {
                'emotions': self.emotions,
                'max_length': self.max_length,
                'vocab_size': self.vocab_size
            }
            with open(f'{model_path}_metadata.pkl', 'wb') as f:
                pickle.dump(metadata, f)
            
            return True
        return False
    
    def load_model(self, model_path='emotion_model'):
        """Load a trained model"""
        try:
            self.model = tf.keras.models.load_model(f'{model_path}.h5')
            
            with open(f'{model_path}_tokenizer.pkl', 'rb') as f:
                self.tokenizer = pickle.load(f)
            
            with open(f'{model_path}_label_encoder.pkl', 'rb') as f:
                self.label_encoder = pickle.load(f)
            
            with open(f'{model_path}_metadata.pkl', 'rb') as f:
                metadata = pickle.load(f)
                self.emotions = metadata['emotions']
                self.max_length = metadata['max_length']
                self.vocab_size = metadata['vocab_size']
            
            return True
        except:
            return False

def plot_training_history(history):
    """Plot training history"""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Model Accuracy', 'Model Loss')
    )
    
    # Accuracy plot
    fig.add_trace(
        go.Scatter(
            x=list(range(1, len(history.history['accuracy']) + 1)),
            y=history.history['accuracy'],
            mode='lines+markers',
            name='Training Accuracy',
            line=dict(color='blue')
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=list(range(1, len(history.history['val_accuracy']) + 1)),
            y=history.history['val_accuracy'],
            mode='lines+markers',
            name='Validation Accuracy',
            line=dict(color='red')
        ),
        row=1, col=1
    )
    
    # Loss plot
    fig.add_trace(
        go.Scatter(
            x=list(range(1, len(history.history['loss']) + 1)),
            y=history.history['loss'],
            mode='lines+markers',
            name='Training Loss',
            line=dict(color='blue'),
            showlegend=False
        ),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Scatter(
            x=list(range(1, len(history.history['val_loss']) + 1)),
            y=history.history['val_loss'],
            mode='lines+markers',
            name='Validation Loss',
            line=dict(color='red'),
            showlegend=False
        ),
        row=1, col=2
    )
    
    fig.update_layout(height=400, title_text="Training History")
    fig.update_xaxes(title_text="Epoch")
    fig.update_yaxes(title_text="Accuracy", row=1, col=1)
    fig.update_yaxes(title_text="Loss", row=1, col=2)
    
    return fig

def plot_confusion_matrix(cm, emotions):
    """Plot confusion matrix"""
    fig = px.imshow(
        cm,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=emotions,
        y=emotions,
        color_continuous_scale='Blues',
        title="Confusion Matrix"
    )
    
    # Add text annotations
    for i in range(len(emotions)):
        for j in range(len(emotions)):
            fig.add_annotation(
                x=j, y=i,
                text=str(cm[i][j]),
                showarrow=False,
                font=dict(color="white" if cm[i][j] > cm.max()/2 else "black")
            )
    
    return fig

def plot_emotion_distribution(emotions):
    """Plot emotion distribution"""
    emotion_counts = Counter(emotions)
    
    fig = px.bar(
        x=list(emotion_counts.keys()),
        y=list(emotion_counts.values()),
        title="Emotion Distribution in Dataset",
        labels={'x': 'Emotions', 'y': 'Count'},
        color=list(emotion_counts.values()),
        color_continuous_scale='viridis'
    )
    
    return fig

def main():
    # Header
    st.markdown('<h1 class="main-header">🤖 Emotion-in-Text Analyzer</h1>', unsafe_allow_html=True)
    st.markdown("### Deep Learning powered emotion detection from text using LSTM networks")
    
    # Initialize session state
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = EmotionAnalyzer()
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False
    if 'training_history' not in st.session_state:
        st.session_state.training_history = None
    
    # Sidebar
    st.sidebar.title("🎛️ Model Controls")
    
    # File upload section
    st.sidebar.subheader("📁 Data Files")
    
    # Check if files exist
    train_exists = os.path.exists('train.txt')
    test_exists = os.path.exists('test.txt')
    val_exists = os.path.exists('val.txt')
    
    col1, col2, col3 = st.sidebar.columns(3)
    with col1:
        st.write("train.txt" + (" ✅" if train_exists else " ❌"))
    with col2:
        st.write("test.txt" + (" ✅" if test_exists else " ❌"))
    with col3:
        st.write("val.txt" + (" ✅" if val_exists else " ❌"))
    
    if not all([train_exists, test_exists, val_exists]):
        st.sidebar.error("Please ensure train.txt, test.txt, and val.txt files are in the same directory as this script.")
        st.error("Required files missing! Please add train.txt, test.txt, and val.txt to the project directory.")
        st.info("""
        **File format expected:**
        ```
        text content here;emotion_label
        another text sample;joy
        feeling sad today;sadness
        ```
        """)
        return
    
    # Model parameters
    st.sidebar.subheader("🔧 Model Parameters")
    epochs = st.sidebar.slider("Training Epochs", 10, 100, 50)
    batch_size = st.sidebar.selectbox("Batch Size", [16, 32, 64, 128], index=1)
    
    # Training section
    st.sidebar.subheader("🚀 Model Training")
    
    if st.sidebar.button("🎯 Train Model", type="primary"):
        with st.spinner("Loading and preprocessing data..."):
            # Load data
            analyzer = st.session_state.analyzer
            data = analyzer.prepare_data('train.txt', 'test.txt', 'val.txt')
            
            if data[0] is None:
                st.error("Failed to load data. Please check your files.")
                return
            
            train_data, test_data, val_data = data
            
            # Display data info
            st.success("✅ Data loaded successfully!")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Training Samples", len(train_data[0]))
            with col2:
                st.metric("Test Samples", len(test_data[0]))
            with col3:
                st.metric("Validation Samples", len(val_data[0]))
            
            # Show emotion distribution
            train_texts, train_emotions = analyzer.load_data('train.txt')
            fig_dist = plot_emotion_distribution(train_emotions)
            st.plotly_chart(fig_dist, use_container_width=True)
            
        with st.spinner("Building and training model..."):
            # Build model
            analyzer.build_model()
            
            # Display model architecture
            st.subheader("🏗️ Model Architecture")
            model_summary = []
            analyzer.model.summary(print_fn=lambda x: model_summary.append(x))
            st.text('\n'.join(model_summary))
            
            # Train model
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Custom callback to update progress
            class StreamlitCallback(tf.keras.callbacks.Callback):
                def __init__(self, epochs, progress_bar, status_text):
                    self.epochs = epochs
                    self.progress_bar = progress_bar
                    self.status_text = status_text
                
                def on_epoch_end(self, epoch, logs=None):
                    progress = (epoch + 1) / self.epochs
                    self.progress_bar.progress(progress)
                    self.status_text.text(f'Epoch {epoch+1}/{self.epochs} - Loss: {logs["loss"]:.4f} - Accuracy: {logs["accuracy"]:.4f}')
            
            callback = StreamlitCallback(epochs, progress_bar, status_text)
            
            history = analyzer.train_model(
                train_data, val_data, 
                epochs=epochs, 
                batch_size=batch_size
            )
            
            st.session_state.training_history = history
            st.session_state.model_trained = True
            
            progress_bar.progress(1.0)
            status_text.text("✅ Training completed!")
            
        with st.spinner("Evaluating model..."):
            # Evaluate model
            report, cm, predictions = analyzer.evaluate_model(test_data)
            
            # Display results
            st.success("🎉 Model training completed successfully!")
            
            # Training history plots
            st.subheader("📈 Training History")
            fig_history = plot_training_history(history)
            st.plotly_chart(fig_history, use_container_width=True)
            
            # Model performance
            st.subheader("📊 Model Performance")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Test Accuracy", f"{report['accuracy']:.3f}")
            with col2:
                st.metric("Macro Avg F1", f"{report['macro avg']['f1-score']:.3f}")
            with col3:
                st.metric("Weighted Avg F1", f"{report['weighted avg']['f1-score']:.3f}")
            
            # Confusion Matrix
            fig_cm = plot_confusion_matrix(cm, analyzer.emotions)
            st.plotly_chart(fig_cm, use_container_width=True)
            
            # Classification Report
            st.subheader("📋 Detailed Classification Report")
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.round(3))
            
            # Save model
            if analyzer.save_model():
                st.success("💾 Model saved successfully!")
    
    # Load existing model
    if st.sidebar.button("📂 Load Saved Model"):
        analyzer = st.session_state.analyzer
        if analyzer.load_model():
            st.session_state.model_trained = True
            st.sidebar.success("✅ Model loaded successfully!")
        else:
            st.sidebar.error("❌ No saved model found!")
    
    # Main content area
    if st.session_state.model_trained:
        st.header("🔮 Emotion Prediction")
        
        # Text input methods
        input_method = st.radio(
            "Choose input method:",
            ["Type text", "Upload text file", "Try examples"]
        )
        
        user_text = ""
        
        if input_method == "Type text":
            user_text = st.text_area(
                "Enter text to analyze:",
                height=100,
                placeholder="Type your text here... e.g., 'I am feeling really excited about this new project!'"
            )
        
        elif input_method == "Upload text file":
            uploaded_file = st.file_uploader("Choose a text file", type=['txt'])
            if uploaded_file is not None:
                user_text = str(uploaded_file.read(), "utf-8")
                st.text_area("Uploaded text:", value=user_text, height=100)
        
        elif input_method == "Try examples":
            examples = [
                "I am feeling really happy and excited about my new job!",
                "This makes me so angry and frustrated with everything.",
                "I feel really sad and lonely today without my friends.",
                "I am scared about what might happen in the future.",
                "I love spending time with my family during holidays.",
                "Wow, I never expected this surprise party for me!"
            ]
            selected_example = st.selectbox("Select an example:", examples)
            user_text = selected_example
        
        # Prediction
        if st.button("🎯 Analyze Emotion", type="primary") and user_text.strip():
            with st.spinner("Analyzing emotion..."):
                results = st.session_state.analyzer.predict_emotion(user_text)
                
                if results:
                    # Main prediction
                    top_emotion = results[0]
                    
                    st.markdown(f"""
                    <div class="emotion-card">
                        <h2>{top_emotion['emoji']} {top_emotion['emotion'].upper()}</h2>
                        <h3>Confidence: {top_emotion['confidence']:.1%}</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Detailed results
                    st.subheader("📊 Detailed Analysis")
                    
                    # Create confidence chart
                    emotions = [r['emotion'] for r in results]
                    confidences = [r['confidence'] for r in results]
                    emojis = [r['emoji'] for r in results]
                    
                    fig = go.Figure(data=[
                        go.Bar(
                            x=confidences,
                            y=[f"{emoji} {emotion}" for emoji, emotion in zip(emojis, emotions)],
                            orientation='h',
                            marker=dict(
                                color=confidences,
                                colorscale='viridis',
                                showscale=True
                            ),
                            text=[f"{conf:.1%}" for conf in confidences],
                            textposition='inside'
                        )
                    ])
                    
                    fig.update_layout(
                        title="Emotion Confidence Levels",
                        xaxis_title="Confidence",
                        yaxis_title="Emotions",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Results table
                    results_df = pd.DataFrame(results)
                    results_df['confidence'] = results_df['confidence'].apply(lambda x: f"{x:.1%}")
                    st.dataframe(results_df, use_container_width=True)
        
        elif user_text.strip() == "":
            st.warning("⚠️ Please enter some text to analyze!")
    
    else:
        st.info("👆 Please train or load a model first using the sidebar controls.")
        
        # Show instructions
        st.subheader("📋 Instructions")
        st.markdown("""
        1. **Prepare your data files:**
           - `train.txt`: Main training data
           - `test.txt`: Test data for evaluation  
           - `val.txt`: Validation data
           
        2. **File format:** Each line should be: `text;emotion_label`
           - Example: `i feel really happy today;joy`
           
        3. **Train the model:** Click "Train Model" in the sidebar
        
        4. **Analyze emotions:** Once trained, enter text to get predictions
        
        5. **Save/Load:** Models are automatically saved and can be reloaded
        """)
        
        # Show sample data format
        st.subheader("📄 Expected Data Format")
        st.code("""
i feel really happy and excited;joy
this makes me so sad and depressed;sadness  
i am angry about this situation;anger
i feel scared about the future;fear
i love spending time with family;love
wow that was totally unexpected;surprise
        """)

if __name__ == "__main__":
    main()