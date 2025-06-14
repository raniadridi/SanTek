# 🎭 Face Mask Detection - Real-Time AI Web App

A modern Flask web application that detects face masks in real-time using MobileNetV2 deep learning model with live webcam feed.

## 📊 Features

- **🎯 Real-Time Detection**: Instant face mask detection using webcam with high accuracy
- **🧠 AI Powered**: Built with MobileNetV2 deep learning architecture using Transfer Learning for optimal performance  
- **💻 Modern UI**: Responsive design with beautiful glassmorphism effects
- **📊 Live Analytics**: Real-time statistics and confidence scores
- **🔒 Error Handling**: Robust error handling with user-friendly messages
- **⚡ Loading States**: Visual feedback during detection process
- **🎯 Multi-Class Detection**: Detects "With Mask" and "Without Mask" states
- **📈 Mobile Friendly**: Fully responsive design for all devices

## 🏗️ Project Structure

```
face-mask-detection/
├── app.py                      # Main Flask application
├── mask_detection_script.py    # AI detection logic
├── model_training.ipynb        # Model Training in the notebook
├── mask_detector.h5            # Trained MobileNetV2 model
├── requirements.txt            # Python dependencies
├── templates/                  # HTML templates
│   ├── index.html             # Landing page
│   ├── face_detection.html    # Detection interface
│   └── statistics.html        # Analytics dashboard
├── static/                     # Static assets
│   ├── css/                   # Stylesheets
│   ├── js/                    # JavaScript files
│   └── images/                # Image assets
└── README.md                  # This file
```

## 🚀 Quick Start

### Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/face-mask-detection.git
   cd face-mask-detection
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Ensure model file exists**
   - Make sure `mask_detector.h5` is in the project root
   - The model should be trained using MobileNetV2 architecture

5. **Run the application**
   ```bash
   python app.py
   ```

6. **Open your browser**
   ```
   http://localhost:5000
   ```

## 🔬 Usage

### Real-Time Detection

- Navigate to the Face Detection page
- Allow camera access when prompted
- Position yourself in front of the camera
- View real-time mask detection with confidence scores

### API Endpoints

- `GET /` - Landing page with feature overview
- `GET /facedetection` - Live detection interface
- `GET /statistics` - Analytics and usage statistics
- `GET /video_feed` - Live video stream endpoint

## 📈 Model Performance

### Architecture Details

- **🏗️ Base Model**: MobileNetV2 (pre-trained on ImageNet) with Transfer Learning
- **🎯 Input Size**: 224x224x3 RGB images
- **⚖️ Optimizer**: Adam with learning rate 1e-4
- **📊 Loss Function**: Binary crossentropy
- **🔍 Classes**: 2 (With Mask, Without Mask)
- **🔄 Transfer Learning**: Froze base layers, trained custom classification head

### Training Configuration

- **📚 Epochs**: 20 with early stopping
- **📦 Batch Size**: 32
- **🔄 Data Augmentation**: Rotation, zoom, shift, flip
- **📈 Validation Split**: 20% stratified sampling

## 🛠️ Technology Stack

- **Backend**: Flask (Python)
- **🧠 ML/AI**: TensorFlow, Keras, MobileNetV2
- **👁️ Computer Vision**: OpenCV, Haar Cascades
- **Frontend**: HTML5, CSS3, JavaScript
- **🎨 Styling**: Custom CSS with glassmorphism effects
- **📱 Responsive**: Mobile-first design approach
- **📊 Version Control**: Git, GitHub

## 📚 Detection Tips

### Best Practices

- **💡 Good Lighting**: Ensure adequate lighting for better detection
- **📐 Face Position**: Keep face clearly visible and centered
- **📏 Distance**: Maintain 1-3 feet distance from camera
- **🎯 Angle**: Face camera directly for optimal results

### Performance Optimization

- **⚡ Model Size**: Optimized MobileNetV2 for fast inference
- **🔄 Frame Processing**: Efficient OpenCV operations
- **📊 Batch Processing**: Multiple face detection in single frame
- **🎯 Confidence Threshold**: Adjustable detection sensitivity
- **🧠 Transfer Learning**: Leveraged pre-trained ImageNet features for faster training and better accuracy

## 📚 Error Handling

The application includes comprehensive error handling for:

- **📷 Camera Access**: Graceful fallback when camera unavailable
- **🤖 Model Loading**: Error recovery and user notifications
- **🔍 Detection Failures**: Robust exception handling
- **🌐 Network Issues**: Timeout and connectivity management

## 📞 Dependencies

- **Flask**: Web framework for Python
- **TensorFlow**: Deep learning framework
- **OpenCV**: Computer vision library
- **NumPy**: Numerical computing
- **Pillow**: Image processing

## 🙏 Acknowledgments

- **🏗️ MobileNetV2**: Google's efficient CNN architecture
- **👁️ OpenCV**: Computer vision capabilities
- **🧠 TensorFlow**: Deep learning framework
- **🌐 Flask**: Lightweight web framework
- **📚 Transfer Learning**: Pre-trained ImageNet features for enhanced performance

## 📞 Contact

- **👨‍💻 Developer**: Rania Dridi
- **📧 Email**: raniadridi42@gmail.com
- **🔗 LinkedIn**: [Rania Dridi](https://linkedin.com/in/raniadridii)
- **🐙 GitHub**: [Rania Dridi](https://github.com/raniadridi)

---

**⭐ Star this repository if you found it helpful!**

Made with ❤️ for AI safety and computer vision enthusiasts