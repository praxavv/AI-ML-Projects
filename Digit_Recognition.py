# Google Colab Compatible Handwritten Digit Recognition
# Run this in Google Colab for an interactive web-based drawing interface

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from IPython.display import HTML, display
import base64
from io import BytesIO
from PIL import Image
import json

class ColabDigitRecognizer:
    def __init__(self):
        """Initialize the digit recognizer"""
        self.model = None
        self.load_or_create_model()
        
    def load_or_create_model(self):
        """Load or create a digit recognition model"""
        print("🚀 Setting up digit recognition model...")
        
        # Load MNIST dataset
        print("📚 Loading MNIST dataset...")
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        
        # Normalize the data
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        
        # Reshape data
        x_train = x_train.reshape(-1, 28, 28, 1)
        x_test = x_test.reshape(-1, 28, 28, 1)
        
        # Convert labels to categorical
        y_train = keras.utils.to_categorical(y_train, 10)
        y_test = keras.utils.to_categorical(y_test, 10)
        
        print("🔧 Building neural network...")
        # Create optimized CNN model
        self.model = keras.Sequential([
            keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.Flatten(),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(10, activation='softmax')
        ])
        
        # Compile model
        self.model.compile(optimizer='adam',
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
        
        print("🎯 Training model (this will take a few minutes)...")
        # Train model
        history = self.model.fit(x_train, y_train, 
                               epochs=3,  # Reduced for faster training in Colab
                               batch_size=128, 
                               validation_split=0.1,
                               verbose=1)
        
        # Evaluate
        test_loss, test_acc = self.model.evaluate(x_test, y_test, verbose=0)
        print(f"✅ Model ready! Test accuracy: {test_acc:.4f}")
        
    def predict_from_canvas(self, image_data):
        """Predict digit from canvas image data"""
        try:
            # Decode base64 image
            image_data = image_data.split(',')[1]  # Remove data:image/png;base64,
            image_bytes = base64.b64decode(image_data)
            
            # Convert to PIL Image
            image = Image.open(BytesIO(image_bytes))
            
            # Convert to grayscale and resize
            image = image.convert('L')
            image = image.resize((28, 28), Image.Resampling.LANCZOS)
            
            # Convert to numpy array
            img_array = np.array(image)
            
            # Invert colors (canvas is black on white, MNIST is white on black)
            img_array = 255 - img_array
            
            # Normalize
            img_array = img_array.astype('float32') / 255.0
            
            # Reshape for model
            img_array = img_array.reshape(1, 28, 28, 1)
            
            # Make prediction
            prediction = self.model.predict(img_array, verbose=0)
            predicted_digit = np.argmax(prediction)
            confidence = np.max(prediction)
            
            return {
                'digit': int(predicted_digit),
                'confidence': float(confidence),
                'all_predictions': prediction[0].tolist()
            }
            
        except Exception as e:
            return {'error': str(e)}

# Initialize the recognizer
print("Initializing Handwritten Digit Recognition for Google Colab...")
recognizer = ColabDigitRecognizer()

# JavaScript function to handle predictions
prediction_js = """
<script>
function makePrediction() {
    const canvas = document.getElementById('drawingCanvas');
    const ctx = canvas.getContext('2d');
    
    // Check if canvas is empty
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const pixels = imageData.data;
    let isEmpty = true;
    
    for (let i = 0; i < pixels.length; i += 4) {
        if (pixels[i] < 255 || pixels[i+1] < 255 || pixels[i+2] < 255) {
            isEmpty = false;
            break;
        }
    }
    
    if (isEmpty) {
        document.getElementById('result').innerHTML = '✏️ Please draw a digit first!';
        document.getElementById('confidence').innerHTML = '';
        return;
    }
    
    // Get image data
    const imageDataURL = canvas.toDataURL('image/png');
    
    // Show loading
    document.getElementById('result').innerHTML = '🔄 Predicting...';
    document.getElementById('confidence').innerHTML = '';
    
    // Send to Python backend
    const prediction_data = {
        'image_data': imageDataURL
    };
    
    // This will be handled by the Python callback
    google.colab.kernel.invokeFunction('predict_digit', [imageDataURL], {});
}

function clearCanvas() {
    const canvas = document.getElementById('drawingCanvas');
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    document.getElementById('result').innerHTML = '✏️ Canvas cleared! Draw a digit (0-9)';
    document.getElementById('confidence').innerHTML = '';
    document.getElementById('allPredictions').innerHTML = '';
}

function updateResult(digit, confidence, allPredictions) {
    document.getElementById('result').innerHTML = `🎯 Predicted Digit: <strong>${digit}</strong>`;
    document.getElementById('confidence').innerHTML = `Confidence: ${(confidence * 100).toFixed(1)}%`;
    
    // Show all predictions as a bar chart
    let predictionHTML = '<div style="margin-top: 15px;"><strong>All Predictions:</strong><br>';
    for (let i = 0; i < allPredictions.length; i++) {
        const percentage = (allPredictions[i] * 100).toFixed(1);
        const barWidth = Math.max(percentage * 2, 2);
        predictionHTML += `
            <div style="margin: 2px 0; display: flex; align-items: center;">
                <span style="width: 20px; text-align: right; margin-right: 10px;">${i}:</span>
                <div style="background: linear-gradient(90deg, #3498db, #2980b9); width: ${barWidth}px; height: 18px; margin-right: 10px;"></div>
                <span style="font-size: 12px;">${percentage}%</span>
            </div>
        `;
    }
    predictionHTML += '</div>';
    document.getElementById('allPredictions').innerHTML = predictionHTML;
}

// Drawing functionality
let isDrawing = false;
let lastX = 0;
let lastY = 0;

function setupCanvas() {
    const canvas = document.getElementById('drawingCanvas');
    const ctx = canvas.getContext('2d');
    
    // Set canvas properties
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.lineWidth = 15;
    ctx.strokeStyle = 'black';
    
    // Clear canvas
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Mouse events
    canvas.addEventListener('mousedown', startDrawing);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseup', stopDrawing);
    canvas.addEventListener('mouseout', stopDrawing);
    
    // Touch events for mobile
    canvas.addEventListener('touchstart', handleTouch);
    canvas.addEventListener('touchmove', handleTouch);
    canvas.addEventListener('touchend', stopDrawing);
    
    function startDrawing(e) {
        isDrawing = true;
        [lastX, lastY] = getMousePos(canvas, e);
    }
    
    function draw(e) {
        if (!isDrawing) return;
        
        const [currentX, currentY] = getMousePos(canvas, e);
        
        ctx.beginPath();
        ctx.moveTo(lastX, lastY);
        ctx.lineTo(currentX, currentY);
        ctx.stroke();
        
        [lastX, lastY] = [currentX, currentY];
    }
    
    function stopDrawing() {
        isDrawing = false;
    }
    
    function handleTouch(e) {
        e.preventDefault();
        const touch = e.touches[0];
        const mouseEvent = new MouseEvent(e.type === 'touchstart' ? 'mousedown' : 
                                         e.type === 'touchmove' ? 'mousemove' : 'mouseup', {
            clientX: touch.clientX,
            clientY: touch.clientY
        });
        canvas.dispatchEvent(mouseEvent);
    }
    
    function getMousePos(canvas, e) {
        const rect = canvas.getBoundingClientRect();
        return [e.clientX - rect.left, e.clientY - rect.top];
    }
}

// Initialize canvas when page loads
setTimeout(setupCanvas, 100);
</script>
"""

# HTML interface
html_interface = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Handwritten Digit Recognition</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            min-height: 100vh;
        }}
        
        .container {{
            background: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }}
        
        h1 {{
            text-align: center;
            color: #2c3e50;
            margin-bottom: 10px;
            font-size: 28px;
        }}
        
        .subtitle {{
            text-align: center;
            color: #7f8c8d;
            margin-bottom: 30px;
            font-size: 16px;
        }}
        
        .canvas-container {{
            text-align: center;
            margin: 30px 0;
        }}
        
        #drawingCanvas {{
            border: 3px solid #3498db;
            border-radius: 10px;
            cursor: crosshair;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            background: white;
        }}
        
        .buttons {{
            text-align: center;
            margin: 25px 0;
        }}
        
        button {{
            font-size: 16px;
            font-weight: bold;
            padding: 12px 25px;
            margin: 0 10px;
            border: none;
            border-radius: 25px;
            cursor: pointer;
            transition: all 0.3s ease;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        .predict-btn {{
            background: linear-gradient(45deg, #3498db, #2980b9);
            color: white;
        }}
        
        .predict-btn:hover {{
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(52, 152, 219, 0.4);
        }}
        
        .clear-btn {{
            background: linear-gradient(45deg, #e74c3c, #c0392b);
            color: white;
        }}
        
        .clear-btn:hover {{
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(231, 76, 60, 0.4);
        }}
        
        .results {{
            background: linear-gradient(135deg, #f8f9fa, #e9ecef);
            border-radius: 15px;
            padding: 25px;
            margin: 25px 0;
            text-align: center;
            min-height: 80px;
        }}
        
        #result {{
            font-size: 24px;
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 10px;
        }}
        
        #confidence {{
            font-size: 18px;
            color: #27ae60;
        }}
        
        #allPredictions {{
            text-align: left;
            max-width: 400px;
            margin: 0 auto;
        }}
        
        .tips {{
            background: linear-gradient(135deg, #74b9ff, #0984e3);
            color: white;
            border-radius: 10px;
            padding: 20px;
            margin-top: 25px;
        }}
        
        .tips h3 {{
            margin-top: 0;
            color: white;
        }}
        
        .tips ul {{
            margin: 0;
            padding-left: 20px;
        }}
        
        .tips li {{
            margin: 8px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>✍️ Handwritten Digit Recognition</h1>
        <div class="subtitle">Draw a digit (0-9) and let AI predict what you wrote!</div>
        
        <div class="canvas-container">
            <canvas id="drawingCanvas" width="400" height="400"></canvas>
        </div>
        
        <div class="buttons">
            <button class="predict-btn" onclick="makePrediction()">🔮 Predict Digit</button>
            <button class="clear-btn" onclick="clearCanvas()">🗑️ Clear Canvas</button>
        </div>
        
        <div class="results">
            <div id="result">✏️ Draw a digit to get started!</div>
            <div id="confidence"></div>
            <div id="allPredictions"></div>
        </div>
        
        <div class="tips">
            <h3>💡 Tips for Better Recognition:</h3>
            <ul>
                <li>Draw digits large and clear</li>
                <li>Use the full canvas area</li>
                <li>Write digits as you normally would</li>
                <li>Make sure lines are thick and connected</li>
                <li>Try redrawing if confidence is low</li>
            </ul>
        </div>
    </div>
    
    {prediction_js}
</body>
</html>
"""

# Colab callback function
def predict_digit(image_data):
    """Callback function for Colab to handle predictions"""
    result = recognizer.predict_from_canvas(image_data)
    
    if 'error' in result:
        js_code = f"""
        document.getElementById('result').innerHTML = '❌ Error: {result["error"]}';
        document.getElementById('confidence').innerHTML = '';
        document.getElementById('allPredictions').innerHTML = '';
        """
    else:
        js_code = f"""
        updateResult({result['digit']}, {result['confidence']}, {result['all_predictions']});
        """
    
    display(HTML(f"<script>{js_code}</script>"))

# Register the callback
try:
    from google.colab import output
    output.register_callback('predict_digit', predict_digit)
    print("✅ Callback registered successfully!")
except ImportError:
    print("⚠️ Running outside Google Colab - callback registration skipped")

# Display the interface
print("\n" + "="*60)
print("🎨 INTERACTIVE DIGIT RECOGNITION INTERFACE")
print("="*60)
print("The web interface is loading below...")
print("Draw digits with your mouse or finger and click 'Predict'!")
print("="*60 + "\n")

display(HTML(html_interface))
