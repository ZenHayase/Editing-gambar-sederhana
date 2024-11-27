from flask import Flask, render_template, request, send_file
import numpy as np
from PIL import Image
import io
import base64

app = Flask(__name__)

def to_grayscale(image):
   
    img_array = np.array(image)
    
   
    grayscale = np.dot(img_array[...,:3], [0.299, 0.587, 0.114])
    
    return Image.fromarray(grayscale.astype(np.uint8))

def apply_blur(image):
  
    img_array = np.array(image)
    
   
    kernel_size = 5
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)
    
  
    height, width = img_array.shape[:2]
    pad = kernel_size // 2
    
   
    output = np.zeros_like(img_array)
    
   
    for y in range(pad, height-pad):
        for x in range(pad, width-pad):
            if len(img_array.shape) == 3:  
                for c in range(3):  
                    window = img_array[y-pad:y+pad+1, x-pad:x+pad+1, c]
                    output[y,x,c] = np.sum(window * kernel)
            else:  
                window = img_array[y-pad:y+pad+1, x-pad:x+pad+1]
                output[y,x] = np.sum(window * kernel)
    
    return Image.fromarray(output.astype(np.uint8))

def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/convert', methods=['POST'])
def convert():
    if 'image' not in request.files:
        return 'No image uploaded', 400
    
    image = request.files['image']
    transform_type = request.form['transform']
    
    
    img = Image.open(image)
    
   
    original_b64 = image_to_base64(img)
    
   
    if transform_type == 'grayscale':
        processed = to_grayscale(img)
    else:  
        processed = apply_blur(img)
    
    
    processed_b64 = image_to_base64(processed)
    
    return render_template('result.html', 
                         original_image=original_b64,
                         processed_image=processed_b64)

if __name__ == '__main__':
    app.run(debug=True)