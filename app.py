from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import base64
from ultralytics import YOLO
import cv2
import numpy as np
from io import BytesIO
from PIL import Image

# Load the YOLO model
banuka_model = YOLO("banuka.pt")
nimsara_skin_model = YOLO("nimsara_skin_model.pt")
nimsara_category_model = YOLO("nimsara_category_model.pt")

def numpy_to_base64(image_np, format="PNG"):
    """Convert NumPy image array to base64-encoded string."""
    image_pil = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
    buffered = BytesIO()
    image_pil.save(buffered, format=format)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


app = Flask(__name__)
CORS(app)


# get disease name 
@app.route('/get_disease', methods=['POST'])
def get_disease():
    data = request.get_json()

    if 'image' not in data or 'filename' not in data:
        return jsonify({"error": "Missing image or filename"}), 400

    try:
        image_data = base64.b64decode(data['image'])
        file_path = os.path.join(UPLOAD_FOLDER, data['filename'])


        with open(file_path, "wb") as f:
            f.write(image_data)

        # Perform YOLO prediction
        results = banuka_model.predict(source=file_path, conf=0.7, show=False, save=False, verbose=False)
        result = results[0]  

        
        pred_image = result.plot()

        orig_img = results[0].orig_img.copy()


        if results[0].boxes is not None:
            for box in results[0].boxes.xyxy:  
                x1, y1, x2, y2 = map(int, box) 
                cv2.rectangle(orig_img, (x1, y1), (x2, y2), (0, 255, 0), 3)

        base64_str = numpy_to_base64(orig_img)

        
        class_names = []
        if result.boxes is not None and len(result.boxes.cls) > 0:
            class_ids = result.boxes.cls.tolist()  
            class_names = [result.names[int(cls_id)] for cls_id in class_ids]

            class_names = list(set(class_names))

        return jsonify({"image": base64_str, "class_names": class_names})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

#test fish skin 

@app.route('/test_skin', methods=['POST'])
def test_skin():
    data = request.get_json()

    if 'image' not in data or 'filename' not in data:
        return jsonify({"error": "Missing image or filename"}), 400

    try:
        image_data = base64.b64decode(data['image'])
        file_path = os.path.join(UPLOAD_FOLDER, data['filename'])


        with open(file_path, "wb") as f:
            f.write(image_data)

        # Perform YOLO prediction
        results = nimsara_skin_model.predict(source=file_path, conf=0.7, show=False, save=False, verbose=False)
        result = results[0]  

        
        pred_image = result.plot()

        if isinstance(results, list):

            results = results[0]

        orig_img = results.orig_img
        original= orig_img.copy()
        # original =cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)

        if results.masks is not None:

            print("A sick fish is detected.")

            masks = results.masks.data.cpu().numpy()  
                            
            orig_h, orig_w = results.orig_shape[:2]  


            combined_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)  

            for mask in masks:

                resized_mask = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                combined_mask = np.maximum(combined_mask, resized_mask)  


            mask_colored = np.zeros_like(orig_img, dtype=np.uint8)
            mask_colored[combined_mask > 0] = [0, 255, 0]  


            blended = cv2.addWeighted(orig_img, 0.8, mask_colored, 0.5, 0)
            blended =cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)                
            base64_str = numpy_to_base64(blended)

        else:
            base64_str = numpy_to_base64(original)

        
        class_names = []
        if result.boxes is not None and len(result.boxes.cls) > 0:
            class_ids = results[0].boxes.cls.cpu().numpy()  
            class_names = [results[0].names[int(cls_id)] for cls_id in class_ids]

        else:
            class_names = ["healthy_skin"]

        return jsonify({"image": base64_str, "class_names": class_names[0]})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

#fish category
@app.route('/fish_category', methods=['POST'])
def fish_category():
    data = request.get_json()

    if 'image' not in data or 'filename' not in data:
        return jsonify({"error": "Missing image or filename"}), 400

    try:
        image_data = base64.b64decode(data['image'])
        file_path = os.path.join(UPLOAD_FOLDER, data['filename'])


        with open(file_path, "wb") as f:
            f.write(image_data)

        # Perform YOLO prediction
        results = nimsara_category_model(file_path, save=False, conf=0.5, verbose=False)

        results1 = results[0]

        # If the model didn't return any detections, assume "not_a_fish"


        if isinstance(results, list):

            results = results[0]

            orig_img = results.orig_img
            original =cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
            
            top1_index = results.probs.top1
            confidence_score = results.probs.top1conf
            # print(confidence_score) 
            CONFIDENCE_THRESHOLD =0.7

            if confidence_score < CONFIDENCE_THRESHOLD:
                return jsonify({"class_name": "not_a_fish"})            

            
            predicted_class_name = results.names[top1_index]        

        return jsonify({"class_names": predicted_class_name})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    







if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
