from flask import Flask, render_template, request, jsonify
import cv2
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/capture', methods=['POST'])
def capture():
    subject_name = request.form['subject_name']
    num_photos = int(request.form['num_photos'])
    
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("Capture Photos")

    angles = ["straight", "left", "right", "up", "down"]
    photos_taken = 0

    try:
        # Create a directory for the subject if it doesn't exist
        os.makedirs(f'faces/{subject_name}', exist_ok=True)

        while photos_taken < num_photos:
            ret, frame = cam.read()
            if not ret:
                print("Failed to grab frame")
                break
            cv2.imshow("Capture Photos", frame)

            angle = angles[photos_taken % len(angles)]
            instruction = f"Look {angle}. Press space to capture."
            print(instruction)
            k = cv2.waitKey(1)

            if k % 256 == 27:
                # ESC pressed
                print("Escape hit, closing...")
                break
            elif k % 256 == 32:
                # SPACE pressed
                img_name = f"faces/{subject_name}/{subject_name}_{angle}_{photos_taken}.png"
                cv2.imwrite(img_name, frame)
                print(f"{img_name} written!")
                photos_taken += 1

    finally:
        cam.release()
        cv2.destroyAllWindows()

    return jsonify({"message": "Photos captured successfully!"})

if __name__ == "__main__":
    app.run(debug=True)
