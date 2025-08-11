import cv2

from insightface.app import FaceAnalysis

# Initialize model
model = FaceAnalysis(name="buffalo_l")
model.prepare(ctx_id=0)

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Run face analysis (age prediction) on current frame
    faces = model.get(frame)

    # Draw predicted age on the frame for each detected face
    for face in faces:
        box = face.bbox.astype(int)
        age = face.age
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        cv2.putText(frame, f"Age: {int(age)}", (box[0], box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Age Estimation (Press Q to quit)", frame)

    # Quit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()