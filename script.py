import cv2
from insightface.app import FaceAnalysis
import matplotlib.pyplot as plt

# Load the model
model = FaceAnalysis(name="buffalo_l")
model.prepare(ctx_id=0)

# Open webcam
cap = cv2.VideoCapture(0)
print("Press SPACE to capture an image.")

img = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    cv2.imshow("Press SPACE to capture", frame)

    key = cv2.waitKey(1)
    if key % 256 == 32:  # SPACE pressed
        img = frame.copy()
        break

cap.release()
cv2.destroyAllWindows()

# Convert to RGB for processing
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Analyze face
faces = model.get(img)

for face in faces:
    age = int(face.age)
    box = face.bbox.astype(int)
    cv2.rectangle(img_rgb, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
    cv2.putText(img_rgb, f"Age: {age}", (box[0], box[1]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    print(f"Predicted Age: {age}")


plt.imshow(img_rgb)
plt.axis("off")
plt.title("Predicted Age")
plt.show()

