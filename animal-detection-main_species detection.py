# ============================================
# ✅ 1. INSTALL REQUIRED LIBRARIES
# ============================================
!pip install ultralytics opencv-python-headless matplotlib


# ============================================
# ✅ 2. USER UPLOADS IMAGE
# ============================================
from google.colab import files
uploaded = files.upload()


# ============================================
# ✅ 3. IMPORT LIBRARIES
# ============================================
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt


# ============================================
# ✅ 4. LOAD MOST POWERFUL YOLO MODEL
# ============================================
model = YOLO("yolov8x.pt")   # Highest accuracy model


# ============================================
# ✅ 5. READ USER IMAGE
# ============================================
img_path = list(uploaded.keys())[0]
print("✅ Uploaded Image:", img_path)

img = cv2.imread(img_path)


# ============================================
# ✅ 6. RUN DETECTION WITH LOW THRESHOLD
# ============================================
results = model(img, conf=0.15)   # lower = detect more objects

boxes = results[0].boxes
names = model.names


# ============================================
# ✅ 7. DETECT ALL ANIMAL SPECIES
# ============================================
animal_classes = [
    "cat", "dog", "horse", "cow", "sheep","lion","Tiger",
    "elephant", "bear", "bird", "zebra", "giraffe"
]

detected_animals = []

print("\n✅ ALL ANIMAL SPECIES DETECTED:\n")

for box in boxes:
    class_id = int(box.cls[0])
    confidence = float(box.conf[0])
    object_name = names[class_id]

    if object_name in animal_classes:
        detected_animals.append(object_name)
        print(f"🐾 {object_name} | Confidence: {confidence:.2f}")


# ============================================
# ✅ 8. DRAW BOXES
# ============================================
output_image = results[0].plot()
cv2.imwrite("all_animals_detected.jpg", output_image)


# ============================================
# ✅ 9. DISPLAY IMAGE
# ============================================
plt.figure(figsize=(10,10))
plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()


# ============================================
# ✅ 10. FINAL SUMMARY
# ============================================
print("\n✅ FINAL ANIMAL LIST FOUND:")
if len(detected_animals) > 0:
    print(set(detected_animals))
else:
    print("❌ No animals detected.")

print("\n📁 Output saved as: all_animals_detected.jpg")
