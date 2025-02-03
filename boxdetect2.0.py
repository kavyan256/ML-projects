import cv2
import numpy as np
import tensorflow as tf  # Load the trained MNIST model

# Load pre-trained MNIST model
model = tf.keras.models.load_model("mnist_model.keras")  # Replace with your model path

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_green = np.array([30, 50, 50])
    upper_green = np.array([80, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    green_regions = cv2.bitwise_and(frame, frame, mask=mask)
    cv2.imshow("Green Regions", green_regions)
    gray = cv2.cvtColor(green_regions, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 2)
    eroded = cv2.erode(blurred, (4, 4), iterations=3)

    contours, _ = cv2.findContours(eroded, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    print(f'{len(contours)} contours found')

    for contour in contours:
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) == 4:
            area = cv2.contourArea(contour)
            if area > 1000:
                x, y, w, h = cv2.boundingRect(approx)
                
                 # Shrink the bounding box by scaling down (e.g., reduce by 20%)
                padding = 0.25  # Shrink factor, adjust as necessary
                new_w = int(w * (1 - padding ))
                new_h = int(h)

                # Adjust x and y to keep the smaller ROI centered
                x_centered = x + (w - new_w) // 2
                y_centered = y + (h - new_h) // 2

                # Define the smaller ROI (region of interest)
                roi = frame[y_centered:y_centered+new_h, x_centered:x_centered+new_w]

                if roi.shape[0] > 0 and roi.shape[1] > 0:
                    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
                    resized_gray_roi = cv2.resize(gray_roi, (28, 28))  # Resize to 28x28

                    # **Preprocess for MNIST Model**
                    normalized_roi = resized_gray_roi / 255.0  # Normalize to [0,1]

                    

                    # **Show Processed Input in a Separate Window**
                    cv2.imshow("Processed Input to Model", resized_gray_roi)  # Show 28x28 input

                    # Flatten the image to (1, 784)
                    flattened_roi = normalized_roi.flatten().reshape(1, 784)

                    # **Make Prediction**
                    prediction = model.predict(flattened_roi)
                    digit = np.argmax(prediction)  # Get the most probable digit

                    # **Display the predicted digit on the frame**
                    cv2.putText(frame, f"Pred: {digit}", (x, y - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Green Rectangles", frame)

    # Press 'q' to quit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
