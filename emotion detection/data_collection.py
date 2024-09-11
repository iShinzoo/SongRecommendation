import mediapipe as mp 
import numpy as np  
import cv2  


cap = cv2.VideoCapture(0)


name = input("Enter the name of the data: ")

# Initialize MediaPipe holistic and hands solutions
holistic = mp.solutions.holistic
hands = mp.solutions.hands
holis = holistic.Holistic()  # Create a Holistic object for processing
drawing = mp.solutions.drawing_utils  # Utility for drawing landmarks on the frame

# Initialize an empty list to store the landmark data and set the initial data size to 0
X = []
data_size = 0

# Start an infinite loop to capture video frames and process them
while True:
    lst = []  # Initialize an empty list to store landmarks of the current frame

    # Capture a frame from the video feed
    _, frm = cap.read()

    # Flip the frame horizontally (mirror effect)
    frm = cv2.flip(frm, 1)

    # Process the frame to detect face and hand landmarks
    res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

    # Check if face landmarks are detected
    if res.face_landmarks:
        # Normalize face landmarks relative to the position of landmark[1] (typically the nose)
        for i in res.face_landmarks.landmark:
            lst.append(i.x - res.face_landmarks.landmark[1].x)
            lst.append(i.y - res.face_landmarks.landmark[1].y)

        # Check if left hand landmarks are detected
        if res.left_hand_landmarks:
            # Normalize left hand landmarks relative to the position of landmark[8] (index finger tip)
            for i in res.left_hand_landmarks.landmark:
                lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
                lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
        else:
            # If left hand landmarks are not detected, append zeros
            for i in range(42):
                lst.append(0.0)

        # Check if right hand landmarks are detected
        if res.right_hand_landmarks:
            # Normalize right hand landmarks relative to the position of landmark[8] (index finger tip)
            for i in res.right_hand_landmarks.landmark:
                lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
        else:
            # If right hand landmarks are not detected, append zeros
            for i in range(42):
                lst.append(0.0)

        # Append the landmarks of the current frame to the data list
        X.append(lst)
        data_size += 1  # Increment the data size counter

    # Draw the detected landmarks on the frame
    drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_CONTOURS)
    drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
    drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)

    # Display the current data size on the video frame
    cv2.putText(frm, str(data_size), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the processed video frame in a window
    cv2.imshow("window", frm)

    # Check if the 'Esc' key is pressed or if 100 frames have been captured
    if cv2.waitKey(1) == 27 or data_size > 99:
        cv2.destroyAllWindows()  # Close the video window
        cap.release()  # Release the video capture object
        break  # Exit the loop

# Save the collected landmark data as a NumPy array
np.save(f"{name}.npy", np.array(X))

# Print the shape of the saved data array
print(np.array(X).shape)
