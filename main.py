# SETUP
import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


###### MAKE DETECTIONS 
"""
cap = cv2.VideoCapture(0)
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        # recolour image 
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # changes bgr to rgb
        image.flags.writeable = False; # performance training 

        # make detection 
        results = pose.process(image);

        #recolor back to bgr 
        image.flags.writeable = True;
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR); 

        #render detection 
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness = 2, circle_radius = 2), # gives the specification for like customising how we wanna draw our landmarks 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness = 2, circle_radius = 2)
                                );
        cv2.imshow('Mediapipe Feed', image)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()


###### DETERMINING JOINTS 

cap = cv2.VideoCapture(0)
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        # recolour image 
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # changes bgr to rgb
        image.flags.writeable = False; # performance training 

        # make detection 
        results = pose.process(image);

        #recolor back to bgr 
        image.flags.writeable = True;
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR); 

        #extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark;
            #print(landmarks);
        except:
            pass

        #list all landmarks
        #for lndmark in mp_pose.PoseLandmark:
        #    print(lndmark);

        # example for extracting a landmark
        #landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]

        #render detection 
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness = 2, circle_radius = 2), # gives the specification for like customising how we wanna draw our landmarks 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness = 2, circle_radius = 2)
                                );
        cv2.imshow('Mediapipe Feed', image)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
"""
###### CURL COUNTER

cap = cv2.VideoCapture(0)

# Curl counter variables
counter = 0 
stage = None

## Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        
        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
      
        # Make detection
        results = pose.process(image)
    
        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark
            
            # Get coordinates
            shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            
            # Calculate angle
            angle = Calculate_angle(shoulder, elbow, wrist)
            
            # Visualize angle
            cv2.putText(image, str(angle), 
                           tuple(np.multiply(elbow, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )
            
            # Curl counter logic
            if angle > 160:
                stage = "down"
            if angle < 30 and stage =='down':
                stage="up"
                counter +=1
                print(counter)
        except:
            pass
        
        # Render curl counter
        # Setup status box
        cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)
        
        # Rep data
        cv2.putText(image, 'REPS', (15,12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, str(counter), 
                    (10,60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
        
        # Stage data
        cv2.putText(image, 'STAGE', (65,12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, stage, 
                    (60,60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
        
        
        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                 )               
        
        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


###### CALCULATE ANGLES 

def Calculate_angle(a,b,c):
    a = np.array(a); # first landmark - shoulder
    b = np.array(b); # second landmark - elbow 
    c = np.array(c); # third landmark - wrist 
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0]);
    angle = np.abs(radians*180/np.pi);
    if angle>180.0:
        angle = 360-angle;
    return angle;

shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

Calculate_angle(shoulder, elbow, wrist);

"""
cap = cv2.VideoCapture(0)
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        # recolour image to rgb
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # changes bgr to rgb
        image.flags.writeable = False; # performance training 

        # make detection 
        results = pose.process(image);

        #recolor back to bgr 
        image.flags.writeable = True;
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR); 

        #extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark;

            #get coordinates
            shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y] 

            #get angle
            angle = Calculate_angle(shoulder, elbow, wrist);

            #Visualize 
            cv2.putText(image, str(angle),
                            tuple(np.multiply(elbow, [640,480]).astype(int)), # we multiply and hence manipulate the coordinates of the elbow since we wanna display 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA #the angle on the elbow itself hence coordinates of elbow * the size of cam feed for accurate placement 
                                )
            #print(landmarks);
        except:
            pass

        #render detection 
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness = 2, circle_radius = 2), # gives the specification for like customising how we wanna draw our landmarks 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness = 2, circle_radius = 2)
                                );
        cv2.imshow('Mediapipe Feed', image)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
"""


