import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose

pose_handler = mp_pose.Pose(
    static_image_mode=False, 
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def get_pose_features(frame_rgb):
    results = pose_handler.process(frame_rgb)
    if not results.pose_landmarks:
        return np.array([0.0, 0.5, 0.0])
        
    points = results.pose_landmarks.landmark
    
    nose = np.array([points[mp_pose.PoseLandmark.NOSE].x, points[mp_pose.PoseLandmark.NOSE].y])
    left_shoulder = np.array([points[mp_pose.PoseLandmark.LEFT_SHOULDER].x, points[mp_pose.PoseLandmark.LEFT_SHOULDER].y])
    right_shoulder = np.array([points[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, points[mp_pose.PoseLandmark.RIGHT_SHOULDER].y])
    
    mid_shoulder = (left_shoulder + right_shoulder) / 2.0
    neck_vector = nose - mid_shoulder
    
    angle = np.arctan2(neck_vector[1], neck_vector[0])
    dist = np.linalg.norm(left_shoulder - right_shoulder)
    offset_x = neck_vector[0]
    
    return np.array([angle, dist, offset_x])