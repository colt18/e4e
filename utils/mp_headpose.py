import mediapipe as mp
import numpy as np


class HeadPoseDetector:    
    def __init__(self):
        self.BaseOptions = mp.tasks.BaseOptions
        self.FaceLandmarker = mp.tasks.vision.FaceLandmarker
        self.FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        self.FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
        self.VisionRunningMode = mp.tasks.vision.RunningMode
        # !wget -O face_landmarker_v2_with_blendshapes.task -q https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
        self.model_path='checkpoints/face_landmarker_v2_with_blendshapes.task'
        
        self.options = self.FaceLandmarkerOptions(
            base_options=self.BaseOptions(model_asset_path=self.model_path),
            running_mode=self.VisionRunningMode.IMAGE,
            output_facial_transformation_matrixes=True,
            )
        
        self.landmarker = self.FaceLandmarker.create_from_options(options=self.options)

    def process_image(self, img):
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)

        face_landmarker_result = self.landmarker.detect(img)
        roll, pitch, yaw = self.rotation_matrix_to_euler_angles(face_landmarker_result.facial_transformation_matrixes[0])
        return roll, pitch, yaw

    def rotation_matrix_to_euler_angles(self, R):
        sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])    
        singular = sy < 1e-6
        
        if not singular:
            x = np.arctan2(R[2, 1], R[2, 2])
            y = np.arctan2(-R[2, 0], sy)
            z = np.arctan2(R[1, 0], R[0, 0])
        else:
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(-R[2, 0], sy)
            z = 0

        pitch = np.degrees(x)
        yaw = np.degrees(y)
        roll = np.degrees(z)
        
        return roll, pitch, yaw
