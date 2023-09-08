import math

import torch
import numpy as np
# from .. import DEVICE # Imagine you have a GPU device and write code using it.
import cv2
from ultralytics import YOLO
device="cuda" if torch.cuda.is_available() else "cpu"
class FallDetecion():
    def __init__(self,cache_size=108):
        self.cache_size=cache_size
        self.cache=torch.zeros(cache_size)


    def __call__(self, skeleton_cache):
        '''
            This __call__ function takes a cache of skeletons as input, with a shape of (M x 17 x 2), where M represents the number of skeletons.
            The value of M is constant and represents time. For example, if you have a 7 fps stream and M is equal to 7 (M = 7), it means that the cache length is 1 second.
            The number 17 represents the count of points in each skeleton (as shown in skeleton.png), and 2 represents the (x, y) coordinates.

            This function uses the cache to detect falls.

            The function will return:
                - bool: isFall (True or False)
                - float: fallScore
        '''

        isFall=False
        fallScore=0
        i=0
        while i < self.cache_size and torch.all(skeleton_cache[i] == 0):
            i += 1
        if i<self.cache_size:
            alpha1=self.angle_calculation(skeleton_cache[i])
            i+=1
        else:
            return
        while i <self.cache_size:
            while i<self.cache_size and torch.all(skeleton_cache[i] == 0):
                i+=1
            if i<self.cache_size:
               alpha2=self.angle_calculation(skeleton_cache[i])
               i+=1
            else:
               break
            mean=torch.mean(alpha2-alpha1)
            self.cache=self.cache.roll(-1)
            self.cache[-1]=mean
            alpha1.copy_(alpha2)
#            formatted="{:.10f}".format(torch.mean(self.cache))
#            print(formatted)
            fallScore = round(torch.mean(self.cache).item(),8)
            if round(torch.mean(self.cache).item(),8)>0.00168:
                isFall=True


        return isFall, fallScore

    def angle_calculation(self,key_points):#key_points_size=[17,2]
        """
        Calculates angles between keypoints on a person's body.

        Args:
        key_points (torch.Tensor): A 2D tensor containing keypoints (size=[17, 2]).

        Returns:
        torch.Tensor: A tensor containing calculated angles.
        """
        (nose_idx,l_eye_idx,r_eye_idx,l_ear_idx,r_ear_idx,l_shoulder_idx,r_shoulder_idx,l_elbow_idx,
        r_elbow_idx,l_wrist_idx,r_wrist_idx,l_hip_idx,r_hip_idx,l_knee_idx,r_knee_idx,l_ankle_idx,r_ankle_idx)=(i for i in range(17))

        angle1=self.angle_between_vector(key_points[r_elbow_idx],key_points[r_shoulder_idx],key_points[r_hip_idx])
        angle2=self.angle_between_vector(key_points[l_hip_idx],key_points[l_shoulder_idx],key_points[l_elbow_idx])
        angle3=self.angle_between_vector(key_points[r_knee_idx],key_points[r_hip_idx],key_points[l_hip_idx])
        angle4=self.angle_between_vector(key_points[r_hip_idx],key_points[l_hip_idx],key_points[l_knee_idx])
        angle5=self.angle_between_vector(key_points[r_hip_idx],key_points[r_knee_idx],key_points[r_ankle_idx])
        angle6=self.angle_between_vector(key_points[l_hip_idx],key_points[l_knee_idx],key_points[l_ankle_idx])
        mid_hip=(key_points[l_hip_idx]+key_points[r_hip_idx])/2
        mid_shoulder=(key_points[l_shoulder_idx]+key_points[r_shoulder_idx])/2

        angle7 = self.angle_between_vector(key_points[nose_idx] + torch.tensor([0, 1]).to(device="cuda"),
                                      key_points[nose_idx],
                                      mid_shoulder)
        angle8 = self.angle_between_vector(key_points[nose_idx] + torch.tensor([0, 1]).to(device="cuda"),
                                      key_points[nose_idx],
                                      mid_hip)
        return torch.tensor([angle1,angle2,angle3,angle4,angle5,angle6,1.5*angle7,4*angle8])
    def angle_between_vector(self,p1, p2, p3):
        """
        Calculates the angle (in degrees) between two vectors formed by three points.

        Args:
        p1 (torch.Tensor): The first point, represented as a vector (tensor).
        p2 (torch.Tensor): The second point, represented as a vector (tensor), serving as the vertex of the angle.
        p3 (torch.Tensor): The third point, represented as a vector (tensor).

        Returns:
        float: The angle between the vectors in degrees.
        """
        vector1 = p1 - p2
        vector2 = p3 - p2
        angle =  torch.arccos(torch.dot(vector1, vector2) / (torch.norm(vector1) * torch.norm(vector2)))
        return (angle*(180.0 / math.pi))/360

#def video_make_frames(video_path):
#    cap=cv2.VideoCapture(video_path)
#    output=[]
#    while True:
#        ret,frame=cap.read()
#        if not ret:
#            break
#        output.append(frame[...,::-1])

#    return output


#def key_points_maker(frames_list):
#    model = YOLO('yolov8n-pose.pt')
#    results = model(frames_list, save=False)
#    output = []

#    i = 0
#    previous = None
#    while i < len(results):
#        if (results[i].keypoints.xy.shape[1] == 17 and results[i].keypoints.xy.shape[2] == 2):
#            output.append(results[i][0].keypoints.xy)
#            previous = results[i][0].keypoints.xy
#            i += 1
#            break
#    while i < len(results):
#        if results[i].keypoints.xy.shape[0] > 1 and results[i].keypoints.xy.shape[1] == 17 and \
#                results[i].keypoints.xy.shape[2] == 2:
#            distances_squared = torch.sum((previous - results[i].keypoints.xy) ** 2, dim=2)
#            distances = torch.sqrt(distances_squared)  # shape is (4,17)
#            row_norms = torch.norm(distances, dim=1)  # shape is(4)
#            min_distance = torch.min(row_norms)
#            min_index = torch.argmin(row_norms)
#            output.append(results[i][min_index].keypoints.xy)
#            previous = results[i][min_index].keypoints.xy
#            i += 1
#        elif results[i].keypoints.xy.shape[0] == 1 and results[i].keypoints.xy.shape[1] == 17 and \
#                results[i].keypoints.xy.shape[2] == 2:
#            output.append(results[i][0].keypoints.xy)
#            previous = results[i][0].keypoints.xy
#            i += 1
#        else:
#            output.append(torch.zeros(17, 2).unsqueeze(0).to(device))
#            i += 1

    #    print(results[i].names)
#    return torch.stack(output).squeeze(1)  # (len(frames_for_person) consisits of (1,17,2)


#def make_cache_size(all_keypoints):
#    i=0
#    fdy = FallDetecion()
#    while i <len(all_keypoints)-cache_size+1:
#        temporary_interval=all_keypoints[i:i+cache_size]
#        i+=cache_size
#        if fdy(temporary_interval)[0]==True:
#            print("SOS")
#            fdy=FallDetecion()
#    if i<len(all_keypoints):
#        temporary_interval = all_keypoints[i:len(all_keypoints)]
#        tensor_zeroes=torch.zeros(cache_size-len(all_keypoints)%cache_size, 17, 2,device=device)
#        temporary_interval=torch.cat((temporary_interval,tensor_zeroes),dim=0)
 #       fdy=FallDetecion()
#        if fdy(temporary_interval)[0]==True:
#            print("SOS")
#            fdy=FallDetecion()

#video_path="document_5462876106166123817.mp4"
#output=video_make_frames(video_path)
#output=key_points_maker(output)
#output (n,17,2) i want to devide (108,17,2)

#make_cache_size(output)