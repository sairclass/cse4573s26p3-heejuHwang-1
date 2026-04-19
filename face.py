'''
Notes:
1. All of your implementation should be in this file. This is the ONLY .py file you need to edit & submit. 
2. Please Read the instructions and do not modify the input and output formats of function detect_faces() and cluster_faces().
3. If you want to show an image for debugging, please use show_image() function in helper.py.
4. Please do NOT save any intermediate files in your final submission.
'''


import torch

import face_recognition

from typing import Dict, List
from utils import show_image

'''
Please do NOT add any imports. The allowed libraries are already imported for you.
'''

def detect_faces(img: torch.Tensor) -> List[List[float]]:
    """
    Args:
        img : input image is a torch.Tensor represent an input image of shape H x W x 3.
            H is the height of the image, W is the width of the image. 3 is the [R, G, B] channel (NOT [B, G, R]!).

    Returns:
        detection_results: a python nested list. 
            Each element is the detected bounding boxes of the faces (may be more than one faces in one image).
            The format of detected bounding boxes a python list of float with length of 4. It should be formed as 
            [topleft-x, topleft-y, box-width, box-height] in pixels.
    """
    """
    Torch info: All intermediate data structures should use torch data structures or objects. 
    Numpy and cv2 are not allowed, except for face recognition API where the API returns plain python Lists, convert them to torch.Tensor.
    
    """
    detection_results: List[List[float]] = []

    ##### YOUR IMPLEMENTATION STARTS HERE #####
    # img: 3 x H x W
    img_np = img.permute(1, 2, 0).numpy()
    detection_list = face_recognition.face_locations(img_np, number_of_times_to_upsample=2)
    for i in range (len(detection_list)):
        (top, right, bottom, left) = detection_list[i]
        detection_results.append([float(left), float(top), float(right-left), float(bottom-top)])


    return detection_results



def cluster_faces(imgs: Dict[str, torch.Tensor], K: int) -> List[List[str]]:
    """
    Args:
        imgs : input images. It is a python dictionary
            The keys of the dictionary are image names (without path).
            Each value of the dictionary is a torch.Tensor represent an input image of shape H x W x 3.
            H is the height of the image, W is the width of the image. 3 is the [R, G, B] channel (NOT [B, G, R]!).
        K: Number of clusters.
    Returns:
        cluster_results: a python list where each elemnts is a python list.
            Each element of the list a still a python list that represents a cluster.
            The elements of cluster list are python strings, which are image filenames (without path).
            Note that, the final filename should be from the input "imgs". Please do not change the filenames.
    """
    """
    Torch info: All intermediate data structures should use torch data structures or objects. 
    Numpy and cv2 are not allowed, except for face recognition API where the API returns plain python Lists, convert them to torch.Tensor.
    
    """
    cluster_results: List[List[str]] = [[] for _ in range(K)] # Please make sure your output follows this data format.
        
    ##### YOUR IMPLEMENTATION STARTS HERE #####
    names_list = list(imgs.keys())

    encoding_list = []
    for i in range(len(names_list)):
        img = imgs[names_list[i]]
        img_np = img.permute(1, 2, 0).numpy()
        detection_results = detect_faces(img)
        face_locations = [(int(box[1]), int(box[0] + box[2]), int(box[1] + box[3]), int(box[0])) for box in detection_results]
        face_encodings = face_recognition.face_encodings(img_np, face_locations)
        encoding = [float(x) for x in face_encodings[0]]
        encoding_list.append(encoding)
    encoding_tensor = torch.tensor(encoding_list)
    kmeans_results = kmeans(encoding_tensor, K)

    for i in range(len(names_list)):
        name = names_list[i]
        label = kmeans_results[i]
        cluster_results[label].append(name)

    return cluster_results


'''
If your implementation requires multiple functions. Please implement all the functions you design under here.
But remember the above 2 functions are the only functions that will be called by task1.py and task2.py.
'''

# TODO: Your functions. (if needed)
def kmeans(points: torch.Tensor, K: int):
    # set furthest data as the initial center
    first_idx = torch.randint(0, len(points), (1,)).item()
    centers_list = [points[first_idx]]
    for _ in range(1, K):
        current_centers = torch.stack(centers_list)
        distances = torch.norm(points.unsqueeze(1) - current_centers.unsqueeze(0), dim=2)
        min_distances, _ = torch.min(distances, dim=1)
        next_center_idx = torch.argmax(min_distances).item()
        centers_list.append(points[next_center_idx])
    centers = torch.stack(centers_list)
    
    best_labels = None
    best_score = float('inf')
    # try multiple times
    for _ in range(5):
        e = 0
        while e < 300:
            distances = []
            # assign each point to nearest cluster
            for point in points:
                dis = []
                for i in range(K):
                    dis.append(torch.norm(point - centers[i]))
                distances.append(dis)
            labels = torch.argmin(torch.tensor(distances), dim=1)

            # compute new center
            new_centers = []
            for k in range(K):
                cluster_k_points = points[labels == k]
                new_center = torch.mean(cluster_k_points, dim=0) if len(cluster_k_points) > 0 else centers[k]
                new_centers.append(new_center)
            new_centers = torch.stack(new_centers)
            
            # break when there is no longer change in centers
            if torch.equal(centers, new_centers):
                min_distances, current_labels = torch.min(torch.tensor(distances), dim=1)
                current_score = torch.sum(min_distances).item()
                if current_score < best_score:
                    best_score = current_score
                    best_labels = current_labels
                break

            centers = new_centers
            e += 1

    return best_labels
