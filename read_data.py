
import os
import cv2 as cv
import numpy as np

# Function to load images and annotations from a subject directory
def load_subject_data(subject_dir, start=0, end=1000):
    l_eye_imgs = []
    r_eye_imgs = []
    face_imgs = []
    labels = []
    
    days = [day_dir for day_dir in os.listdir(subject_dir) if day_dir.startswith("day")]
    index = 0
    
    for day in days:
        day_path = os.path.join(subject_dir, day)
        if os.path.isdir(day_path):
            
            annotation_path = os.path.join(day_path, 'data.txt') 
            
            # Load annotations from data.txt
            with open(annotation_path, 'r') as f:
                annotations = [line.strip().split() for line in f.readlines()]
                
            # Load images and annotations
            for ann in annotations:
                
                if index < start :
                    index += 1
                    continue
                
                if index == end:
                    break

                image_name = ann[0]  
                label = ann[1:]
                
                l_eye_img = cv.imread(f"{day_path}/left_eye/{image_name}",cv.IMREAD_GRAYSCALE) / 255.0
                
                r_eye_img = cv.imread(f"{day_path}/right_eye/{image_name}",cv.IMREAD_GRAYSCALE) / 255.0
                
                face_img = cv.imread(f"{day_path}/face/{image_name}")
                face_img = cv.cvtColor(face_img,cv.COLOR_BGR2RGB)  / 255.0
                
                l_eye_imgs.append(l_eye_img)
                r_eye_imgs.append(r_eye_img)
                face_imgs.append(face_img)
                labels.append(label)
                
                index +=1

    print("Rows: " + str(len(labels)))

    return np.array(l_eye_imgs),np.array(r_eye_imgs),np.array(face_imgs),np.array(labels, dtype=float) 

# Function to load the entire dataset
def load_dataset(dataset_type="original", start=0, end=1000):
    path = f"data_subset/{dataset_type}" 

    

    l_eye_list = []
    r_eye_list = []
    face_list = []
    label_list = []
    
    for i in range(15):
        subject_path = os.path.join(path, f"p{i:02d}")

        print(f"Reading => {subject_path}")

        if os.path.isdir(subject_path):
            l_eye_imgs, r_eye_imgs, face_imgs, labels = load_subject_data(subject_path, start=start, end=end)
            
            l_eye_list.extend(l_eye_imgs)
            r_eye_list.extend(r_eye_imgs)
            face_list.extend(face_imgs)
            label_list.extend(labels)
            
    return np.array(l_eye_list), np.array(r_eye_list), np.array(face_list), np.array(label_list, dtype=float)