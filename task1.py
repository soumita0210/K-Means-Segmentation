"""
K-Means Segmentation Problem
(Due date: Nov. 25, 11:59 P.M., 2019)
The goal of this task is to segment image using k-means clustering.

Do NOT modify the code provided to you.
Do NOT import ANY library or API besides what has been listed.
Hint: 
Please complete all the functions that are labeled with '#to do'. 
You are allowed to add your own functions if needed.
You should design you algorithm as fast as possible. To avoid repetitve calculation, you are suggested to depict clustering based on statistic histogram [0,255]. 
You will be graded based on the total distortion, e.g., sum of distances, the less the better your clustering is.
"""


import utils
import numpy as np
import json
import time


def kmeans(img,k):
    """
    Implement kmeans clustering on the given image.
    Steps:
    (1) Random initialize the centers.
    (2) Calculate distances and update centers, stop when centers do not change.
    (3) Iterate all initializations and return the best result.
    Arg: Input image;
         Number of K. 
    Return: Clustering center values;
            Clustering labels of all pixels;
            Minimum summation of distance between each pixel and its center.  
    """
    # TODO: implement this function.
    
    img_list=img.flatten()
    
    unique_pixels=np.unique(img_list)
    combinations=[]

    for i in range(len(unique_pixels)):
        for j in range(i,len(unique_pixels)):
            if(unique_pixels[i]!=unique_pixels[j]):
                combinations.append([unique_pixels[i],unique_pixels[j]])
                
    final_centroids=[]
    final_labels=[]
    sum_min=1000000000000000
    
    m=len(img_list)   
    
    for centroids in combinations:
        new_centroids=[]   

        pixel_labels=[]
        distance=np.array([]).reshape(m,0)


        while(centroids!=new_centroids):
    
            new_centroids=centroids.copy()
    
            distance=np.array([]).reshape(m,0)
    
            for i in range(k):
                temp=abs(img_list-centroids[i])
                distance=np.c_[distance,temp]
        
            pixel_labels=np.argmin(distance,axis=1)
    
            centroids=list(np.array([img_list[pixel_labels==x].mean(axis=0) for x in range(k)]))

        min_dist=np.amin(distance,axis=1)  
        sum_of_dist=np.sum(min_dist) 
        
        if(sum_of_dist<sum_min):
            sum_min=sum_of_dist
            final_labels=np.reshape(pixel_labels,(len(img),len(img[0])))
            final_centroids=centroids
    
    
    return (final_centroids,final_labels,sum_min)


def visualize(centers,labels):
    """
    Convert the image to segmentation map replacing each pixel value with its center.
    Arg: Clustering center values;
         Clustering labels of all pixels. 
    Return: Segmentation map.
    """
    # TODO: implement this function.
    segmentation_map=labels.copy()
    
    for i in range(len(labels)):
        for j in range(len(labels[0])):
            segmentation_map[i][j]=centers[labels[i][j]]
            
    
    segmentation_map=segmentation_map.astype(np.uint8)  
    
    return segmentation_map
        

     
if __name__ == "__main__":
    img = utils.read_image('lenna.png')
    k = 2

    start_time = time.time()
    centers, labels, sumdistance = kmeans(img,k)
    result = visualize(centers, labels)
    end_time = time.time()

    running_time = end_time - start_time
    print(running_time)

    centers = list(centers)
    with open('results/task1.json', "w") as jsonFile:
        jsonFile.write(json.dumps({"centers":centers, "distance":sumdistance, "time":running_time}))
    utils.write_image(result, 'results/task1_result.jpg')
