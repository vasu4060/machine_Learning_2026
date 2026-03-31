import numpy as np

p_list = [[1,2],[1,4],[1,0],[10,2],[10,4],[10,0]]

p_arr = np.array(p_list)



k =5

C1 = np.array([1,0])

C2 = np.array([10,0])
clus_1 = []
clus_2 = []
dist_C1 =[]
dist_C2 =[]

def distance(a,b):
                    
         # return np.sqrt(np.sum((a[1]-b[1])**2)+(np.sum((a[0]-b[0])**2)))
         return np.linalg.norm(a - b)

def compute_mean(C1,C2):
          C1_mean = np.mean(C1,axis=0)
          C2_mean = np.mean(C2,axis=0)
          return C1_mean, C2_mean


for i in range(k):
          clus_1 = []
          clus_2 = []
          for p in range(len(p_arr)):
                    print("p_arr[p]: ",p_arr[p])
                    print("C1: ",C1)
                    print("C2: ",C2)
                    dist_C1 = round(distance(p_arr[p],C1),2)
                    
                    dist_C2 = round(distance(p_arr[p],C2),2)
                    if dist_C1 < dist_C2:
                              print("p_arr[p] belongs to C1")
                              
                              clus_1.append(p_arr[p])
                    else:
                              print("p_arr[p] belongs to C2")
                              
                              clus_2.append(p_arr[p])       
                              
                    print(dist_C1)
                    print(dist_C2)
          print("clus_1: ",clus_1)
          print("clus_2: ",clus_2)

          new_C1,new_C2 = compute_mean(clus_1,clus_2)
         

          if np.allclose(C1, new_C1) and np.allclose(C2, new_C2):
                    print("Centroids have converged. Stopping iterations.")
                    print("New C1 ",new_C1)
                    print("New C2 ",new_C2)
                    break
          C1, C2 = new_C1, new_C2

def compute_wcss(clusters, centroids):
    wcss = 0
    for i, cluster in enumerate(clusters):
        for point in cluster:
            wcss += np.linalg.norm(point - centroids[i])**2
    return wcss





