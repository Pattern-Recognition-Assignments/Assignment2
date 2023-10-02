import numpy as np
import random
class clusterPoint:
    def __init__(self, point, cluster):
        self.point = point
        self.cluster = cluster
class Cluster_representation:
    def __init__(self,mean,covariance,weight):
        self.mean=mean
        self.covariance=covariance
        self.weight=weight
        
class K_Mahalanobis:
            
    def dissimilarity(self, point1, which_cluster):
        # print(self.clusters[which_cluster].mean)
        diff = np.array(point1) - np.array(self.clusters[which_cluster].mean)
        return np.sqrt(np.dot(diff.T, np.dot(np.linalg.inv(self.clusters[which_cluster].covariance), diff)))
    
    def __init__(self, k):
        self.k = k
        self.clusters = [Cluster_representation(None,None,None) for i in range(k)]
        
    def distortion(self, data):
        sum = 0
        for point in data:
            sum += self.dissimilarity(point, self.clusters[point.cluster].mean)
        return sum
    
    def updateK(self):
        self.k+=1
        
    # data here is a list of feature vectors return the cluster points
    def fit(self,data):
        # initilizing cluster centers
        centers=set()
        while len(centers)<self.k:
            centers.add(random.randint(0,len(data)-1))
        centers=list(centers)
        
        self.clusters = [Cluster_representation(data[i],np.eye(data[0].shape[0]),1/self.k) for i in centers]

        cluster_points = []
        for points in data:
            cluster_points.append(clusterPoint(points, 0))
            
        iterations=0
        while True:
            # assign each feature vector to a cluster(we have hard clustering in K-means)
            new_clusters = [[] for i in range(self.k)]
            for point in cluster_points:
                which_cluster = 0
                for i in range(self.k):
                    if self.dissimilarity(point.point, i) < self.dissimilarity(point.point, which_cluster):
                        which_cluster = i
                new_clusters[which_cluster].append(point)
                point.cluster = which_cluster
                
            # update cluster parameters
            new_means = []
            for i in range(self.k):
                new_means.append(np.mean([point.point for point in new_clusters[i]], axis=0))
            new_covariances = []
            for i in range(self.k):
                new_covariances.append(np.cov([point.point for point in new_clusters[i]], rowvar=False))
            
            new_weights = []
            for i in range(self.k):
                new_weights.append(len(new_clusters[i]) / len(cluster_points))
            
            prev_centers=[cluster.mean for cluster in self.clusters]
            self.clusters = [Cluster_representation(new_means[i],new_covariances[i],new_weights[i]) for i in range(self.k)]
                
            converged=True
            for i in range(self.k):
                if not np.array_equal(prev_centers[i], self.clusters[i].mean):
                   converged = False
            
            if(converged):
                break
            
            iterations+=1
            
            print("iteration",iterations )
            
            if(iterations>5):
                break
        
        return cluster_points,iterations


class K_Means(K_Mahalanobis):
    def __init__(self, k):
        super().__init__(k)
               
    def dissimilarity(self, point1, point2):
        return mahalanobis(point1, point2, np.linalg.inv(np.cov(np.array(point1).T, np.array(point2).T)))
        