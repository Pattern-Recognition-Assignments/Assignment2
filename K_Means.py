class clusterPoint:
    def __init__(self, point, cluster):
        self.point = point
        self.cluster = cluster
        
class K_means:
            
    def  dissimilarity(self, point1, point2):
        return np.linalg.norm(np.array(point1)-np.array(point2));
    
    def __init__(self, k):
        self.k = k
        self.clusters = [None for i in range(k)]
        
    # data here is a list of feature vectors
    def fit(self,data):
        # initilizing cluster centers
        centers=set()
        while len(centers)<self.k:
            centers.add(random.randint(0,len(data)-1))
        centers=list(centers)
        
        Centroids = [data[i] for i in centers]
        
        while True:
            slef.EM(data)
            if
        
    def EM(self, data):
        # assign each feature vector to a cluster(we have hard clustering in K-means)
        cluster_points = []
        for points in data:
            which_cluster = 0
            for i in range(self.k):
                if self.dissimilarity(points, Centroids[i]) < self.dissimilarity(points, Centroids[which_cluster]):
                    which_cluster = i
            cluster_points.append(clusterPoint(points, which_cluster))
            
        # update cluster parameters
        for i in range(self.k):
            self.clusters[i]=cluster()
            self.clusters[i].set(Centroids[i])
            
            
# little change in K_means
class K_Mahalanobis(K_means):
    def __init__(self, k):
        super().__init__(k)
               
    def dissimilarity(self, point1, point2):
        return mahalanobis(point1, point2, np.linalg.inv(np.cov(np.array(point1).T, np.array(point2).T)))
            
            
        
    
    