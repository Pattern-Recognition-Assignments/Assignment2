def mahalanobis(point, mean, covariance):
    diff = point - mean
    
    # Calculate the Mahalanobis distance
    inv_covariance = np.linalg.inv(covariance)
    distance = np.sqrt(np.dot(np.dot(diff.T, inv_covariance), diff))
    
    return distance