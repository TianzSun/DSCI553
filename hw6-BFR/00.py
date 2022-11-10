centroid = DS[cluster]["SUM"] / len(DS[cluster]["N"])
sigma = DS[cluster]["SUMSQ"] / len(DS[cluster]["N"]) - (DS[cluster]["SUM"]/len(DS[cluster]["N"]))**2
z = (point - centroid)/sigma
m_distance = np.dot(z, z) ** (1/2)