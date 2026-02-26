from sklearn import load_iris 
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.utils import shuffle

data = load_iris()

X = data.data

X = shuffle(X,random_state=42) #since that the dataSet is ordered , it better to shuffle it

scaler = StandardScaler()

scaled_data = scaler.fit_transform(X)

#here i will define my kmeans model algorithm :
kmeans = KMeans(
    n_clusters=3 ,  # there is 3 types of iris flowers 
    random_state= 42 ,
    n_init = 10 # the safer one , it will make 10 random intial centroid and then pick the one with the least inertia
)

kmeans.fit(scaled_data)

myLabels = kmeans.labels_

print(myLabels[:10]) #printing an example of 10 labels

#i would add a success rate for this cz we had labels in the iris dataSets (comparing it to what we had and then getting the pourcentage of successful clustring)

# but since in real kmeans situations we can't do that since we don't have lables i didn't do so 

#however you can do it by adding a simple algorithm where you will use myLabels and data.target(the real iris labels)



