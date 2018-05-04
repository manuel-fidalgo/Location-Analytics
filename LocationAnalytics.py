"""
Importing the libraries
"""
import pandas as pd
import json 
import numpy as np

import matplotlib.pyplot as plt
import datetime
from sklearn.cluster import KMeans
from mpl_toolkits.basemap import Basemap
import statsmodels.formula.api as smf


#%%
"""
Aux functions
"""
#Lower left corner, Uppder rigth corner, location to test
def inside_the_region(lat_00, lon_00, lat_11, lon_11, lat_a, lon_a):
    
    if(lat_a > lat_00 & lat_a < lat_11 & lon_a > lon_00 & lat_a < lon_11):
        return True
    else:
        return False
    


def distance_between_locations(lat_a, lon_a, lat_b, lon_b):
    from math import sin, cos, sqrt, atan2, radians    
    # approximate radius of earth in km
    R = 6373.0

    lat_a_rad = radians(lat_a)
    lon_a_rad = radians(lon_a)
    lat_b_rad = radians(lat_b)
    lon_b_rad = radians(lon_b)

    diff_lon = lon_b_rad - lon_a_rad
    diff_lat = lat_b_rad - lat_a_rad

    a = sin(diff_lat / 2)**2 + cos(lat_a_rad) * cos(lat_b_rad) * sin(diff_lon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    return distance

#%%
"""
    Reading the JSON file.
"""

#Read the json file
dt = pd.read_json("Location History.json")
with open('Location History.json', 'r') as fh:
    raw = json.loads(fh.read())
    
data = pd.DataFrame(raw['locations'])


#Remove the raw data to free up memory
del raw 
del dt

#%%
"""
    Cleaning the data
"""

#Ignore locations with accuracy estimates over 500m
data = data[data.accuracy < 500]
data['latitudeE7'] = data['latitudeE7']/float(1e7) 
data['longitudeE7'] = data['longitudeE7']/float(1e7)
data['timestampMs'] = data['timestampMs'].map(lambda x: float(x)/1000) #to seconds
data['datetime'] = data.timestampMs.map(datetime.datetime.fromtimestamp)
# Rename fields based on the conversions we just did
data.rename(columns={'latitudeE7':'latitude', 'longitudeE7':'longitude', 'timestampMs':'timestamp'}, inplace=True)
data = data[data.accuracy < 100] #Ignore locations with accuracy estimates over 100m
data.reset_index(drop=True, inplace=True)


#Cheching the altitude, as we can see these values are nor correct
#So lets delete all the values over the quantile 90
data["altitude"].max()
data["altitude"].min()

#Lets remove all the altitudes 
altitude_q = data["altitude"].quantile(0.90)
data["altitude"] = data[data["altitude"] < altitude_q]


"""
Clustering the data
"""

plt.hist(data['accuracy'])

cols = ["latitude","longitude"]
kmeans_data = data[cols]


#Lets test is kmeans is a angorith to cluster location into different countries.
kmeans = KMeans(init='random', n_clusters=5, random_state=0).fit(kmeans_data)
clusters = kmeans.predict(kmeans_data)

#plt.scatter(kmeans_data["longitude"],kmeans_data["latitude"], c=clusters)


#%%
"""
Plotting in a map
"""

plt.clf()
plt.figure(figsize=(15,9))


lat_a, lon_a = 33.906957, -13.870325 #Lower left corner
lat_b, lon_b = 57.633679, 53.942175  #Uppder rigth corner


m = Basemap(llcrnrlon=lon_a, llcrnrlat=lat_a, urcrnrlon=lon_b, urcrnrlat=lat_b, resolution='l',projection='cass',lon_0=-4.36,lat_0=54.7)

m.drawcoastlines()
m.fillcontinents(color='linen',lake_color='skyblue')

# draw parallels and meridians.
m.drawparallels(np.arange(-70,61.,2.))
m.drawmeridians(np.arange(-20.,71.,2.))

m.drawmapboundary(fill_color='skyblue')
m.drawcountries()


lons = kmeans_data["longitude"].tolist()
lats = kmeans_data["latitude"].tolist()


x, y = m(lons,lats)
m.scatter(x,y,c=clusters,marker = 'o', zorder=10)


plt.title("European Map")
plt.show()

#%%
"""
Computing fligths
"""

##CODE TAKEN FROM WEB 

degrees_to_radians = np.pi/180.0 
data['phi'] = (90.0 - data.latitude) * degrees_to_radians 
data['theta'] = data.longitude * degrees_to_radians
# Compute distance between two GPS points on a unit sphere
data['distance'] = np.arccos( 
    np.sin(data.phi)*np.sin(data.phi.shift(-1)) * np.cos(data.theta - data.theta.shift(-1)) + 
    np.cos(data.phi)*np.cos(data.phi.shift(-1))
    ) * 6378.100 # radius of earth in km

data['speed'] = data.distance/(data.timestamp - data.timestamp.shift(-1))*3600 #km/hr
## END REGION


possible_flights = data[data["distance"] > 500]

lons = possible_flights["longitude"].tolist()
lats = possible_flights["latitude"].tolist()


plt.clf()
plt.figure(figsize=(15,9))

m = Basemap(llcrnrlon=lon_a, llcrnrlat=lat_a, urcrnrlon=lon_b, urcrnrlat=lat_b, resolution='l',projection='cass',lon_0=-4.36,lat_0=54.7)

m.drawcoastlines()
m.fillcontinents(color='linen',lake_color='skyblue')
# draw parallels and meridians.
m.drawparallels(np.arange(-70,61.,2.))
m.drawmeridians(np.arange(-20.,71.,2.))

m.drawmapboundary(fill_color='skyblue')
m.drawcountries()


x, y = m(lons,lats)
m.plot(x, y, 'o-', markersize=5, linewidth=1) 

plt.title("Possible flights")
plt.show()

#%% 
"""
Checking errors
Observing this map I realize that there is a point in sicilia, This is really weird because
I've never been to sicily, so lets analyze the error
"""

sicilia_lat, sicilia_lon = 38.110274, 13.362093

data["error"] = data.apply(lambda row: distance_between_locations(row["latitude"], row["longitude"], sicilia_lat, sicilia_lon) < 100 , axis=1)

errors = pd.DataFrame()
error = data[data.error == True]

#Asuming a flight is 900kmh
errors_due_speed = data[data["speed"] > 900]

plt.figure(figsize=(15,9))
m = Basemap(llcrnrlon=lon_a, llcrnrlat=lat_a, urcrnrlon=lon_b, urcrnrlat=lat_b, resolution='l',projection='cass',lon_0=-4.36,lat_0=54.7)
m.drawcoastlines()
m.fillcontinents(color='linen',lake_color='skyblue')

# draw parallels and meridians.
m.drawparallels(np.arange(-70,61.,2.))
m.drawmeridians(np.arange(-20.,71.,2.))

m.drawmapboundary(fill_color='skyblue')
m.drawcountries()

lons = errors_due_speed["longitude"].tolist()
lats = errors_due_speed["latitude"].tolist()

x, y = m(lons,lats)
m.scatter(x,y,marker='x',zorder=10)


plt.title("Possible errors in map")
plt.show()

#%%
"""
Predictive models with linear regresion
"""

model = smf.ols(formula="timestamp~latitude+longitude",data=data).fit()
print(model.params)
#print(model.sumary())

#Spliting the datagrame into train and test
mask = np.random.rand(len(data)) < 0.8
train = data[mask]
test = data[~mask]

pred = model.predict(train[["latitude","longitude"]])



fig = plt.figure(figsize=(15,9))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(train["latitude"], train["longitude"], train["timestamp"], zdir='z', s=20, c="coral", depthshade=True)
ax.scatter(train["latitude"], train["longitude"], pred, zdir='z', s=20, c=None, depthshade=True)


