import numpy as np
import random
import matplotlib.pyplot as plt
import copy
import cv2
import h5py
import sklearn.metrics
import math
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from mpl_toolkits.mplot3d import Axes3D


# Function： Set up distance matrix
#  Input：Dataset
#  Output: distance matrix
def compute_squared_EDM(Dataset):
  return squareform(pdist(Dataset,metric='euclidean'))


# Function：Calculate the core point ratio CPR
#  Input: DistMatrix、Eps、Minpts
#  Output: CPR
def Build_CorePoint_Rate(Dataset,Eps,Minpts):
    disMat = compute_squared_EDM(Dataset)
    core_points_index = np.where(np.sum(np.where(disMat <= Eps, 1, 0), axis=1) >= Minpts)[0]
    CorePoint_Ratio = len(core_points_index)/len(Dataset)
    return CorePoint_Ratio

# Function：Establish a set of standardized fixation point coordinatesXY
#  Input：Fixation point coordinates x y
#  Output：Normalized set of fixation point coordinates XY
def build_XY(x, y):
    FixationPointX = np.asarray(x)
    FixationPointY = np.asarray(y)
    XY = list(zip(FixationPointX,FixationPointY))
    XY = np.asarray(XY)
    XY = np.squeeze(XY)
    return XY

# Function：Find the neighbor of the target object in the eps neighborhood
#  Input：Target object index j, eye movement fixation coordinate Dataset, eps set
#  Output：The set of index values N of the neighborhood points of the target object
def find_neighbor(j, Dataset, eps):
    N = list()
    temp = np.sum((Dataset-Dataset[j])**2, axis=1)**0.5
    N = np.argwhere(temp <= eps).flatten().tolist() # np.argwhere: Returns the index value that matches the condition in ()
    return set(N)
 
# Function: Implement DBSCAN algorithm
#  Input：Dataset of eye movement fixation point coordinates, eps and Minpts
#  Output：Label the cluster number set C_label for each index value
def DBSCAN(Dataset, eps, min_Pts):
    k = -1
    neighbor_list = []  # The neighborhood used to hold each piece of data
    omega_list = []  # omega is the core object set
    gama = set([x for x in range(len(Dataset))])  # Initially mark all points as unvisited (gama is a collection of unvisited points)
    C_label = [-1 for _ in range(len(Dataset))]  # Clustering (-1 indicates no clustering processing)
    for i in range(len(Dataset)):
        neighbor_list.append(find_neighbor(i, Dataset, eps))
        if len(neighbor_list[-1]) >= min_Pts:
            omega_list.append(i)  # Add the sample to the core object collection
    omega_list = set(omega_list)  # Convert to a collection for easy manipulation
    while len(omega_list) > 0:
        gama_old = copy.deepcopy(gama)
        j = random.choice(list(omega_list))  # Pick a core object at random
        k = k + 1 # One k corresponds to one cluster, one at a time
        Q = list()
        Q.append(j) # Initialize the sequence
        gama.remove(j)# Delete from the never access collection
        while len(Q) > 0:
            q = Q[0] # In the first loop, there's only one element in Q, and then there's more
            Q.remove(q) # Walk through the core object in turn (because the density is connected also counts as a cluster)
            if len(neighbor_list[q]) >= min_Pts: # If the adjacent object is also the core object
                delta = neighbor_list[q] & gama # delta is the intersection of the neighborhood and the unaccessed set, and is the cluster around the current core object
                deltalist = list(delta) # Convert to list form
                for i in range(len(delta)):
                    Q.append(deltalist[i]) # Add clusters of j core objects to Q collection in turn
                    gama = gama - delta # You don't need to repeat, can you optimize outside of the for loop?
        Ck = gama_old - gama # Previously unvisited set - Now unvisited set, get the visited set, Ck is the cluster
        Cklist = list(Ck)
        for i in range(len(Ck)):
            C_label[Cklist[i]] = k # Tags objects belonging to the cluster with a uniform number
        omega_list = omega_list - Ck # Remove objects that have been classified into the cluster until all removed
    return C_label

# Function: Based on C_label, the Dataset of the original fixation point was grouped into clusters, stored in the form of 3D list, and the xy coordinates of all noise points were gathered into aset
#  Input：The Dataset of eye movement fixation point coordinates marked the cluster number set C_label for each index value
#  Output：C_list contains a 3 D list of xy coordinates of each cluster fixation point
def build_C_list(Dataset,C_label):
    # Cl=np.array([[],]*(cluster_num+1))
    C_list = list([[],]*(Cluster_num))
    Noise = list([])
    # According to C_label, the data is grouped by Cluster and stored in the form of 3D list
    for i in range(len(Dataset)):
        if C_label[i] == -1:
            Noise=Noise+list([Dataset[i]])
        if C_label[i] != -1:
            C_list[C_label[i]] = C_list[C_label[i]]+[Dataset[i]]
            # Cl[C[i]]=np.append(Cl[C[i]],X[i])
            # Cl[C[i]].append(list(X[i]))
    for i in range(len(C_list)): # Converts C_list to list form for easy processing
        for j in range(len(C_list[i])):
            C_list[i][j] = list(C_list[i][j])
    
    for i in range(len(Noise)):# Convert the noise to a list form for easy processing
            Noise[i] = list(Noise[i])

    return C_list, Noise

# Function: Record the bounds of the cluster region
#  Input：C_list contains a 3 D list of xy coordinates of each cluster fixation point
#  Output：The upper, lower,left, and left borders of each classification cluster top,bottom,left,right
def build_range(C_list):
    top = []
    bottom = []
    left = []
    right = []
    for i in range(len(C_list)):
        x_temp = []
        y_temp = []
        for j in range(len(C_list[i])):  
            y_temp.append(C_list[i][j][0])
            x_temp.append(C_list[i][j][1]) # Note: In the picture output, the axes represented by x and y seem counter-intuitive
        top.append(min(x_temp))
        bottom.append(max(x_temp))
        left.append(min(y_temp))
        right.append(max(y_temp))
    return top,bottom,left,right


# Function：Calculate Euclidean distance between two points (two-dimensional)
#  Input：Sample pointa、b
#  Output: Euclidean distance between two sample points
def dist(a,b):
    return math.sqrt(math.pow(a[0]-b[0],2) + math.pow(a[1]-b[1],2)) # 二维以上时：for i in range(x.shape[0]):   temp = np.sqrt(np.sum(np.square(x[j]-x[i])))


# Function：Construct distance distribution matrix, D(n×n) is n×n real symmetric matrix; n is the number of objects contained in dataset D; Dist(i,j) is the distance from the ith object to the JTH object in dataset D
#  Input：Dataset
#  Output: Distance matrix
def calculate_distMatrix(Dataset):
    DistMatrix = [[0 for j in range(len(Dataset))] for i in range(len(Dataset))] # Let's create an all-zero matrix
    for i in range(len(Dataset)):
        for j in range(len(Dataset)):
            DistMatrix[i][j] = dist(Dataset[i], Dataset[j]) # Store all mutual distances in turn as the original all-zero distance matrix
    return DistMatrix




##########  Main function  ##########

if __name__ == '__main__':

    ### Read the overall data set data

    f = h5py.File("dataset\etdb_v1.0.hdf5", "r") 
    trial = f["/Age study/trial"][:]
    SUBJECTINDEX = f["/Age study/SUBJECTINDEX"][:]
    category = f["/Age study/category"][:]
    filenumber = f["/Age study/filenumber"][:]
    age = f["/Age study/age"][:]
    x = f["/Age study/x"][:]
    y = f["/Age study/y"][:]

    ### Extract eye movement data records for specific images

    # It is divided into four groups of xy coordinates according to synthesis, age label 2, 1, 0
    pic=[]  # Records the eye movement data label for the specified image file
    pic_x=[]
    pic_y=[]
    pic_x_2=[]
    pic_y_2=[]
    pic_x_1=[]
    pic_y_1=[]
    pic_x_0=[]
    pic_y_0=[]
    Category_Aim=8
    Filenumber_Aim=11
    for i in range(len(trial)):
        if category[i] == Category_Aim and filenumber[i] == Filenumber_Aim:
            pic.append(i)
            pic_x.append(x[i])
            pic_y.append(y[i])
        if category[i] == Category_Aim and filenumber[i] == Filenumber_Aim and age[i] == 2:
            pic.append(i)
            pic_x_2.append(x[i])
            pic_y_2.append(y[i])
        if category[i] == Category_Aim and filenumber[i] == Filenumber_Aim and age[i] == 1:
            pic.append(i)
            pic_x_1.append(x[i])
            pic_y_1.append(y[i])
        if category[i] == Category_Aim and filenumber[i] == Filenumber_Aim and age[i] == 0:
            pic.append(i)
            pic_x_0.append(x[i])
            pic_y_0.append(y[i])

        
    ### Call the build_XY function to build a normalized set of fixation points XY

    XY=build_XY(pic_x,pic_y)
    Dataset=XY

    ### Calculated CPR surface
    Eps_max = 50
    Minpts_max = 50
    Eps_step = 2
    Minpts_step = 1
    CPR_aim = 0.3
    EpsCandidate = []
    MinptsCandidate = []
    CPR_offset = 0.01
    
    CPR = [[[] for j in range(Minpts_max)]for i in range(Eps_max)]
    for i in range(Eps_max):
        for j in range(Minpts_max):
            Eps = Eps_step*i
            Minpts = Minpts_step*j
            CPR[i][j] = Build_CorePoint_Rate(Dataset,Eps,Minpts)
            if CPR[i][j] >= CPR_aim-CPR_offset and CPR[i][j] <= CPR_aim+CPR_offset:
                EpsCandidate.append(Eps)
                MinptsCandidate.append(Minpts)

    # Create three-dimensional coordinates
    Eps_axis = np.array([i*Eps_step for i in range(Eps_max)])
    Minpts_axis = np.array([j*Minpts_step for j in range(Minpts_max)])
    Eps_axis,Minpts_axis = np.meshgrid(Eps_axis,Minpts_axis)
    CPR_axis = np.array(CPR)

    fig = plt.figure()
    ax = Axes3D(fig)
    
    # Draw a three-dimensional surface diagram
    ax.plot_surface(Eps_axis, Minpts_axis, CPR_axis, rstride=1, cstride=1,cmap=plt.cm.coolwarm)

    # Set three axes information
    ax.set_xlabel('Eps', color='b')
    ax.set_ylabel('Minpts', color='g')
    ax.set_zlabel('CPR', color='r')
    plt.draw()
    plt.show()

    # Show equal objective CPR curve
    Max_para = 15
    x = EpsCandidate
    y = MinptsCandidate
    plt.plot(x,y,label="CPR_aim is = {}".format(CPR_aim),color='r',marker='o',markerfacecolor='blue',markersize=0)
    plt.xlabel('Eps')
    plt.ylabel('Minpts')
    plt.xlim(0, 100)
    plt.ylim(0, 50)
    plt.title('parameter')
    plt.legend()
    plt.show()

    ### Run the DBSCAN algorithm

    # Set eps and min_pts
    Minpts = 10
    Eps = 40

    # Call DBSCAN to create the cluster label set C_label
    C_label = DBSCAN(Dataset, Eps, Minpts)

    ### Sort the DBSCAN output

    # Record and output the total number of clusters
    Cluster_num = max(C_label)+1 # Output the total number of clusters (+1 since 0 counts as a cluster label)
    print("Total cluster number is:")
    print(Cluster_num)
    # The build_C_list function was called, based on C_label, the Dataset of the original fixation point was grouped by cluster, stored as C_list in the form of 3D list, and the xy coordinates of all noise points were gathered into aset
    C_list,noise=build_C_list(Dataset,C_label)
    # Call build_range and record the top,bottom,left, and right boundaries of each sort cluster
    top,bottom,left,right = build_range(C_list)

    # Calculate and display contour coefficients
    print("silhouette coefficient is:")
    print(sklearn.metrics.silhouette_score(Dataset, C_label))

    # Calculate and display the Davis-Bouldin Index
    print("Davies-Bouldin Index is:")
    print(sklearn.metrics.davies_bouldin_score(Dataset, C_label))

    ### Display picture results

    # Draw the cluster area in the picture
    # You aggregate points that belong to the same cluster
    img = cv2.imread("imgs\\8_11.png",-1)
    plt.figure()

    # Draw the cluster class points and noise points in the figure
    for i in range(len(C_list)):
        for j in range(len(C_list[i])):  
            cv2.circle(img,(int(C_list[i][j][0]),int(C_list[i][j][1])), 3, (150,125+50*i,50), -1) # Center coordinates, radius, color (BGR), line width (if -1, it is the fill color)
    for i in range(len(noise)):
        cv2.circle(img,(int(noise[i][0]),int(noise[i][1])), 3, (0,0,0), -1) # Center coordinates, radius, color (BGR), line width (if -1, it is the fill color)

    # With a certain amount of staring time as the criterion, the regions that can be defined as AOI are screened out, and the limits of cluster regions are drawn on the graph, and the number of clusters is marked
    print("these Clusters are not being displayed")
    for i in range(len(C_list)):
        # Draw the bounds of the cluster area on the graph
        LeftTop = (int(left[i]),int(top[i]))
        RightBottom = (int(right[i]), int(bottom[i]))
        cv2.rectangle(img, LeftTop, RightBottom, (0, 0, 255), 3, 8)
        # And mark the number of the cluster
        cen_x=((left[i]+right[i])/2).astype(int)
        cen_y=((top[i]+bottom[i])/2).astype(int)
        cv2.putText(img, str(i), (cen_x,cen_y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2, 8, None)

    cv2.namedWindow("test")
    cv2.imshow('test', img)
    cv2.waitKey ()
    cv2.destroyAllWindows()