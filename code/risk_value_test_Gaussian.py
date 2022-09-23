#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import cv2
from os import walk
import numpy as np
import matplotlib.pyplot as plt
import time
import heapq
from datetime import timedelta
from sklearn.cluster import AgglomerativeClustering
import multiprocessing
import sys
import random

CORE_NUM = 16
prediction_horizon_list = [10, 20, 30, 40, 50, 60]  # in minutes
SOG_threshold = [2, 50]

initial_sigma = int(sys.argv[1])
a_sigma = float(sys.argv[2])
print("initial_sigma: ", initial_sigma, ", a_sigma: ", a_sigma)

# In[2]:


class Vertex():  # for PRM
    def __init__(self, idx, x, y):
        self.idx = idx
        self.x = x
        self.y = y

class Node():  # for tree
    def __init__(self, idx, x, y, parent_idx=None):
        self.idx = idx
        self.x = x
        self.y = y
        self.parent_idx = parent_idx
        

def not_connected(V, E, vertex):
    V_num = len(V.keys())
    connected_set = set()
    queue = [vertex]
    while queue:
        idx = queue.pop()
        connected_set.add(idx)
        for neighbor_idx in E[idx]:
            if neighbor_idx not in connected_set:
                queue.append(neighbor_idx)
    
    return connected_set ^ set(V.keys())
                
def path_good(chart, pos0, pos1, increment):
    x0, y0 = pos0
    x1, y1 = pos1
    dist = np.sqrt((x1 - x0)**2 + (y1 - y0)**2)
    direction = [(x1-x0)/dist, (y1-y0)/dist]
    for i in range(1, int(dist//increment), increment):
        if chart[int(y0+direction[1]*i*increment)][int(x0+direction[0]*i*increment)][0] > 0:
            return False
    
    return True

def geo_to_pixel(chart, geo_position, extent):
    lon_0, lon_1, lat_0, lat_1 = extent
    
    x_geo = geo_position[0]
    y_geo = geo_position[1]
    x_geo_range = lon_1 - lon_0
    y_geo_range = lat_1 - lat_0
    
    x_pixel = int((x_geo - lon_0) / x_geo_range * chart.shape[1])
    y_pixel = int((lat_1 - y_geo) / y_geo_range * chart.shape[0])
    
    pos_in_pixel = (x_pixel, y_pixel)
    return pos_in_pixel

def get_phi(x0, y0, x1, y1, l_local):
    x_disp = (x1 - x0)
    if x_disp > l_local:
        x_disp = l_local
    if x_disp < - l_local:
        x_disp = - l_local

    if y1 - y0 > 0:
        phi = np.arccos(x_disp / l_local)
    else:
        phi = 2*np.pi - np.arccos(x_disp / l_local)
    return phi

def draw_path(chart, path, color=None, width=None):
    if not color:
        color = (0, 255, 0)
    if not width:
        width = 5
    for i in range(len(path) - 1):
        start_point = path[i]
        end_point = path[i+1]
        cv2.line(chart, start_point, end_point, color, width)

def draw_ship(chart, pos, course_in_degree, scale, color):
    x, y = pos
    phi = (course_in_degree - 90) * np.pi / 180
    ship_contour = scale * np.matrix([[30, 0], 
                                      [10, -10], 
                                      [-30, -10], 
                                      [-30, 10], 
                                      [10, 10]]).T
    rot = np.matrix([[np.cos(phi), -np.sin(phi)], 
                     [np.sin(phi), np.cos(phi)]])
    ship_contour = np.round(np.matmul(rot, ship_contour)).astype(int)
    
    for i in range(5):
        ship_contour[0, i] += x
        ship_contour[1, i] += y
        
    cv2.fillPoly(chart, pts=[ship_contour.T], color=color)
    
    return chart

def draw_square(chart, pos, scale, color):
    x, y = pos
    ship_contour = scale * np.matrix([[20, -20], 
                                      [-20, -20], 
                                      [-20, 20], 
                                      [20, 20]]).T
    
    for i in range(4):
        ship_contour[0, i] += x
        ship_contour[1, i] += y
        
    cv2.fillPoly(chart, pts=[ship_contour.T], color=color)
    
    return chart


# In[3]:


directory = "../nautical_chart/"
df_description = pd.read_csv(directory+"chart_detail.csv")
print(df_description)

resolution, x_km_per_deg, y_km_per_deg, lon_0, lon_1, lat_0, lat_1 = df_description.loc[0, :]
extent = (lon_0, lon_1, lat_0, lat_1)


# # good test cases (21:00)
# 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 29, 30, 31
# 
# 4: good: 21:56:57 to 22:59:56 <br>
# 5: simple: 21:03:23 to 22:46:24 <br>
# 6: good: 21:48:03 to 22:23:13 <br>
# 12: following another ship: 21:00:25 to 22:29:55 <br>
# 19: good: 22:06:13 to 22:52:11 <br>
# 25: good: 22:12:47 to 22:50:18 <br>

# In[4]:


# read test cases
test_cases = {4: ["21:56:57", "22:59:56"], 
              5: ["21:03:23", "22:46:24"], 
              6: ["21:48:03", "22:23:13"], 
              12: ["21:00:25", "22:29:55"], 
              19: ["22:06:13", "22:52:11"], 
              25: ["22:12:47", "22:50:18"]}
date = 4
begin = test_cases[date][0]
end = test_cases[date][1]

# read test cases
test_case_path = "../test_cases/"
df_test_cases = pd.read_csv(test_case_path + "test_cases.csv")
own_mmsi = int(df_test_cases.loc[date-1, "own_mmsi"])
if date < 10:
    filename = "AIS_2020_07_0" + str(date) + ".csv"
else:
    filename = "AIS_2020_07_" + str(date) + ".csv"
csv_directory = "../processed_data/"
df = pd.read_csv(csv_directory + filename)


# In[5]:


# collect the latest ship positions (SOG < 0.5) in the past 10 minutes

chart_path = "../nautical_chart/res_100_lon_-122.67_-122.22_lat_37.54_38.17.jpg"
chart = cv2.imread(chart_path)

df = pd.read_csv(csv_directory + filename)

t_end = pd.to_datetime('2020-07-' + filename[-6:-4] + ' ' + begin)
t_start = t_end - timedelta(minutes=10)

ship_dict = {}
for i in range(df.shape[0]):
    MMSI, BaseDateTime, LAT, LON, SOG, COG, Heading = df.loc[i, :]
    if t_start <= pd.to_datetime(BaseDateTime) <= t_end:
        if MMSI not in ship_dict:
            ship_dict[MMSI] = []
        ship_dict[MMSI].append([BaseDateTime, LAT, LON, SOG, COG])

for key in ship_dict:
    ship_dict[key].sort(key=lambda x : x[0])


ships_10min = {}
for mmsi in ship_dict:
    for (BaseDateTime, LAT, LON, SOG, COG) in ship_dict[mmsi]:
        if SOG < 0.5:
            if mmsi not in ships_10min:
                ships_10min[mmsi] = []
            pos = geo_to_pixel(chart, [LON, LAT], extent)
            ships_10min[mmsi] = [pos[0], pos[1], COG]

to_be_clustered = []
for mmsi in ships_10min:
    to_be_clustered.append(ships_10min[mmsi][:2])


# In[6]:


# perform hierarchical clustering and display all the clusters by color

display_each = False
# n_clusters = 20
# clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage="complete").fit(to_be_clustered)
clustering = AgglomerativeClustering(n_clusters=None, linkage="complete", distance_threshold=400, compute_full_tree=True).fit(to_be_clustered)


ship_mmsi_list = list(ships_10min.keys())
cm = plt.get_cmap('gist_rainbow')

if display_each:
    for target_label in range(clustering.n_clusters_):
        image = chart.copy()
        for i in range(len(ship_mmsi_list)):
            mmsi = ship_mmsi_list[i]
            pos = ships_10min[mmsi][:2]
            COG = ships_10min[mmsi][2]
            label = clustering.labels_[i]
            if label != target_label:
                continue
            color = [int(x*255) for x in cm(label/n_clusters)[:3]]

            image = draw_ship(image, pos, COG, scale=3, color=color)

        image = cv2.resize(image, [image.shape[1] // 10, image.shape[0] // 10])
        cv2.imshow("image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

else:
    image = chart.copy()
    for i in range(len(ship_mmsi_list)):
        mmsi = ship_mmsi_list[i]
        pos = ships_10min[mmsi][:2]
        COG = ships_10min[mmsi][2]
        label = clustering.labels_[i]
        color = [int(x*255) for x in cm(label/clustering.n_clusters_)[:3]]

        image = draw_ship(image, pos, COG, scale=3, color=color)

#     image = cv2.resize(image, [image.shape[1] // 10, image.shape[0] // 10])
#     cv2.imshow("image", image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()


# In[7]:


# get the median position of each cluster
clusters = {}
for i in range(len(ship_mmsi_list)):
    mmsi = ship_mmsi_list[i]
    label = clustering.labels_[i]
    if label not in clusters:
        clusters[label] = []
    clusters[label].append(ships_10min[mmsi][0:2])

cluster_medians = []
for label in range(clustering.n_clusters_):
    mean = [0, 0]
    for pos in clusters[label]:
        mean[0] += pos[0]
        mean[1] += pos[1]
    mean[0] /= len(clusters[label])
    mean[1] /= len(clusters[label])
    
    median = []
    min_dist_square = 1000000
    for pos in clusters[label]: 
        if (pos[0] - mean[0])**2 + (pos[1] - mean[1])**2 < min_dist_square:
            min_dist_square = (pos[0] - mean[0])**2 + (pos[1] - mean[1])**2
            median = pos.copy()
    cluster_medians.append(median)

image = chart.copy()
for i in range(len(ship_mmsi_list)):
    mmsi = ship_mmsi_list[i]
    pos = ships_10min[mmsi][:2]
    COG = ships_10min[mmsi][2]
    label = clustering.labels_[i]
    color = [int(x*255) for x in cm(label/clustering.n_clusters_)[:3]]
    if pos in cluster_medians:
#         color = [0, 0, 255]
        image = draw_ship(image, pos, COG, scale=3, color=color)

# image = cv2.resize(image, [image.shape[1] // 10, image.shape[0] // 10])
# cv2.imshow("image", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

print("cluster medians: ", cluster_medians)


# In[8]:

def get_risk_values(seed):
    random.seed(seed)
    np.random.seed(seed)
    
    final_predict_risk_list = {10:[], 20:[], 30:[], 40:[], 50:[], 60:[]}  # {10:[], 20:[], 30:[]}
    final_non_predict_risk_list = {10:[], 20:[], 30:[], 40:[], 50:[], 60:[]}

    # use PRM to generate a road map
    chart_path = "../nautical_chart/res_100_lon_-122.67_-122.22_lat_37.54_38.17.jpg"
    chart = cv2.imread(chart_path)

    # PRM_save_path = "../draft/figures/generated_PRM/"

    V = {}  # idx: Vertex()
    E = {}  # idx: [neighbor_idx]
    V_count = 0
    max_V_num = 15000
    max_neighbor_num = 16
    max_course_change = np.pi/4  # rad

    # initialize vertices for targets
    targets_in_ocean = [[50, 3200], [50, 5000], [50, 6800], [1200, 6800], [3800, 6500], [3800, 5300], [3950, 1200], [3300, 100]]

    # targets = [[1641, 3443], [3132, 3678], [2713, 4017], [2679, 2837], [2386, 4087], [3353, 4218], [1865, 3692], [3078, 3960], [2548, 4687], [2495, 4324], [2953, 4590], [1500, 2210], [1987, 4047], [2275, 2754], [1721, 2902], [1189, 6926], [3756, 1233], [3522, 766], [3562, 4288], [3093, 6450], [2202, 3479], [2921, 4809]]
    targets = cluster_medians + targets_in_ocean
    for target in targets:  
        # if the ship position is not in the water on the chart (due to chart accuracy)
        if chart[target[1]][target[0]][0] > 0:  
            while True:
                dx = int(np.random.rand() * 100 - 50)
                dy = int(np.random.rand() * 100 - 50)
                if chart[target[1] + dy][target[0] + dx][0] == 0:
                    target[0] += dx
                    target[1] += dy
                    break

        V[V_count] = Vertex(V_count, target[0], target[1])
        V_count += 1

    t = time.time()
            
    # generate PRM
    while V_count < max_V_num:
        x = int(np.random.rand() * chart.shape[1])
        y = int(np.random.rand() * chart.shape[0])
        if chart[y][x][0] > 0:
            continue
        V[V_count] = Vertex(V_count, x, y)
        V_count += 1

    for i in range(V_count):
        E[i] = set()

    for vertex in range(V_count):
        if len(E[vertex]) >= max_neighbor_num:
            continue
        dist_heap = []
        x0, y0 = V[vertex].x, V[vertex].y
        for neighbor in range(V_count):
            x1, y1 = V[neighbor].x, V[neighbor].y
            dist = np.sqrt((x1 - x0)**2 + (y1 - y0)**2)
            if dist < 1:  # if at the same pixel position
                continue
            
            if len(dist_heap) < max_neighbor_num:
                heapq.heappush(dist_heap, [-dist, neighbor])
            else:
                heapq.heappushpop(dist_heap, [-dist, neighbor])
        
        while len(E[vertex]) < max_neighbor_num and dist_heap:
            neg_dist, neighbor = heapq.heappop(dist_heap)
            x1, y1 = V[neighbor].x, V[neighbor].y
            if path_good(chart, [x0, y0], [x1, y1], 2):
                E[vertex].add(neighbor)
                E[neighbor].add(vertex)

            
    # print("time consumed: ", time.time() - t)


    # In[9]:


    # save the generated roadmap
    # image = chart.copy()

    # for vertex in range(V_count):
    #     x0, y0 = V[vertex].x, V[vertex].y
    #     for neighbor in E[vertex]:
    #         x1, y1 = V[neighbor].x, V[neighbor].y
    #         draw_path(image, [[x0, y0], [x1, y1]], color=[0, 0, 255])

    # cv2.imwrite(PRM_save_path + str(date) + "/PRM_" + str(max_V_num) + "_" + str(max_neighbor_num) + ".jpg", image)


    # In[9]:


    # from the road map, generate a tree for each target by Dijkstra's algorithm (modified cost function)

    coef0 = 1000
    coef1 = 0.1
    p_threshold = 1000000  # not activated

    def get_cost(dist_sum, p_sum, p_max):
        dist_cost = dist_sum
        turn_cost = (p_max + coef1 * p_sum)
        return dist_cost + coef0 * turn_cost


    tree_list = []  # list of node_dict
    for t_idx in range(len(targets)):
        if len(not_connected(V, E, t_idx)) > 0.5 * V_count:  # if the target is not connected, find a nearby one as the new target
            x0, y0 = V[t_idx].x, V[t_idx].y
            dist_heap = []
            for i in range(V_count):
                x1, y1 = V[i].x, V[i].y
                heapq.heappush(dist_heap, [np.sqrt((x1-x0)**2+(y1-y0)**2), i])
            while dist_heap:
                dist, idx = heapq.heappop(dist_heap)
                if len(not_connected(V, E, t_idx)) < 0.1 * V_count:
                    target = [V[idx].x, V[idx].y]
                    break
        else:
            target = targets[t_idx]
            idx = t_idx
            
        node_heap = [[0, 0, 0, 0, idx, None, None, None]]  # heap of [cost, dist_sum, p_sum, p_max, vertex_idx, parent_idx, phi, l] 
        node_dict = {}  # {node_idx: Node()}
        visited = set()
        
        while node_heap:
            cost, dist_sum, p_sum, p_max, vertex_idx, parent_idx, phi_parent, l_parent = heapq.heappop(node_heap)
            if vertex_idx in visited:
                continue
            visited.add(vertex_idx)
            x0, y0 = V[vertex_idx].x, V[vertex_idx].y
            node_dict[vertex_idx] = Node(vertex_idx, x0, y0, parent_idx)
            
            for neighbor in E[vertex_idx]:
                x1, y1 = V[neighbor].x, V[neighbor].y
                l_local = np.sqrt((x1-x0)**2+(y1-y0)**2)  # edge length
                phi = get_phi(x0, y0, x1, y1, l_local)
                
                # calculate the cost function
                p_local = 0
                if phi_parent != None:
                    p_local = abs(np.tan(0.5 * (phi - phi_parent))) / min(l_local, l_parent)
                
                # do not allow sharp turn
                if p_local > p_threshold:
                    continue
                
                cost = get_cost(dist_sum+l_local, p_sum+p_local, max(p_max,p_local))
                heapq.heappush(node_heap, [cost, dist_sum+l_local, p_sum+p_local, max(p_max,p_local), neighbor, vertex_idx, phi, l_local])
        
        tree_list.append(node_dict)
        
        image = chart.copy()

        for vertex in node_dict:
            x0, y0 = node_dict[vertex].x, node_dict[vertex].y
            if node_dict[vertex].parent_idx != None:
                parent = node_dict[vertex].parent_idx
                x1, y1 = node_dict[parent].x, node_dict[parent].y
                draw_path(image, [[x0, y0], [x1, y1]], color=[0, 0, 255])
        
        image = draw_square(image, target, scale=2, color=[0, 255, 0])

    #     cv2.imwrite(PRM_save_path + str(date) + "/PRM_tree_" + str(idx) + "_" + str(coef0) + "_" + str(coef1) + ".jpg", image)


    # In[10]:


    def get_pos_prediction(path, arrival_time, path_start_time, cur_time):
        t = (cur_time - path_start_time).seconds
        if t >= arrival_time[-1]:
            return [path[-1][0], path[-1][1]]
        
        i, j = 0, len(arrival_time) - 1
        while i < j:
            mid = (i+j) // 2
            if arrival_time[mid] < t:
                i = mid + 1
            else:
                j = mid
        
        x0, y0 = path[i-1]
        x1, y1 = path[i]
        x = x0 + (x1-x0) * (t - arrival_time[i-1]) / (arrival_time[i] - arrival_time[i-1])
        y = y0 + (y1-y0) * (t - arrival_time[i-1]) / (arrival_time[i] - arrival_time[i-1])
        return [x, y]
        
        


    # In[11]:


    def get_path(pos, COG, node_dict, V):
        own_phi = COG*np.pi/180 - np.pi/2
        path = [[pos[0], pos[1]]]
        min_cost = 100000000
        closest_node_idx = None
        for i in node_dict:
            x, y = node_dict[i].x, node_dict[i].y
            dist = np.sqrt((x-pos[0])**2 + (y-pos[1])**2)
            if dist == 0:
                closest_node_idx = i
                break
            phi = get_phi(pos[0], pos[1], x, y, dist)
            p_local = abs(np.tan(0.5 * (phi - own_phi))) # / l_local
            cost = get_cost(dist, p_local, p_local)
            if cost < min_cost:
                min_cost = cost
                closest_node_idx = i
        
        cur_node = node_dict[closest_node_idx]
        x, y = cur_node.x, cur_node.y
        path.append([x, y])
        while cur_node.parent_idx != None:
            cur_node = node_dict[cur_node.parent_idx]
            x, y = cur_node.x, cur_node.y
            path.append([x, y])
        return path

    def get_weight(pos, COG, path, dist_threshold=None):
        course = COG * np.pi / 180 - np.pi/2  # transfer the COG to the image coordinate
        if not dist_threshold:
            dist_threshold = 5000
        dist_sum = 0
        for i in range(1, len(path)):
            x0, y0 = path[i-1]
            x1, y1 = path[i]
            dist_sum += np.sqrt((x1-x0)**2 + (y1-y0)**2)
            if dist_sum > dist_threshold:
                break
        
        x, y = pos
        xd, yd = path[i]
        dist_d = np.sqrt((xd-x)**2 + (yd-y)**2)
        
        phi = get_phi(x, y, xd, yd, dist_d)
        
        return 7 * np.cos(phi-course)
        


    # In[13]:


    

    # read chart
    chart_path = "../nautical_chart/res_100_lon_-122.67_-122.22_lat_37.54_38.17.jpg"
    chart = cv2.imread(chart_path)

    save_path = "../risk_test/"


    # initial_sigma = 10
    # a_sigma = 0.02
	
        
    # read test cases
    test_case_path = "../test_cases/"
    df_test_cases = pd.read_csv(test_case_path + "test_cases.csv")
    own_mmsi = int(df_test_cases.loc[date-1, "own_mmsi"])
    if date < 10:
        filename = "AIS_2020_07_0" + str(date) + ".csv"
    else:
        filename = "AIS_2020_07_" + str(date) + ".csv"
    csv_directory = "../processed_data/"
    df = pd.read_csv(csv_directory + filename)


    # define start and end time, and read AIS data in this time period
    start_time = pd.to_datetime('2020-07-' + filename[-6:-4] + ' ' + begin) 
    end_time = pd.to_datetime('2020-07-' + filename[-6:-4] + ' ' + end)

    ship_dict = {}  # ship_dict[MMSI]: list of [BaseDateTime, LAT, LON, SOG, COG]
    for i in range(df.shape[0]):
        MMSI, BaseDateTime, LAT, LON, SOG, COG, Heading = df.loc[i, :]
        if start_time <= pd.to_datetime(BaseDateTime) <= end_time + timedelta(hours=1):
            if MMSI not in ship_dict:
                ship_dict[MMSI] = []
            ship_dict[MMSI].append([BaseDateTime, LAT, LON, SOG, COG])
    for key in ship_dict:
        ship_dict[key].sort(key=lambda x : x[0])


    def get_start_end(records):  # find the starting position and the position after the prediction horizon of each ship
        result = []
        for i in range(len(records) - 1):
            BaseDateTime0, LAT0, LON0, SOG0, COG0 = records[i]
    #         if SOG0 < SOG_threshold:
    #             continue
            if not SOG_threshold[0] <= SOG0 < SOG_threshold[1]:
                continue

            found = False
            for j in range(i, len(records)):
                BaseDateTime1, LAT1, LON1, SOG1, COG1 = records[j]
                if pd.to_datetime(BaseDateTime1) - pd.to_datetime(BaseDateTime0) > timedelta(minutes=prediction_horizon):
                    found = True
                    break
            if found:
                result = [[BaseDateTime0, LAT0, LON0, SOG0, COG0], [BaseDateTime1, LAT1, LON1, SOG1, COG1]]
                break
        return result
        
    
    for prediction_horizon in prediction_horizon_list:
        non_predict_risk_list = []
        predict_risk_list = []
        for mmsi in ship_dict:
            records = ship_dict[mmsi]
            start_end_set = get_start_end(records)
            if not start_end_set:
                continue
            BaseDateTime, LAT, LON, SOG, COG = start_end_set[0]
            BaseDateTime1, LAT1, LON1, SOG1, COG1 = start_end_set[1]

            possible_path = {}
            if mmsi not in possible_path:
                possible_path[mmsi] = {}

            possible_path[mmsi]["time"] = BaseDateTime

            pos = geo_to_pixel(chart, [LON, LAT], extent)
            possible_path[mmsi]["pos"] = pos
            possible_path[mmsi]["SOG"] = SOG
            possible_path[mmsi]["COG"] = COG

            possible_path[mmsi]["path_list"] = []
            possible_path[mmsi]["arrival_time_list"] = []
            possible_path[mmsi]["weight_list"] = []
            for tree in tree_list:  # the root of the tree is the position of the possible destination
                ship_path = get_path(pos, COG, tree, V)
                possible_path[mmsi]["path_list"].append(ship_path)
                arrival_time_list = [0]
                for i in range(1, len(ship_path)):
                    x0, y0 = ship_path[i-1]
                    x1, y1 = ship_path[i]
                    edge_dist = np.sqrt((x1-x0)**2 + (y1-y0)**2)
                    arrival_time = arrival_time_list[i-1] + edge_dist / (SOG * 0.514444 * resolution / 1000)
                    arrival_time_list.append(arrival_time)
                possible_path[mmsi]["arrival_time_list"].append(arrival_time_list)
                possible_path[mmsi]["weight_list"].append(get_weight(pos, COG, ship_path))

            weight_sum = sum([np.exp(w) for w in possible_path[mmsi]["weight_list"]])
            for i, w in enumerate(possible_path[mmsi]["weight_list"]):
                possible_path[mmsi]["weight_list"][i] = np.exp(w) / weight_sum


            x1, y1 = geo_to_pixel(chart, [LON1, LAT1], extent)


            # calculate the risk value at final_pos WITH prediction
            path_list = possible_path[mmsi]["path_list"]
            arrival_time_list = possible_path[mmsi]["arrival_time_list"]
            weight_list = possible_path[mmsi]["weight_list"]
            risk_target = 0

            dt = (pd.to_datetime(BaseDateTime1) - pd.to_datetime(BaseDateTime)).seconds
            sigma = initial_sigma + a_sigma * dt

            for i in range(len(path_list)):
                path = path_list[i]
                arrival_time = arrival_time_list[i]
                weight = weight_list[i]

                x_target, y_target = get_pos_prediction(path, arrival_time, pd.to_datetime(BaseDateTime), pd.to_datetime(BaseDateTime1))

                dist_target = np.sqrt((x_target-x1)**2 + (y_target-y1)**2)
        #         if dist_target == 0:
        #             risk_target += 10000000
        #         else:
        #             risk_target += weight / dist_target

                risk_target += weight * np.exp(-0.5*(dist_target/sigma)**2) / sigma 
            predict_risk_list.append(risk_target)


            # calculate the risk value at final_pos WITHOUT prediction
            speed = (SOG * 0.514444 * resolution / 1000)
            phi = COG*np.pi/180 - np.pi/2

            delta_time = (pd.to_datetime(BaseDateTime1) - pd.to_datetime(BaseDateTime)).seconds
            x_target = pos[0] + np.cos(phi) * speed * delta_time
            y_target = pos[1] + np.sin(phi) * speed * delta_time
            dist_target = np.sqrt((x_target-x1)**2 + (y_target-y1)**2)
            risk_target = np.exp(-0.5*(dist_target/sigma)**2) / sigma

            non_predict_risk_list.append(risk_target)
        
        final_predict_risk_list[prediction_horizon] += predict_risk_list
        final_non_predict_risk_list[prediction_horizon] += non_predict_risk_list
    return final_predict_risk_list, final_non_predict_risk_list		


final_predict_risk_list = {10:[], 20:[], 30:[], 40:[], 50:[], 60:[]}
final_non_predict_risk_list = {10:[], 20:[], 30:[], 40:[], 50:[], 60:[]}
 
pool = multiprocessing.Pool(CORE_NUM)
results = pool.map(func=get_risk_values, iterable=list(range(10)))
pool.close()
pool.join()

for result in results:
    d1, d2 = result
    for key in d1:
        # print(len(d1[key]))
        final_predict_risk_list[key] += d1[key]
        final_non_predict_risk_list[key] += d2[key]

try:
    df_result = pd.read_csv("risk_analysis.csv")
except:
    columns = ["horizon_minutes", "SOG_min", "SOG_max", 
            "sigma_0", "sigma_1", 
            "my_risk_mean", "my_risk_std", 
            "non_pred_risk_mean", "non_pred_std",]
    df_result = pd.DataFrame(columns=columns)
 
for prediction_horizon in prediction_horizon_list:
    idx = df_result.shape[0]
    df_result.loc[idx, "horizon_minutes"] = prediction_horizon
    df_result.loc[idx, "SOG_min"] = SOG_threshold[0]
    df_result.loc[idx, "SOG_max"] = SOG_threshold[1]
    df_result.loc[idx, "sigma_0"] = initial_sigma
    df_result.loc[idx, "sigma_1"] = a_sigma
    df_result.loc[idx, "my_risk_mean"] = np.mean(final_predict_risk_list[prediction_horizon])
    df_result.loc[idx, "my_risk_std"] = np.std(final_predict_risk_list[prediction_horizon])
    df_result.loc[idx, "non_pred_risk_mean"] = np.mean(final_non_predict_risk_list[prediction_horizon])
    df_result.loc[idx, "non_pred_std"] = np.std(final_non_predict_risk_list[prediction_horizon])
    df_result.to_csv("risk_analysis.csv", index_label=False, index=False)
     
    filename_pred = "risk_value_results/" + "_".join([str(initial_sigma), str(a_sigma), str(prediction_horizon), "pred"]) + ".txt"
    with open(filename_pred, 'w') as f:
        for risk_value in final_predict_risk_list[prediction_horizon]:
            f.write(str(risk_value) + "\n")
            
    filename_non_pred = "risk_value_results/" + "_".join([str(initial_sigma), str(a_sigma), str(prediction_horizon), "non_pred"]) + ".txt"
    with open(filename_non_pred, 'w') as f:
        for risk_value in final_non_predict_risk_list[prediction_horizon]:
            f.write(str(risk_value) + "\n")

