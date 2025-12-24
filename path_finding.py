import numpy as np
import heapq
import pandas as pd
import data
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

data_collected = False
criteria = np.array(["Pop", "Rap", "Blues - Jazz", "Country - Rock - Métal",
                     "Disco, House, Electro, Techno", "Chill - Energique Agressif"]) 

# -- VARIABLES INITIALIZATION --

d = len(criteria) #Total number of criteria
n = data.n  # Total number of musics
occurences = np.zeros(n) #Array to store the number of occurrences of each music in user responses
scores = np.array([[0.0]*data.d]*n)  # Matrix (n*d) to store scores associated with criteria

if data_collected:
    df = pd.read_csv('responses.csv')
    total_scores = df.iloc[1:, 1:].to_numpy()
    for (i, L) in enumerate(total_scores):
        idx = data.music_dict[df.iloc[i+1][0]]
        scores[idx] += L
        occurences[idx] += 1
    for i in range(n):
        if occurences[i] > 0:
            scores[i] /= occurences[i]

    # Distances matrix computation
    distances = np.array([[0.0]*n]*n) #Matrix (n*n) to store distances between each pair of musics   
    for i in range(n-1):
        for j in range(i+1, n):
            dist = np.linalg.norm(scores[i] - scores[j]) #Calculation of the Euclidean distance between musics i and j
            distances[i][j] = dist
            distances[j][i] = dist #The distance is symmetric


# -- FUNCTIONS --

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def heuristic(depart, arrivee):
    return lambda x: 1.2*d + np.dot(normalize(scores[x] - scores[depart]), scores[x] - scores[arrivee])


def k_nearest(music_index, k):
    music_distances = distances[music_index]
    sorted_indices = np.argsort(music_distances)
    return sorted_indices[1:k+1]  # Exclude the music's own index
    
def find_path(depart, arrivee):
    d = distances[depart][arrivee]
    min_step = d / 20
    max_step = d / 5

    optimal_path = []
    total_distance = 0
    current_music = depart
    optimal_path.append(current_music)

    while current_music != arrivee:
        possible_neighbors = []
        for i in range(n):
            if i != current_music:
                dist = distances[current_music][i]
                if min_step <= dist  and dist <= max_step:
                    possible_neighbors.append((i, dist))

        if not possible_neighbors:
            print("No possible path with the given constraints.")
            return []

        # Choose the neighbor closest to the destination
        chosen_neighbor = min(possible_neighbors, key=lambda x: distances[x[0]][arrivee])
        current_music = chosen_neighbor[0]
        total_distance += chosen_neighbor[1]
        optimal_path.append(current_music)

    print(f"Total distance traveled: {total_distance}")
    return optimal_path

def find_optimal_path(depart, arrivee):
    """
    Implements the A* algorithm to find the optimal path. 
    The considered paths are those whose distances are between d/20 and d/5,
    h: heuristic function h(node) that returns an estimation of the distance from node to arrivee.
    """
    if depart == arrivee:
        return [depart]
    
    h = heuristic(depart, arrivee)
    d = distances[depart][arrivee]
    min_step = d / 20
    max_step = d / 5
    
    # Frontier : priority queue with (f_score, node)
    frontier = []
    heapq.heappush(frontier, (0, depart))
    
    came_from = {depart: None}
    g_score = {i: float('inf') for i in range(n)}
    g_score[depart] = 0
    f_score = {i: float('inf') for i in range(n)}
    f_score[depart] = h(depart)
    
    while frontier:
        current_f, current = heapq.heappop(frontier)
        
        if current == arrivee:
            # Reconstruct the path
            path = []
            while current is not None:
                path.append(current)
                current = came_from[current]
            path.reverse()
            return path
        
        # Find valid neighbors
        for neighbor in range(n):
            if neighbor != current:
                dist = distances[current][neighbor]
                if min_step <= dist <= max_step:
                    tentative_g = g_score[current] + dist
                    if tentative_g < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g
                        f_score[neighbor] = tentative_g + h(neighbor)
                        heapq.heappush(frontier, (f_score[neighbor], neighbor))
    
    return []  # No path found

def find_optimal_path_bis(depart, arrivee):
    """
    Implements the A* algorithm to find the optimal path. 
    The considered paths are those whose distances are between d/20 and d/5,
    h: heuristic function h(node) that returns an estimation of the distance from node to arrivee.
    """
    if depart == arrivee:
        return [depart]
    h = heuristic(depart, arrivee)
    d = distances[depart][arrivee]
    f = lambda x: 3 * (x - d/10)**2 #constant TO ADJUST
    distances_bis = [[f(distances[i][j]) for j in range(n)] for i in range(n)]
    
    # Frontier: priority queue with (f_score, node)
    frontier = []
    heapq.heappush(frontier, (0, depart))
    
    came_from = {depart: None}
    g_score = {i: float('inf') for i in range(n)}
    g_score[depart] = 0
    f_score = {i: float('inf') for i in range(n)}
    f_score[depart] = h(depart)
    
    while frontier:
        current_f, current = heapq.heappop(frontier)
        
        if current == arrivee:
            # Reconstruct the path
            path = []
            while current is not None:
                path.append(current)
                current = came_from[current]
            path.reverse()
            return path
        
        # Find valid neighbors
        for neighbor in range(n):
            if neighbor != current:
                dist = distances_bis[current][neighbor]
                tentative_g = g_score[current] + dist
                if tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + h(neighbor)
                    heapq.heappush(frontier, (f_score[neighbor], neighbor))
    
    return []  # No path found


# -- test function --
def tests():
    print(f"number of musics: {n}")
    print(f"number of criteria: {d}")

    closest = k_nearest(5, 16)
    print("The musics closest to 'All Star' are:")
    for idx in closest:
        print(f"- {data.titles[idx]}")

    path = find_optimal_path(5, 11)
    for idx in path:
        print(f"- {data.titles[idx]}")




# Display principal components in terms of original features
def display_pca_results():

    pca = PCA(n_components=2)
    scores_2d = pca.fit_transform(scores)

    for i, component in enumerate(pca.components_):
        print(f"Principal Component {i+1}:")
        for j, coeff in enumerate(component):
            print(f"  {coeff:.3f} * {criteria[j]}")
        print()

    plt.figure(figsize=(10, 7))
    plt.scatter(scores_2d[:, 0], scores_2d[:, 1])
    for i, title in enumerate(data.titles):
        plt.annotate(title, (scores_2d[i, 0], scores_2d[i, 1]))
    plt.title("2D representation of musics after PCA")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid()
    plt.show()