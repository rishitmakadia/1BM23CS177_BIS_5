import numpy as np
import random

h, w = 128, 128

image = np.zeros((h, w), dtype=np.float32)
image[32:96, 32:96] = 1.0 
image[48:80, 48:80] = 0.0  
K = 80 # no of ants
N = 15 # no of ite
M = 8 # random walkers 
rho = 0.1
alpha = 1.0
beta = 2.0
threshold = 0.05
Q = 1.0 

pheromone = np.ones((h, w), dtype=np.float32) * 0.01

visited = np.zeros((h, w), dtype=np.uint8)

gx = np.zeros_like(image)
gy = np.zeros_like(image)

gx[:, 1:-1] = image[:, 2:] - image[:, :-2]  
gy[1:-1, :] = image[2:, :] - image[:-2, :]  

heuristic = np.sqrt(gx * gx + gy * gy) + 1e-6  

neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1),
             (-1, -1), (-1, 1), (1, -1), (1, 1)]

for n in range(N):
    step_visits = 0 
    for m in range(M):
        for k in range(K):  
            x = random.randint(1, h - 2)
            y = random.randint(1, w - 2)

            probs = []  
            coords = []  

            for dx, dy in neighbors:
                nx, ny = x + dx, y + dy
                if abs(image[nx, ny] - image[x, y]) > threshold:
                    p = (pheromone[nx, ny] ** alpha) * (heuristic[nx, ny] ** beta)
                    probs.append(p)
                    coords.append((nx, ny))

            if not probs:
                continue  

            probs = np.array(probs)
            probs /= probs.sum()

            idx = np.random.choice(len(coords), p=probs)
            x, y = coords[idx]

            pheromone[x, y] = (1 - rho) * pheromone[x, y] + rho * heuristic[x, y]
            visited[x, y] = 1  
            step_visits += 1  

    pheromone /= pheromone.max()

    print(f"Iteration {n+1}/{N} completed | Visited pixels: {step_visits}")

edges = (pheromone > pheromone.mean()).astype(np.uint8)

print("\nEdge detection finished")
print("Total edge pixels detected:", edges.sum())
