print("Hello, World!!!")

import numpy as np

np.random.seed()  # different each run
n_samples = 30000
n_features = 2
n_classes = 4

feature_names = ["Voltage (pu)", "Current (pu)"]
print("\n🔹 Features used as input:")
for i, f in enumerate(feature_names):
    print(f"  Feature {i+1}: {f}")

# Initialize voltage/current values (0–1 normalized)
X = np.random.rand(n_samples, n_features)

y = np.zeros(n_samples, dtype=int)
# 0 = Normal, 1 = Single Line, 2 = Line-Line, 3 = 3 phase
# 0 = Normal, 1 = Single Line, 2 = Line-Line, 3 = 3 phase
y[:] = 0  # Normal
y[(X[:,0] <= 0.4) & (X[:,1] >= 0.6)] = 1     # Single Line
y[(X[:,0] < 0.6) & (X[:,1] > 0.4)] = 2     # Line-Line
y[(X[:,0] <= 0.3) & (X[:,1] >= 0.7)] = 3     # 3 Phase

# Split dataset
train_size = int(0.8 * n_samples) #80% used for training
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]


def predict(W, X):
    logits = np.dot(X, W.T)
    return np.argmax(logits, axis=1)

def accuracy(W, X, y):
    y_pred = predict(W, X)
    return np.mean(y_pred == y)


def fitness_function(W_flat):
    W = W_flat.reshape(n_classes, n_features)
    return accuracy(W, X_train, y_train)


num_particles = 150
num_iterations = 15
dim = n_classes * n_features  # Only weights

# PSO parameters
w = 0.7   # inertia
c1 = 1.8  # cognitive coefficient
c2 = 1.2  # social coefficient

# Initialize positions (weights) and velocities
pos = np.random.uniform(-1, 1, (num_particles, dim)) #NumPy array
vel = np.zeros_like(pos)

# Initialize bests
p_best = pos.copy()
p_best_scores = np.array([fitness_function(p) for p in p_best])
g_best = p_best[np.argmax(p_best_scores)].copy()
g_best_score = np.max(p_best_scores)

# Show first weight matrix
first_W = pos[0].reshape(n_classes, n_features)
print("\n📘 Initial random weight matrix (first particle):")
print(first_W)

print("\n Starting PSO optimization\n")
for it in range(num_iterations):
    for i in range(num_particles):
        r1, r2 = np.random.rand(), np.random.rand()
        vel[i] = (w * vel[i] +
                  c1 * r1 * (p_best[i] - pos[i]) +
                  c2 * r2 * (g_best - pos[i]))
        pos[i] += vel[i]

        score = fitness_function(pos[i])
        if score > p_best_scores[i]:
            p_best[i] = pos[i].copy()
            p_best_scores[i] = score
            if score > g_best_score:
                g_best = pos[i].copy()
                g_best_score = score

    W_iter = g_best.reshape(n_classes, n_features)
    y_pred_iter = predict(W_iter, X_test)

    class_counts = np.bincount(y_pred_iter, minlength=n_classes)

    print(f"Iteration {it+1:02d}/{num_iterations} → Best Acc: {g_best_score:.4f}")
    print(f"   Predicted counts → Normal: {class_counts[0]}, Single Line: {class_counts[1]}, Line-Line: {class_counts[2]}, 3 Phase: {class_counts[3]}")

W_best = g_best.reshape(n_classes, n_features)
test_acc = accuracy(W_best, X_test, y_test)

print("\n✅ PSO Optimization Completed!")
print("-----------------------------------")
print("Best weight matrix (W):")
print(W_best)
print(f"\nFinal Test Accuracy: {test_acc:.4f}")
print("-----------------------------------")
