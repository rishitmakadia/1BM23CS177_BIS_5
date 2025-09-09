import random

# Step 1: Define the De Jong fitness function
def de_jong(position):
    x, y = position
    return x**2 + y**2

# Step 2: Particle class
class Particle:
    def __init__(self, bounds):
        self.position = [random.uniform(low, high) for low, high in bounds]
        self.velocity = [random.uniform(-1, 1) for _ in bounds]
        self.best_position = list(self.position)
        self.best_score = de_jong(self.position)

    def update_velocity(self, global_best_position, w, c1, c2):
        for i in range(len(self.velocity)):
            r1 = random.random()
            r2 = random.random()
            cognitive = c1 * r1 * (self.best_position[i] - self.position[i])
            social = c2 * r2 * (global_best_position[i] - self.position[i])
            self.velocity[i] = w * self.velocity[i] + cognitive + social

    def update_position(self, bounds):
        for i in range(len(self.position)):
            self.position[i] += self.velocity[i]
            # Clamp to bounds
            self.position[i] = max(bounds[i][0], min(self.position[i], bounds[i][1]))

        current_score = de_jong(self.position)
        if current_score < self.best_score:
            self.best_score = current_score
            self.best_position = list(self.position)

# Step 3: PSO Algorithm
def pso(
    function,
    bounds=[(-10, 10), (-10, 10)],
    num_particles=30,
    max_iter=10,
    w=0.5,       # inertia weight
    c1=1.5,      # personal influence | Cognitive Coefficient
    c2=1.5       # best influence | Social Coefficient
):
    # Initialize particles
    swarm = [Particle(bounds) for _ in range(num_particles)]

    # Initialize global best
    global_best_position = min(swarm, key=lambda p: p.best_score).best_position
    global_best_score = function(global_best_position)

    # Main loop
    for iteration in range(max_iter):
        for particle in swarm:
            particle.update_velocity(global_best_position, w, c1, c2)
            particle.update_position(bounds)

        # Update global best
        for particle in swarm:
            if particle.best_score < global_best_score:
                global_best_score = particle.best_score
                global_best_position = list(particle.best_position)

        print(f"Iteration {iteration+1}/{max_iter} - Global Best: {global_best_score:.6f}")

    print("\n✅ Optimization Complete!")
    print(f"Best Position: {global_best_position}")
    print(f"Best Score (f(x,y)): {global_best_score:.6f}")
    return global_best_position, global_best_score

# Step 4: Run the PSO
if __name__ == "__main__":
    pso(de_jong)
