import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import random

# Reinforcement Learning libraries
import gym
from stable_baselines3 import PPO  # DRL model

# Genetic Algorithm
from deap import base, creator, tools, algorithms

# Load datasets
orders_df = pd.read_csv('./orders.csv')
shelves_df = pd.read_csv('./shelves.csv')
robots_df = pd.read_csv('./robots.csv')

# Create a NumPy array for shelf locations
shelves_locations = shelves_df[['Location_X', 'Location_Y']].to_numpy()

# Define the Distance Matrix Function
def shelf_distance(s1, s2):
    if not (0 <= s1 < len(shelves_locations)) or not (0 <= s2 < len(shelves_locations)):
        raise IndexError(f"Invalid index: s1={s1}, s2={s2} for shelves_locations with size {len(shelves_locations)}")
    return np.linalg.norm(shelves_locations[s1] - shelves_locations[s2])

# Create a distance matrix
num_shelves = len(shelves_locations)
distance_matrix = np.zeros((num_shelves, num_shelves))

for i in range(num_shelves):
    for j in range(num_shelves):
        distance_matrix[i][j] = shelf_distance(i, j)

# Implement the Ant Colony Optimization Class
class AntColony:
    def __init__(self, distance_matrix, n_ants, n_best, n_iterations, decay, alpha=1, beta=1):
        self.distances = distance_matrix
        self.pheromone = np.ones(self.distances.shape) / len(distance_matrix)
        self.all_inds = range(len(distance_matrix))
        self.n_ants = n_ants
        self.n_best = n_best
        self.n_iterations = n_iterations
        self.decay = decay
        self.alpha = alpha
        self.beta = beta

    def run(self):
        all_time_shortest_path = ("placeholder", np.inf)
        for _ in range(self.n_iterations):
            all_paths = self.gen_all_paths()
            self.spread_pheromone(all_paths, self.n_best)
            shortest_path = min(all_paths, key=lambda x: x[1])  # x[1] is the distance
            if shortest_path[1] < all_time_shortest_path[1]:
                all_time_shortest_path = shortest_path
            self.pheromone *= self.decay  # Apply decay

        return all_time_shortest_path

    def spread_pheromone(self, all_paths, n_best):
        sorted_paths = sorted(all_paths, key=lambda x: x[1])  # Sort based on distance
        for path, dist in sorted_paths[:n_best]:  # Take n_best paths
            for move in path:
                self.pheromone[move] += 1.0 / self.distances[move]

    def gen_path_dist(self, path):
        total_dist = 0
        for i in range(len(path) - 1):
            total_dist += self.distances[path[i]][path[i + 1]]  # Compute distance from current to next
        return total_dist

    def gen_all_paths(self):
        all_paths = []
        for _ in range(self.n_ants):
            path = self.gen_path(0)  # Start at shelf 0
            path_distance = self.gen_path_dist(path)
            all_paths.append((path, path_distance))  # Append path and its distance
        return all_paths

    def gen_path(self, start):
        path = []
        visited = set()
        visited.add(start)
        prev = start
        for _ in range(len(self.distances) - 1):
            move = self.pick_move(self.pheromone[prev], self.distances[prev], visited)
            path.append(move)
            prev = move
            visited.add(move)
        path.append(start)  # Returning to start
        return path

    def pick_move(self, pheromone, dist, visited):
        pheromone = np.copy(pheromone)
        pheromone[list(visited)] = 0  # Block visited nodes
        dist = np.where(dist == 0, np.inf, dist)
        row = pheromone * self.alpha * ((1.0 / dist) * self.beta)

        if np.sum(row) == 0:
            return np.random.choice(list(set(self.all_inds) - visited))

        norm_row = row / row.sum()
        move = np.random.choice(list(self.all_inds), p=norm_row)
        return move

# Initialize and Run the Ant Colony Optimization
aco = AntColony(distance_matrix, n_ants=5, n_best=2, n_iterations=100, decay=0.95)
best_shelf_route = aco.run()

# Create the Gym Environment for Reinforcement Learning
class WarehouseEnv(gym.Env):
    def __init__(self, robot_id, assigned_orders):  # Accept robot ID and orders
        super(WarehouseEnv, self).__init__()
        self.robot_id = robot_id
        self.assigned_orders = assigned_orders
        self.action_space = gym.spaces.Discrete(num_shelves)  # Pick a shelf
        self.observation_space = gym.spaces.Box(low=0, high=100, shape=(2,), dtype=np.float32)

    def reset(self):
        self.state = np.zeros(2)  # Reset warehouse state to start position
        return self.state

    def step(self, action):
        distance_to_shelf = shelf_distance(int(self.state[0]), action)  # Get distance
        reward = -distance_to_shelf  # Penalize distance
        self.state[0] = action  # Update the current shelf
        done = False  # Indicate whether the episode is done
        return self.state, reward, done, {}

# Assign orders to robots (randomly for demonstration)
num_robots = len(robots_df)
orders_to_robots = {i: [] for i in range(num_robots)}
for index, order in orders_df.iterrows():
    assigned_robot = random.choice(range(num_robots))  # Randomly assign order to a robot
    orders_to_robots[assigned_robot].append(order)

# Create environment and train the RL Model for each robot
models = {}
for robot_id, assigned_orders in orders_to_robots.items():
    env = WarehouseEnv(robot_id, assigned_orders)
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=10000)
    models[robot_id] = model

# Streamlit UI
st.title("Warehouse Management System")

# Display Best Shelf Route
st.header("Best Shelf Route Found by Ant Colony Optimization")
st.write("Best shelf route:", best_shelf_route)

# Function to plot the route for each robot
def plot_robot_orders(robot_id, selected_robot_orders):
    for order in selected_robot_orders:
        fig, ax = plt.subplots(figsize=(8, 6))
        shelves_to_pick = np.random.choice(range(len(shelves_df)), size=min(order['Number_of_Items'], len(shelves_df)), replace=False)
        colors = plt.cm.viridis(np.linspace(0, 1, len(shelves_to_pick) - 1))

        for i in range(len(shelves_to_pick) - 1):
            x1, y1 = shelves_df[['Location_X', 'Location_Y']].iloc[shelves_to_pick[i]].values
            x2, y2 = shelves_df[['Location_X', 'Location_Y']].iloc[shelves_to_pick[i + 1]].values
            plt.plot([x1, x2], [y1, y2], color=colors[i], marker='o', label=f'Task {i + 1}')

        plt.title(f'Path Taken by Robot {robot_id + 1} for Order {order["Order_ID"]}')
        plt.xlabel('Location X')
        plt.ylabel('Location Y')
        plt.legend(loc='best')

        # Save the figure to a BytesIO object
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        st.image(buf, caption=f'Robot {robot_id + 1} Path for Order {order["Order_ID"]}', use_column_width=True)

# Test the trained models for all robots
for robot_id, model in models.items():
    env = WarehouseEnv(robot_id, orders_to_robots[robot_id])
    obs = env.reset()
    for step in range(100):
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        if done:
            break

    selected_robot_orders = orders_to_robots[robot_id]
    plot_robot_orders(robot_id, selected_robot_orders)

# Bar plot for the best shelf route
fig, ax = plt.subplots()
colors = plt.cm.rainbow(np.linspace(0, 1, len(best_shelf_route[0]) - 1))
plt.bar(range(len(best_shelf_route[0]) - 1),
        [shelf_distance(best_shelf_route[0][i], best_shelf_route[0][i + 1]) for i in range(len(best_shelf_route[0]) - 1)],
        color=colors)

plt.title('Task Completion Times (Robot Scheduling)')
plt.xlabel('Task Index')
plt.ylabel('Distance')

# Save the bar plot to a BytesIO object
buf = io.BytesIO()
plt.savefig(buf, format='png')
plt.close(fig)
buf.seek(0)
st.image(buf, caption='Task Completion Times (Robot Scheduling)', use_column_width=True)

# Run the Streamlit app
# if __name__ == "__main__":
#     st.run()
