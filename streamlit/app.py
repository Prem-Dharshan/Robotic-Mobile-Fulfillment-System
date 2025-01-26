import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random as status
import gym
from stable_baselines3 import PPO

import os
current_dir = os.path.dirname(os.path.abspath(__file__))
orders_path = os.path.join(current_dir, 'orders.csv')
shelves_path = os.path.join(current_dir, 'shelves.csv')
robots_path = os.path.join(current_dir, 'robots.csv')

orders_df = pd.read_csv(orders_path)
shelves_df = pd.read_csv(shelves_path)
robots_df = pd.read_csv(robots_path)

# Ant Colony Optimization Class
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
            shortest_path = min(all_paths, key=lambda x: x[1])
            if shortest_path[1] < all_time_shortest_path[1]:
                all_time_shortest_path = shortest_path
            self.pheromone *= self.decay
        return all_time_shortest_path

    def spread_pheromone(self, all_paths, n_best):
        sorted_paths = sorted(all_paths, key=lambda x: x[1])
        for path, dist in sorted_paths[:n_best]:
            for move in path:
                self.pheromone[move] += 1.0 / self.distances[move]

    def gen_path_dist(self, path):
        total_dist = 0
        for i in range(len(path) - 1):
            total_dist += self.distances[path[i]][path[i + 1]]
        return total_dist

    def gen_all_paths(self):
        all_paths = []
        for _ in range(self.n_ants):
            path = self.gen_path(0)
            path_distance = self.gen_path_dist(path)
            all_paths.append((path, path_distance))
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
        path.append(start)
        return path

    def pick_move(self, pheromone, dist, visited):
        pheromone = np.copy(pheromone)
        pheromone[list(visited)] = 0
        dist = np.where(dist == 0, np.inf, dist)
        row = pheromone * self.alpha * ((1.0 / dist) * self.beta)

        if np.sum(row) == 0:
            return np.random.choice(list(set(self.all_inds) - visited))

        norm_row = row / row.sum()
        move = np.random.choice(list(self.all_inds), p=norm_row)
        return move

# Define the Distance Matrix Function
shelves_locations = shelves_df[['Location_X', 'Location_Y']].to_numpy()

def shelf_distance(s1, s2):
    return np.linalg.norm(shelves_locations[s1] - shelves_locations[s2])

num_shelves = len(shelves_locations)
distance_matrix = np.zeros((num_shelves, num_shelves))
for i in range(num_shelves):
    for j in range(num_shelves):
        distance_matrix[i][j] = shelf_distance(i, j)

# Gym Environment
class WarehouseEnv(gym.Env):
    def __init__(self, robot_id, assigned_orders):
        super(WarehouseEnv, self).__init__()
        self.robot_id = robot_id
        self.assigned_orders = assigned_orders
        self.action_space = gym.spaces.Discrete(num_shelves)
        self.observation_space = gym.spaces.Box(low=0, high=100, shape=(2,), dtype=np.float32)

    def reset(self):
        self.state = np.zeros(2)
        return self.state

    def step(self, action):
        distance_to_shelf = shelf_distance(int(self.state[0]), action)
        reward = -distance_to_shelf
        self.state[0] = action
        done = False
        return self.state, reward, done, {}

# Main Function
def main():
    # Initialize and Run ACO
    aco = AntColony(distance_matrix, n_ants=5, n_best=2, n_iterations=100, decay=0.95)
    best_shelf_route = aco.run()
    st.write("Best shelf route:", best_shelf_route)

    # RL Model Training and Visualization
    num_robots = len(robots_df)
    orders_to_robots = {i: [] for i in range(num_robots)}
    for index, order in orders_df.iterrows():
        assigned_robot = status.choice(range(num_robots))
        orders_to_robots[assigned_robot].append(order)

    models = {}
    for robot_id, assigned_orders in orders_to_robots.items():
        env = WarehouseEnv(robot_id, assigned_orders)
        model = PPO("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=10000)
        models[robot_id] = model

        # Visualization
        obs = env.reset()
        for step in range(100):
            action, _states = model.predict(obs)
            obs, rewards, done, info = env.step(action)
            if done:
                break

        for order in orders_to_robots[robot_id]:
            fig, ax = plt.subplots(figsize=(8, 6))
            shelves_to_pick = np.random.choice(range(len(shelves_df)), size=min(order['Number_of_Items'], len(shelves_df)), replace=False)
            colors = plt.cm.viridis(np.linspace(0, 1, len(shelves_to_pick) - 1))
            for i in range(len(shelves_to_pick) - 1):
                x1, y1 = shelves_df[['Location_X', 'Location_Y']].iloc[shelves_to_pick[i]].values
                x2, y2 = shelves_df[['Location_X', 'Location_Y']].iloc[shelves_to_pick[i + 1]].values
                ax.plot([x1, x2], [y1, y2], color=colors[i], marker='o', label=f'Task {i + 1}')
            ax.set_title(f'Path Taken by Robot {robot_id + 1} for Order {order["Order_ID"]}')
            ax.set_xlabel('Location X')
            ax.set_ylabel('Location Y')
            ax.legend(loc='best')
            st.pyplot(fig)

    # Best Route Visualization
    fig, ax = plt.subplots()
    colors = plt.cm.rainbow(np.linspace(0, 1, len(best_shelf_route[0]) - 1))
    distances = [shelf_distance(best_shelf_route[0][i], best_shelf_route[0][i + 1]) for i in range(len(best_shelf_route[0]) - 1)]
    ax.bar(range(len(distances)), distances, color=colors)
    ax.set_title('Task Completion Times (Robot Scheduling)')
    ax.set_xlabel('Task Index')
    ax.set_ylabel('Distance')
    st.pyplot(fig)

# Streamlit Interface
st.title("Warehouse Optimization with ACO and PPO")
if st.button("Start Optimization"):
    main()
