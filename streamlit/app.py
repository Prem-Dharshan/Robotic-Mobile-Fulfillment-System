import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import streamlit as st

# Load datasets
basedir = os.path.abspath(os.path.dirname(__file__))
orders_df = pd.read_csv(os.path.join(basedir, "orders.csv"))
shelves_df = pd.read_csv(os.path.join(basedir, "shelves.csv"))
robots_df = pd.read_csv(os.path.join(basedir, "robots.csv"))

# 1. Create a NumPy array for shelf locations
shelves_locations = shelves_df[["Location_X", "Location_Y"]].to_numpy()


# 2. Define the Distance Matrix Function
def shelf_distance(s1, s2):
    if not (0 <= s1 < len(shelves_locations)) or not (0 <= s2 < len(shelves_locations)):
        raise IndexError(
            f"Invalid index: s1={s1}, s2={s2} for shelves_locations with size {len(shelves_locations)}"
        )
    return np.linalg.norm(shelves_locations[s1] - shelves_locations[s2])


# Create a distance matrix
num_shelves = len(shelves_locations)
distance_matrix = np.zeros((num_shelves, num_shelves))

for i in range(num_shelves):
    for j in range(num_shelves):
        distance_matrix[i][j] = shelf_distance(i, j)


# 3. Ant Colony Optimization Class
class AntColony:
    def __init__(
        self, distance_matrix, n_ants, n_best, n_iterations, decay, alpha=1, beta=1
    ):
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
            for i in range(len(path) - 1):  # Loop through the path
                move_from = path[i]
                move_to = path[i + 1]
                if (
                    self.distances[move_from][move_to] > 0
                ):  # Ensure we don't divide by zero
                    self.pheromone[move_from][move_to] += (
                        1.0 / self.distances[move_from][move_to]
                    )
                else:
                    pass

    def gen_path_dist(self, path):
        total_dist = 0
        for i in range(len(path) - 1):
            total_dist += self.distances[path[i]][
                path[i + 1]
            ]  # Compute distance from current to next
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
        dist = np.where(dist == 0, np.inf, dist)  # Avoid dividing by zero

        row = pheromone * self.alpha * ((1.0 / dist) * self.beta)

        if np.sum(row) == 0:
            return np.random.choice(
                list(set(self.all_inds) - visited)
            )  # Randomly pick unvisited node

        norm_row = row / row.sum()
        move = np.random.choice(list(self.all_inds), p=norm_row)
        return move


# 4. Run the Ant Colony Optimization to get the best shelf route
aco = AntColony(distance_matrix, n_ants=5, n_best=2, n_iterations=100, decay=0.95)
best_shelf_route = aco.run()


# 5. Plotting functions using networkx
def plot_path_with_shelves(shelves_to_pick, robot_id, order_id):
    G = nx.Graph()

    # Add nodes for each shelf location
    for i, (x, y) in enumerate(shelves_locations):
        G.add_node(i, pos=(x, y))

    # Add edges for the path the robot travels
    for i in range(len(shelves_to_pick) - 1):
        G.add_edge(
            shelves_to_pick[i],
            shelves_to_pick[i + 1],
            weight=shelf_distance(shelves_to_pick[i], shelves_to_pick[i + 1]),
        )

    pos = nx.get_node_attributes(G, "pos")

    # Plotting the graph
    plt.figure(figsize=(8, 6))

    # Draw the shelves
    nx.draw_networkx_nodes(G, pos, node_color="blue", node_size=300, label="Shelves")

    # Draw the path traveled by the robot
    edges = [
        (shelves_to_pick[i], shelves_to_pick[i + 1])
        for i in range(len(shelves_to_pick) - 1)
    ]
    nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color="red", width=2)

    # Draw the labels (shelf IDs)
    nx.draw_networkx_labels(G, pos)

    # Title and labels
    plt.title(f"Robot {robot_id + 1} Path for Order {order_id}")
    plt.xlabel("Location X")
    plt.ylabel("Location Y")
    plt.grid(True)
    st.pyplot(plt)


# 6. Display metrics and graphs in Streamlit
def display_metrics_and_graphs():
    st.title("Robotic Path Visualization and Metrics")

    # Select a specific robot and orders to visualize
    selected_robot_id = st.selectbox("Select Robot ID", robots_df["Robot_ID"].unique())
    selected_orders = st.multiselect("Select Orders", orders_df["Order_ID"].unique())

    if selected_orders:
        for order_id in selected_orders:
            order = orders_df[orders_df["Order_ID"] == order_id].iloc[0]

            # For visualization, assume each order corresponds to a set of shelves
            shelves_to_pick = np.random.choice(
                range(len(shelves_df)),
                size=min(order["Number_of_Items"], len(shelves_df)),
                replace=False,
            )

            # Plot path with shelves
            plot_path_with_shelves(shelves_to_pick, selected_robot_id, order_id)

    # Plot bar chart for task completion times
    colors = plt.cm.rainbow(np.linspace(0, 1, len(best_shelf_route[0]) - 1))

    plt.figure(figsize=(8, 6))
    plt.bar(
        range(len(best_shelf_route[0]) - 1),
        [
            shelf_distance(best_shelf_route[0][i], best_shelf_route[0][i + 1])
            for i in range(len(best_shelf_route[0]) - 1)
        ],
        color=colors,
    )

    plt.title("Task Completion Times (Robot Scheduling)")
    plt.xlabel("Task Index")
    plt.ylabel("Distance")
    plt.grid(True)
    st.pyplot(plt)


# Run the Streamlit app
if __name__ == "__main__":
    display_metrics_and_graphs()
