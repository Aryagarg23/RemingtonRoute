import random
from typing import Optional, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces

# ==============================
# 1. HAMILTONIAN PATH GENERATOR
# ==============================

def _get_neighbors(r, c, rows, cols):
    neighbors = []
    if r > 0: neighbors.append((r - 1, c))
    if r < rows - 1: neighbors.append((r + 1, c))
    if c > 0: neighbors.append((r, c - 1))
    if c < cols - 1: neighbors.append((r, c + 1))
    return neighbors

def _create_serpentine_path(rows, cols):
    path = []
    for r in range(rows):
        if r % 2 == 0:
            for c in range(cols):
                path.append((r, c))
        else:
            for c in reversed(range(cols)):
                path.append((r, c))
    path_map = {vertex: i for i, vertex in enumerate(path)}
    return path, path_map

def _is_circuit(path, rows, cols):
    if len(path) < 2: return False
    return path[-1] in _get_neighbors(path[0][0], path[0][1], rows, cols)

def _apply_pivot_move(path, path_map, rows, cols):
    end_idx = random.choice([0, len(path) - 1])
    if end_idx == 0:
        end, adj = path[0], path[1]
        neighbors = _get_neighbors(end[0], end[1], rows, cols)
        pivots = [path_map[v] for v in neighbors if v != adj and v in path_map]
        if not pivots: return False
        pivot = random.choice(pivots)
        path[0:pivot] = reversed(path[0:pivot])
    else:
        end, adj = path[-1], path[-2]
        neighbors = _get_neighbors(end[0], end[1], rows, cols)
        pivots = [path_map[v] for v in neighbors if v != adj and v in path_map]
        if not pivots: return False
        pivot = random.choice(pivots)
        path[pivot+1:] = reversed(path[pivot+1:])
    for i, v in enumerate(path):
        path_map[v] = i
    return True

def generate_hamiltonian_path(rows: int, cols: int, circuits_only: bool = False):
    if rows <= 0 or cols <= 0: return []
    if rows * cols == 1: return [(0, 0)]

    path, path_map = _create_serpentine_path(rows, cols)
    num_moves = rows * cols * 50  # Randomization factor
    for _ in range(num_moves):
        _apply_pivot_move(path, path_map, rows, cols)

    if not circuits_only and _is_circuit(path, rows, cols):
        _apply_pivot_move(path, path_map, rows, cols)

    return path

# ==============================
# 2. PUZZLE DATA GENERATORS
# ==============================

def place_checkpoints(path: list, num_checkpoints: int) -> dict:
    if len(path) < 2:
        return {'start': None, 'goal': None, 'checkpoints': []}

    indices = np.linspace(0, len(path)-1, num_checkpoints+2, dtype=int)
    return {
        'start': path[0],
        'goal': path[-1],
        'checkpoints': [path[i] for i in indices[1:-1]]
    }

def generate_walls(path: list, rows: int, cols: int, wall_probability: float) -> set:
    allowed = {frozenset([path[i], path[i+1]]) for i in range(len(path)-1)}
    blocked = set()
    for r in range(rows):
        for c in range(cols):
            cell = (r, c)
            for n in _get_neighbors(r, c, rows, cols):
                move = frozenset([cell, n])
                if move not in allowed and random.random() < wall_probability:
                    blocked.add(move)
    return blocked

def generate_puzzle(rows: int, cols: int, num_checkpoints: int, wall_probability: float):
    path = generate_hamiltonian_path(rows, cols)
    checkpoints = place_checkpoints(path, num_checkpoints)
    walls = generate_walls(path, rows, cols, wall_probability)
    return {
        'rows': rows,
        'cols': cols,
        'solution_path': path,
        'checkpoints': checkpoints,
        'walls': walls,
        'wall_set': walls
    }

# ==============================
# 3. GYMNASIUM ENVIRONMENT
# ==============================

class HamiltonianPuzzleEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode: Optional[str] = None, **kwargs):
        super().__init__()
        self.rows = kwargs.get('rows', 7)
        self.cols = kwargs.get('cols', 7)
        self.num_checkpoints = kwargs.get('num_checkpoints', 3)
        self.wall_probability = kwargs.get('wall_probability', 0.1)
        self.max_steps = kwargs.get('max_steps', self.rows * self.cols * 2)
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0]),
            high=np.array([self.rows-1, self.cols-1, self.num_checkpoints]),
            dtype=np.int32
        )

        self.puzzle_data = None
        self.agent_pos = None
        self.current_cp_idx = 0
        self.current_step = 0

    def _get_obs(self):
        return np.array([*self.agent_pos, self.current_cp_idx], dtype=np.int32)

    def _get_info(self):
        target = (self.puzzle_data['checkpoints']['checkpoints'][self.current_cp_idx]
                  if self.current_cp_idx < self.num_checkpoints
                  else self.puzzle_data['checkpoints']['goal'])
        return {"current_target": target, "next_cp_to_visit": self.current_cp_idx + 1}

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        self.puzzle_data = generate_puzzle(
            self.rows, self.cols, self.num_checkpoints, self.wall_probability
        )
        self.agent_pos = self.puzzle_data['checkpoints']['start']
        self.current_cp_idx = 0
        self.current_step = 0
        obs, info = self._get_obs(), self._get_info()
        if self.render_mode == "human": self.render()
        return obs, info

    def step(self, action: int):
        self.current_step += 1
        r, c = self.agent_pos
        next_r, next_c = r, c
        if action == 0: next_r -= 1
        elif action == 1: next_r += 1
        elif action == 2: next_c -= 1
        elif action == 3: next_c += 1
        next_pos = (next_r, next_c)
        reward = -1.0
        terminated = truncated = False

        if not (0 <= next_r < self.rows and 0 <= next_c < self.cols) or \
           frozenset([self.agent_pos, next_pos]) in self.puzzle_data['wall_set']:
            reward = -5.0
        else:
            self.agent_pos = next_pos
            reward = -0.1
            if self.current_cp_idx < self.num_checkpoints and \
               next_pos == self.puzzle_data['checkpoints']['checkpoints'][self.current_cp_idx]:
                self.current_cp_idx += 1
                reward = 50.0
            elif self.current_cp_idx == self.num_checkpoints and \
                 next_pos == self.puzzle_data['checkpoints']['goal']:
                reward = 100.0
                terminated = True

        if self.current_step >= self.max_steps and not terminated:
            truncated = True

        obs, info = self._get_obs(), self._get_info()
        if self.render_mode == "human": self.render()
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode is None or self.puzzle_data is None:
            return

        puzzle = self.puzzle_data
        walls = puzzle['walls']
        cp_data = puzzle['checkpoints']
        rows, cols = puzzle['rows'], puzzle['cols']

        plt.figure(figsize=(cols, rows))
        ax = plt.gca()
        ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
        ax.grid(which='minor', color='gray', linestyle=':', linewidth=0.5)
        ax.set_xticks(np.arange(cols))
        ax.set_yticks(np.arange(rows))
        ax.invert_yaxis()
        ax.set_aspect('equal')
        ax.set_title(f"Step: {self.current_step} | Next CP: {self.current_cp_idx + 1}")

        # Plot walls
        for wall in walls:
            a, b = list(wall)
            r1, c1 = a; r2, c2 = b
            if r1 == r2:
                plt.plot([min(c1, c2)+0.5]*2, [r1-0.5, r1+0.5], 'r-', linewidth=4, solid_capstyle='butt')
            elif c1 == c2:
                plt.plot([c1-0.5, c1+0.5], [min(r1, r2)+0.5]*2, 'r-', linewidth=4, solid_capstyle='butt')

        # Plot start, goal, and checkpoints
        plt.plot(cp_data['start'][1], cp_data['start'][0], 'go', markersize=15, alpha=0.5)
        plt.text(cp_data['start'][1], cp_data['start'][0], 'S', ha='center', va='center', color='white', weight='bold')

        plt.plot(cp_data['goal'][1], cp_data['goal'][0], 'rs', markersize=15, alpha=0.5)
        plt.text(cp_data['goal'][1], cp_data['goal'][0], 'G', ha='center', va='center', color='white', weight='bold')

        for i, cp in enumerate(cp_data['checkpoints']):
            color = 'bP' if i < self.current_cp_idx else 'yP'
            plt.plot(cp[1], cp[0], color, markersize=15)
            plt.text(cp[1], cp[0], str(i+1), ha='center', va='center', color='black', weight='bold')

        # Plot agent
        plt.plot(self.agent_pos[1], self.agent_pos[0], 'kH', markersize=20)

        if self.render_mode == "human":
            plt.show()
            plt.pause(0.1)
        plt.close()

    def close(self):
        plt.close()


def save_puzzle_as_image(puzzle: dict, filename: str):
    import matplotlib.pyplot as plt
    import numpy as np

    path = puzzle['solution_path']
    cp_data = puzzle['checkpoints']
    walls = puzzle['walls']
    rows = puzzle['rows']
    cols = puzzle['cols']

    if not path:
        print("Cannot save empty path.")
        return

    rows_y, cols_x = zip(*path)
    plt.figure(figsize=(cols, rows))

    # Plot grid
    ax = plt.gca()
    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle=':', linewidth=0.5)
    ax.set_xticks(np.arange(0, cols, 1))
    ax.set_yticks(np.arange(0, rows, 1))
    ax.invert_yaxis()
    ax.set_aspect('equal')

    # Plot walls
    for wall in walls:
        cell_a, cell_b = list(wall)
        r1, c1 = cell_a
        r2, c2 = cell_b
        if r1 == r2:
            plt.plot([min(c1, c2)+0.5]*2, [r1-0.5, r1+0.5], 'r-', linewidth=4, solid_capstyle='butt')
        elif c1 == c2:
            plt.plot([c1-0.5, c1+0.5], [min(r1, r2)+0.5]*2, 'r-', linewidth=4, solid_capstyle='butt')

    # Plot Start, Goal, Checkpoints
    plt.plot(cp_data['start'][1], cp_data['start'][0], 'go', markersize=15)
    plt.text(cp_data['start'][1], cp_data['start'][0], 'S', ha='center', va='center', color='white', weight='bold')

    plt.plot(cp_data['goal'][1], cp_data['goal'][0], 'rs', markersize=15)
    plt.text(cp_data['goal'][1], cp_data['goal'][0], 'G', ha='center', va='center', color='white', weight='bold')

    for i, cp in enumerate(cp_data['checkpoints']):
        plt.plot(cp[1], cp[0], 'yP', markersize=15)
        plt.text(cp[1], cp[0], str(i+1), ha='center', va='center', color='black', weight='bold')

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# ==============================
# 4. EXAMPLE USAGE
# ==============================

if __name__ == "__main__":
    env = HamiltonianPuzzleEnv(
        rows=5, cols=5, num_checkpoints=2, wall_probability=0.1,
        render_mode="human", max_steps=100
    )

    obs, info = env.reset()
    print(f"Initial Observation: {obs}, Info: {info}")

    total_reward = 0
    for _ in range(env.max_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break

    print(f"Episode finished. Total Reward: {total_reward:.1f}, Steps: {env.current_step}")
    env.close()
