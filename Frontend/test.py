import random
import os
import matplotlib.pyplot as plt
import numpy as np

# --- 1. HAMILTONIAN PATH GENERATOR (Unchanged) ---

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
            for c in range(cols - 1, -1, -1):
                path.append((r, c))
    path_map = {vertex: i for i, vertex in enumerate(path)}
    return path, path_map

def _is_circuit(path, rows, cols):
    if len(path) < 2: return False
    head = path[0]
    tail = path[-1]
    return tail in _get_neighbors(head[0], head[1], rows, cols)

def _apply_pivot_move(path, path_map, rows, cols):
    active_end_pos = random.choice([0, len(path) - 1])
    
    if active_end_pos == 0:
        end_vertex, adj_vertex = path[0], path[1]
        neighbors = _get_neighbors(end_vertex[0], end_vertex[1], rows, cols)
        potential_pivots = [path_map[v] for v in neighbors if v != adj_vertex and v in path_map]
        if not potential_pivots: return False
        pivot_idx = random.choice(potential_pivots)
        segment_to_reverse = path[0 : pivot_idx]
        segment_to_reverse.reverse()
        path[0 : pivot_idx] = segment_to_reverse
    else:
        end_vertex, adj_vertex = path[-1], path[-2]
        neighbors = _get_neighbors(end_vertex[0], end_vertex[1], rows, cols)
        potential_pivots = [path_map[v] for v in neighbors if v != adj_vertex and v in path_map]
        if not potential_pivots: return False
        pivot_idx = random.choice(potential_pivots)
        segment_to_reverse = path[pivot_idx + 1 : len(path)]
        segment_to_reverse.reverse()
        path[pivot_idx + 1 : len(path)] = segment_to_reverse
    
    for i, vertex in enumerate(path):
        path_map[vertex] = i
    return True

def generate_hamiltonian_path(rows: int, cols: int, circuits_only: bool = False, must_fill: bool = True):
    if not must_fill:
        raise NotImplementedError("This function only implements the 'must_fill=True' logic.")
    if rows <= 0 or cols <= 0: return []
    if rows * cols == 1: return [(0, 0)]

    path, path_map = _create_serpentine_path(rows, cols)
    num_vertices = rows * cols
    
    # We'll use a QF=1.0 equivalent for solid randomization
    num_moves = int(num_vertices * 50 * 1.0) 
    
    for _ in range(num_moves):
        _apply_pivot_move(path, path_map, rows, cols)
        
    # Handle 'circuits_only' - Not relevant for 5x5 but good to keep
    if circuits_only:
        if (rows * cols) % 2 != 0:
            return path # Circuit impossible, return path
            
        circuit_attempts = 0
        max_attempts = num_vertices * 100
        while not _is_circuit(path, rows, cols) and circuit_attempts < max_attempts:
            _apply_pivot_move(path, path_map, rows, cols)
            circuit_attempts += 1
            
    # For your 5x5 request, it's impossible to make a circuit.
    # If this is true, we must re-run moves until it's NOT a circuit.
    if (rows * cols) % 2 != 0 and circuits_only:
         print(f"Warning: A {rows}x{cols} grid has an odd number of vertices ({num_vertices}), so a circuit is impossible. Returning a path.")
         pass # A 5x5 can't be a circuit, so this is fine.
    
    # Ensure it's NOT a circuit if circuits_only=False
    if not circuits_only and _is_circuit(path, rows, cols):
        # Just make one more move to break the circuit
        _apply_pivot_move(path, path_map, rows, cols)

    return path

# --- 2. ✨ NEW: PUZZLE DATA GENERATORS ---

def place_checkpoints(path: list, num_checkpoints: int) -> dict:
    """
    Places sequential checkpoints along a given path.
    Returns a dictionary with 'start', 'goal', and an ordered 'checkpoints' list.
    """
    path_len = len(path)
    if path_len < 2:
        return {'start': None, 'goal': None, 'checkpoints': []}

    # The puzzle has N checkpoints + 1 goal
    # We divide the path into (num_checkpoints + 1) segments
    num_segments = num_checkpoints + 1
    
    # Calculate approximate indices for even spacing
    # This ensures checkpoints are spread out
    indices = np.linspace(0, path_len - 1, num_segments + 1, dtype=int)
    
    # Start is always the first point
    start_point = path[0]
    
    # Goal is always the last point
    goal_point = path[-1]
    
    # Checkpoints are the points at the segment divisions
    checkpoint_list = []
    for i in range(1, num_segments):
        checkpoint_list.append(path[indices[i]])

    return {
        'start': start_point,
        'goal': goal_point,
        'checkpoints': checkpoint_list # This list is sequential
    }

def generate_walls(path: list, rows: int, cols: int, wall_probability: float) -> set:
    """
    Generates a set of "blocked moves" (walls).
    Moves on the solution path are always allowed.
    Moves *not* on the path are blocked based on the wall_probability.
    """
    allowed_moves = set()
    
    # A "move" is a frozenset of two adjacent cells
    # We use frozenset so (A, B) is the same as (B, A)
    
    # 1. Create the set of all allowed moves from the path
    for i in range(len(path) - 1):
        cell_a = path[i]
        cell_b = path[i+1]
        allowed_moves.add(frozenset([cell_a, cell_b]))

    # 2. Find all "invalid" adjacent moves and potentially add them as walls
    blocked_walls = set()
    for r in range(rows):
        for c in range(cols):
            cell_a = (r, c)
            # Get all *grid-adjacent* neighbors
            neighbors = _get_neighbors(r, c, rows, cols)
            for cell_b in neighbors:
                move = frozenset([cell_a, cell_b])
                
                # If this adjacent move is NOT part of the solution...
                if move not in allowed_moves:
                    # ...roll the dice to see if we build a wall here.
                    if random.random() < wall_probability:
                        blocked_walls.add(move)

    return blocked_walls

# --- 3. ✨ NEW: MAIN PUZZLE GENERATOR FUNCTION ---

def generate_puzzle(rows: int, cols: int, num_checkpoints: int, wall_probability: float):
    """
    Generates a complete puzzle with a path, checkpoints, and walls.
    """
    # 1. Generate the solution path
    # We force 'circuits_only=False' for 5x5, as a circuit is impossible
    # and would break the 'must_fill' logic.
    is_circuit_possible = (rows * cols) % 2 == 0
    path = generate_hamiltonian_path(rows, cols, circuits_only=False)

    # 2. Place checkpoints along the path
    checkpoints_data = place_checkpoints(path, num_checkpoints)
    
    # 3. Generate the walls
    walls_data = generate_walls(path, rows, cols, wall_probability)
    
    return {
        'rows': rows,
        'cols': cols,
        'solution_path': path,
        'checkpoints': checkpoints_data,
        'walls': walls_data
    }

# --- 4. ✨ NEW: VISUALIZATION (Updated for Checkpoints) ---

def save_puzzle_as_image(puzzle: dict, filename: str):
    """
    Saves a visual representation of the puzzle.
    - Draws the solution path
    - Highlights start, goal, and checkpoints
    - !! NOW draws the walls (as red lines between cells)
    """
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
    
    # 1. Plot the solution path
    plt.plot(cols_x, rows_y, 'b-', alpha=0.3, label='Solution Path') # Faint blue line
    
    # 2. Plot the grid
    ax = plt.gca()
    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle=':', linewidth=0.5)
    ax.set_xticks(np.arange(0, cols, 1))
    ax.set_yticks(np.arange(0, rows, 1))
    ax.invert_yaxis()
    ax.set_aspect('equal', adjustable='box')
    
    # 3. ✨ NEW: Plot the walls ✨
    for wall in walls:
        # Get the two cells the wall is between
        # A frozenset isn't indexable, so convert to list
        cell_a, cell_b = list(wall)
        r1, c1 = cell_a
        r2, c2 = cell_b

        if r1 == r2:
            # This is a VERTICAL wall between (r1, c_min) and (r1, c_max)
            c_min = min(c1, c2)
            wall_x_pos = c_min + 0.5
            wall_y_span = [r1 - 0.5, r1 + 0.5]
            plt.plot([wall_x_pos, wall_x_pos], wall_y_span, 'r-', linewidth=4, solid_capstyle='butt')
        elif c1 == c2:
            # This is a HORIZONTAL wall between (r_min, c1) and (r_max, c1)
            r_min = min(r1, r2)
            wall_x_span = [c1 - 0.5, c1 + 0.5]
            wall_y_pos = r_min + 0.5
            plt.plot(wall_x_span, [wall_y_pos, wall_y_pos], 'r-', linewidth=4, solid_capstyle='butt')

    # 4. Plot checkpoints
    # Plot Start
    start_pos = cp_data['start']
    plt.plot(start_pos[1], start_pos[0], 'go', markersize=15, label='Start')
    plt.text(start_pos[1], start_pos[0], 'S', ha='center', va='center', color='white', weight='bold')

    # Plot Goal
    goal_pos = cp_data['goal']
    plt.plot(goal_pos[1], goal_pos[0], 'rs', markersize=15, label='Goal')
    plt.text(goal_pos[1], goal_pos[0], 'G', ha='center', va='center', color='white', weight='bold')
    
    # Plot numbered checkpoints
    for i, cp_pos in enumerate(cp_data['checkpoints']):
        plt.plot(cp_pos[1], cp_pos[0], 'yP', markersize=15, label=f'CP {i+1}') # Yellow Plus
        plt.text(cp_pos[1], cp_pos[0], str(i+1), ha='center', va='center', color='black', weight='bold')

    plt.title(f"{rows}x{cols} Puzzle | {len(cp_data['checkpoints'])} CPs | {len(walls)} Walls")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
   # --- EXAMPLE USAGE (Fixed filenames and print statements) ---
if __name__ == "__main__":
    
    # Create an 'output' directory if it doesn't exist
    output_dir = "hamiltonian_puzzles"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Generating puzzles... saving in '{output_dir}'")
    
    # --- Puzzle 1: 3 Checkpoints, 50% Wall Probability ---
    puzzle_1 = generate_puzzle(
        rows=7,
        cols=10, 
        num_checkpoints=3, 
        wall_probability=0.2
    )
    # Fixed filename and print statement
    save_puzzle_as_image(puzzle_1, os.path.join(output_dir, "puzzle_7x10_3cp_50w.png"))
    
    print("\n--- Puzzle 1 (7x10, 3 CPs, 0.5 Wall Prob) ---")
    print(f"  Start: {puzzle_1['checkpoints']['start']}")
    print(f"  Checkpoints: {puzzle_1['checkpoints']['checkpoints']}")
    print(f"  Goal: {puzzle_1['checkpoints']['goal']}")
    print(f"  Solution path length: {len(puzzle_1['solution_path'])}")
    print(f"  Number of walls: {len(puzzle_1['walls'])}")

    # --- Puzzle 2: 5 Checkpoints, 90% Wall Probability ---
    puzzle_2 = generate_puzzle(
        rows=10, 
        cols=12, 
        num_checkpoints=15, 
        wall_probability=0.4
    )
    # Fixed filename and print statement
    save_puzzle_as_image(puzzle_2, os.path.join(output_dir, "puzzle_10x12_15cp_90w.png"))

    print("\n--- Puzzle 2 (10x12, 15 CPs, 0.9 Wall Prob) ---")
    print(f"  Start: {puzzle_2['checkpoints']['start']}")
    print(f"  Checkpoints: {puzzle_2['checkpoints']['checkpoints']}")
    print(f"  Goal: {puzzle_2['checkpoints']['goal']}")
    print(f"  Solution path length: {len(puzzle_2['solution_path'])}")
    print(f"  Number of walls: {len(puzzle_2['walls'])}")

    # --- Puzzle 3: 1 Checkpoint, 10% Wall Probability ---
    puzzle_3 = generate_puzzle(
        rows=6, 
        cols=8, 
        num_checkpoints=4, 
        wall_probability=0.1
    )
    # Fixed filename and print statement
    save_puzzle_as_image(puzzle_3, os.path.join(output_dir, "puzzle_6x8_4cp_10w.png"))
    
    print("\n--- Puzzle 3 (6x8, 4 CPs, 0.1 Wall Prob) ---")
    print(f"  Start: {puzzle_3['checkpoints']['start']}")
    print(f"  Checkpoints: {puzzle_3['checkpoints']['checkpoints']}")
    print(f"  Goal: {puzzle_3['checkpoints']['goal']}")
    print(f"  Solution path length: {len(puzzle_3['solution_path'])}")
    print(f"  Number of walls: {len(puzzle_3['walls'])}")

    print(f"\nSuccessfully generated {3} puzzles.")