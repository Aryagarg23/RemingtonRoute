import random
import os
import matplotlib.pyplot as plt
from hamiltonenv import HamiltonianPuzzleEnv

# ==============================
# Dataset parameters
# ==============================
num_samples = 5
rows, cols = 7, 7
checkpoint_range = (13, 13)
wall_probability = 0.2
output_dir = "dataset_samples_side_by_side/"
os.makedirs(output_dir, exist_ok=True)

# ==============================
# Helper function to draw puzzle
# ==============================
def draw_puzzle(puzzle, show_solution=False, ax=None, title=""):
    """
    Draws the puzzle on the given matplotlib axis.
    - show_solution=True → draw full solution path
    - show_solution=False → only walls and checkpoints
    """
    path = puzzle['solution_path']
    checkpoints = puzzle['checkpoints']
    walls = puzzle['walls']
    rows, cols = puzzle['rows'], puzzle['cols']

    if ax is None:
        fig, ax = plt.subplots(figsize=(cols, rows))

    # Grid
    ax.set_xticks(range(cols))
    ax.set_yticks(range(rows))
    ax.set_xticks([x-0.5 for x in range(1, cols)], minor=True)
    ax.set_yticks([y-0.5 for y in range(1, rows)], minor=True)
    ax.grid(which='minor', color='gray', linestyle=':', linewidth=0.5)
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.set_title(title)

    # Draw walls
    for wall in walls:
        a, b = list(wall)
        r1, c1 = a
        r2, c2 = b
        if r1 == r2:  # vertical wall
            ax.plot([min(c1,c2)+0.5]*2, [r1-0.5, r1+0.5], 'r-', linewidth=4, solid_capstyle='butt')
        elif c1 == c2:  # horizontal wall
            ax.plot([c1-0.5, c1+0.5], [min(r1,r2)+0.5]*2, 'r-', linewidth=4, solid_capstyle='butt')

    # Draw solution path if requested
    if show_solution and len(path) > 1:
        r_coords, c_coords = zip(*path)
        ax.plot(c_coords, r_coords, 'b-', alpha=0.6, linewidth=2, label='Solution Path')

    # Draw start, goal
    ax.plot(checkpoints['start'][1], checkpoints['start'][0], 'go', markersize=12)
    ax.text(checkpoints['start'][1], checkpoints['start'][0], 'S', ha='center', va='center', color='white', weight='bold')

    ax.plot(checkpoints['goal'][1], checkpoints['goal'][0], 'rs', markersize=12)
    ax.text(checkpoints['goal'][1], checkpoints['goal'][0], 'G', ha='center', va='center', color='white', weight='bold')

    # Draw numbered checkpoints
    for i, cp in enumerate(checkpoints['checkpoints']):
        ax.plot(cp[1], cp[0], 'yP', markersize=12)
        ax.text(cp[1], cp[0], str(i+1), ha='center', va='center', color='black', weight='bold')

    return ax

# ==============================
# Generate dataset
# ==============================
for i in range(num_samples):
    num_checkpoints = random.randint(*checkpoint_range)

    # Create environment
    env = HamiltonianPuzzleEnv(
        rows=rows, cols=cols,
        num_checkpoints=num_checkpoints,
        wall_probability=wall_probability,
        render_mode=None
    )

    # Reset env to generate puzzle
    env.reset()
    puzzle = env.puzzle_data

    # Side-by-side figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # Left: incomplete puzzle (walls + checkpoints)
    draw_puzzle(puzzle, show_solution=False, ax=axes[0], title=f"Incomplete ({num_checkpoints} CPs)")

    # Right: complete puzzle (walls + solution path)
    draw_puzzle(puzzle, show_solution=True, ax=axes[1], title=f"Complete Solution ({num_checkpoints} CPs)")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"sample_{i}_side_by_side.png"))
    plt.close()

    env.close()

print(f"Dataset generated! Images saved in: {output_dir}")
