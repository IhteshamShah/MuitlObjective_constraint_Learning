# import matplotlib.pyplot as plt
# import numpy as np
# import matplotlib.patches as mpatches

# # 1. Define the 8x8 Gridworld (0: Grass, 1: Rock, 2: Water)
# # Let's put Grass mainly on the top/right, Rock on the bottom/left, and Water in the middle.
# grid = np.array([
#     [0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 1, 1, 1],
#     [1, 1, 0, 2, 2, 1, 1, 1],
#     [1, 1, 0, 2, 2, 0, 1, 1],
#     [1, 1, 1, 0, 0, 0, 0, 0],
#     [1, 1, 1, 1, 1, 1, 0, 0],
#     [1, 1, 1, 1, 1, 1, 0, 0]
# ])

# # Define colors for the grid
# colors = {
#     0: '#A9DFBF',  # Light Green (Grass)
#     1: '#D5DBDB',  # Light Gray (Rock)
#     2: '#5DADE2'   # Blue (Water - Hard Constraint)
# }

# fig, ax = plt.subplots(figsize=(10, 8))

# # Draw the grid cells
# for i in range(8):
#     for j in range(8):
#         rect = plt.Rectangle((j, 7-i), 1, 1, facecolor=colors[grid[i, j]], edgecolor='white', linewidth=2)
#         ax.add_patch(rect)

# # 2. Define Trajectories (Coordinates (x, y) where x is col, y is row from bottom)
# # Note: In matplotlib, (0,0) is bottom-left. 
# # Grass-lovers prefer the green areas (top right route)
# grass_trajectories = [
#     [(0.5, 7.5), (1.5, 7.5), (2.5, 7.5), (3.5, 7.5), (4.5, 7.5), (5.5, 6.5), (6.5, 5.5), (7.5, 4.5)],
#     [(0.5, 7.5), (0.5, 6.5), (1.5, 6.5), (2.5, 6.5), (3.5, 6.5), (4.5, 6.5), (5.5, 5.5), (7.5, 4.5)],
#     [(0.5, 7.5), (1.5, 7.5), (2.5, 6.5), (3.5, 5.5), (4.5, 5.5), (5.5, 5.5), (6.5, 4.5), (7.5, 4.5)],
#     [(0.5, 7.5), (1.5, 6.5), (2.5, 6.5), (3.5, 6.5), (4.5, 5.5), (5.5, 6.5), (6.5, 4.5), (7.5, 4.5)],
#     [(0.5, 7.5), (0.5, 6.5), (1.5, 5.5), (2.5, 5.5), (3.5, 5.5), (4.5, 6.5), (5.5, 6.5), (7.5, 4.5)]
# ]

# # Rock-lovers prefer the gray areas (bottom left route)
# rock_trajectories = [
#     [(0.5, 7.5), (0.5, 6.5), (0.5, 5.5), (0.5, 4.5), (0.5, 3.5), (1.5, 2.5), (2.5, 1.5), (3.5, 1.5), (4.5, 1.5), (5.5, 1.5), (6.5, 2.5), (7.5, 4.5)],
#     [(0.5, 7.5), (0.5, 5.5), (1.5, 4.5), (1.5, 3.5), (2.5, 2.5), (3.5, 2.5), (4.5, 2.5), (5.5, 2.5), (6.5, 3.5), (7.5, 4.5)],
#     [(0.5, 7.5), (0.5, 6.5), (0.5, 4.5), (1.5, 3.5), (1.5, 2.5), (2.5, 1.5), (3.5, 0.5), (4.5, 0.5), (5.5, 1.5), (6.5, 2.5), (7.5, 4.5)],
#     [(0.5, 7.5), (1.5, 6.5), (1.5, 5.5), (1.5, 4.5), (2.5, 3.5), (2.5, 2.5), (3.5, 2.5), (4.5, 1.5), (5.5, 2.5), (6.5, 3.5), (7.5, 4.5)],
#     [(0.5, 7.5), (0.5, 5.5), (0.5, 4.5), (0.5, 2.5), (1.5, 1.5), (2.5, 0.5), (3.5, 1.5), (4.5, 2.5), (5.5, 3.5), (6.5, 3.5), (7.5, 4.5)]
# ]

# # 3. Plot the trajectories
# # Add slight jitter to overlapping lines so they are all visible
# np.random.seed(42)

# for path in grass_trajectories:
#     x_coords = [p[0] + np.random.uniform(-0.1, 0.1) for p in path]
#     y_coords = [p[1] + np.random.uniform(-0.1, 0.1) for p in path]
#     ax.plot(x_coords, y_coords, color='#229954', linewidth=3, alpha=0.8, marker='o', markersize=5)

# for path in rock_trajectories:
#     x_coords = [p[0] + np.random.uniform(-0.1, 0.1) for p in path]
#     y_coords = [p[1] + np.random.uniform(-0.1, 0.1) for p in path]
#     ax.plot(x_coords, y_coords, color='#5D6D7E', linewidth=3, alpha=0.8, marker='o', markersize=5)

# # 4. Formatting and Legend
# ax.set_xlim(0, 8)
# ax.set_ylim(0, 8)
# ax.set_xticks(np.arange(0, 9, 1))
# ax.set_yticks(np.arange(0, 9, 1))
# ax.grid(color='black', linestyle='-', linewidth=1)
# ax.set_xticklabels([])
# ax.set_yticklabels([])
# ax.tick_params(axis='both', which='both', length=0)

# # Custom legend
# grass_patch = mpatches.Patch(color='#A9DFBF', label='Grass (Preference 1)')
# rock_patch = mpatches.Patch(color='#D5DBDB', label='Rock (Preference 2)')
# water_patch = mpatches.Patch(color='#5DADE2', label='Water (Hard Constraint)')
# grass_line = plt.Line2D([0], [0], color='#229954', lw=3, label='Grass-lover Trajectories')
# rock_line = plt.Line2D([0], [0], color='#5D6D7E', lw=3, label='Rock-lover Trajectories')

# plt.legend(handles=[grass_patch, rock_patch, water_patch, grass_line, rock_line], 
#            loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize=12, frameon=True)

# plt.title('8x8 Gridworld: Heterogeneous Expert Trajectories', fontsize=16, pad=20)
# plt.gca().set_aspect('equal', adjustable='box')
# plt.tight_layout()

# # Save the figure to be used in PowerPoint
# plt.savefig('gridworld_trajectories.png', dpi=300, bbox_inches='tight')
# plt.show()

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

# 1. Define the 8x8 Gridworld
# 0: Normal, 1: Grass, 2: Rocks, 3: Water (Hard Constraint)
grid = np.array([
    [0, 0, 0, 1, 1, 1, 1, 1],
    [0, 0, 0, 1, 1, 1, 1, 1],
    [0, 0, 3, 3, 1, 1, 1, 1],
    [2, 2, 3, 3, 3, 1, 1, 1],
    [2, 2, 2, 3, 3, 3, 1, 1],
    [2, 2, 2, 2, 3, 3, 0, 0],
    [2, 2, 2, 2, 2, 0, 0, 0],
    [2, 2, 2, 2, 2, 0, 0, 0]
])

# Define a custom colormap
# Normal: Beige/White, Grass: Light Green, Rocks: Light Gray, Water: Blue
colors = ['#FDFEFE', '#A9DFBF', '#D5DBDB', '#5DADE2']
cmap = ListedColormap(colors)

fig, ax = plt.subplots(figsize=(10, 8))

# Draw the grid
cax = ax.imshow(grid, cmap=cmap, origin='upper')

# Add gridlines
ax.set_xticks(np.arange(-.5, 8, 1), minor=True)
ax.set_yticks(np.arange(-.5, 8, 1), minor=True)
ax.grid(which='minor', color='black', linestyle='-', linewidth=1)
ax.tick_params(which='minor', size=0)

# Remove major ticks
ax.set_xticks([])
ax.set_yticks([])

# Mark Start and Goal states
ax.text(0, 0, 'Start', ha='center', va='center', fontsize=12, fontweight='bold', color='black', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))
ax.text(7, 7, 'Goal', ha='center', va='center', fontsize=12, fontweight='bold', color='black', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))

# 2. Define Base Trajectories (Format: (col_x, row_y))
# Note: In imshow, x is column index, y is row index.
grass_base_paths = [
    [(0,0), (1,0), (2,0), (3,0), (4,0), (4,1), (5,1), (5,2), (6,2), (6,3), (7,3), (7,4), (7,5), (7,6), (7,7)],
    [(0,0), (0,1), (1,1), (2,1), (3,1), (4,1), (5,1), (6,1), (7,1), (7,2), (7,3), (7,4), (7,5), (7,6), (7,7)],
    [(0,0), (1,0), (2,0), (3,0), (4,0), (5,0), (6,0), (7,0), (7,1), (7,2), (7,3), (7,4), (7,5), (7,6), (7,7)],
    [(0,0), (1,0), (1,1), (2,1), (3,1), (4,1), (5,2), (6,2), (6,3), (7,3), (7,4), (7,5), (7,6), (7,7)],
    [(0,0), (0,1), (1,1), (2,0), (3,0), (4,0), (5,0), (6,0), (6,1), (6,2), (7,2), (7,3), (7,4), (7,5), (7,7)]
]

rock_base_paths = [
    [(0,0), (0,1), (0,2), (0,3), (0,4), (1,4), (1,5), (2,5), (2,6), (3,6), (4,6), (5,6), (6,6), (7,6), (7,7)],
    [(0,0), (0,1), (0,2), (0,3), (0,4), (0,5), (0,6), (0,7), (1,7), (2,7), (3,7), (4,7), (5,7), (6,7), (7,7)],
    [(0,0), (0,1), (0,2), (0,3), (1,3), (1,4), (2,4), (2,5), (3,5), (3,6), (4,6), (5,6), (6,6), (7,6), (7,7)],
    [(0,0), (0,1), (0,2), (0,3), (0,4), (0,5), (1,5), (1,6), (2,6), (3,6), (4,6), (5,7), (6,7), (7,7)],
    [(0,0), (0,1), (0,2), (1,2), (1,3), (1,4), (2,4), (2,5), (2,6), (3,7), (4,7), (5,7), (6,7), (7,7)]
]

# 3. Plot the trajectories with slight random jitter to prevent perfect overlap
np.random.seed(42)

# Plot Grass-Lover Trajectories (Green)
for path in grass_base_paths:
    x_coords = [p[0] + np.random.uniform(-0.15, 0.15) for p in path]
    y_coords = [p[1] + np.random.uniform(-0.15, 0.15) for p in path]
    # Force start and end to be exact
    x_coords[0], y_coords[0] = 0, 0
    x_coords[-1], y_coords[-1] = 7, 7
    ax.plot(x_coords, y_coords, color='#229954', linewidth=2.5, alpha=0.85)

# Plot Rock-Lover Trajectories (Dark Gray/Black)
for path in rock_base_paths:
    x_coords = [p[0] + np.random.uniform(-0.15, 0.15) for p in path]
    y_coords = [p[1] + np.random.uniform(-0.15, 0.15) for p in path]
    # Force start and end to be exact
    x_coords[0], y_coords[0] = 0, 0
    x_coords[-1], y_coords[-1] = 7, 7
    ax.plot(x_coords, y_coords, color='#34495E', linewidth=2.5, alpha=0.85)


# 4. Create Custom Legend
legend_elements = [
    mpatches.Patch(facecolor='#FDFEFE', edgecolor='black', label='Normal Tiles'),
    mpatches.Patch(facecolor='#A9DFBF', edgecolor='black', label='Grass Tiles'),
    mpatches.Patch(facecolor='#D5DBDB', edgecolor='black', label='Rock Tiles'),
    mpatches.Patch(facecolor='#5DADE2', edgecolor='black', label='Water (Hard Constraint)'),
    plt.Line2D([0], [0], color='#229954', lw=3, label='Expert 1 (Grass-Lover) Paths'),
    plt.Line2D([0], [0], color='#34495E', lw=3, label='Expert 2 (Rock-Lover) Paths')
]

# Place legend cleanly outside the main grid
ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1),
          fontsize=12, frameon=True, title="Legend", title_fontsize='14')

plt.title('Multi-Feature 8x8 Gridworld\nHeterogeneous Expert Trajectories', fontsize=16, pad=20)
plt.tight_layout()

# Save and Show
plt.savefig('gridworld_heterogeneous_experts.png', dpi=300, bbox_inches='tight')
plt.show()