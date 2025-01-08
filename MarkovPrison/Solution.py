import numpy as np
import random
import time
from typing import Dict, Tuple

class Solution:
    def __init__(self, display_moves=True):
        self.grid = [[0 for _ in range(4)] for _ in range(4)]
        self.guard1_pos = None
        self.guard2_pos = None
        self.directions = ["up", "down", "left", "right"]
        self.prisoner_pos = (0, 0)
        self.goal_pos = (3, 3)
        self.guard1_probabilities = {}
        self.guard2_probabilities = {}
        self.iteration = 0
        self.display_moves = display_moves

        # Initialize transition matrices
        self.grid_size = 4
        self.num_positions = self.grid_size * self.grid_size
        self.guard1_transition_matrix = np.zeros((self.num_positions, self.num_positions))
        self.guard2_transition_matrix = np.zeros((self.num_positions, self.num_positions))

        # Initialize guard position distributions
        # Start with uniform distribution excluding prisoner start position
        self.guard1_distribution = np.ones(self.num_positions) / (self.num_positions - 1)
        # self.guard1_distribution[0] = 0  # Exclude (0,0)
        self.guard2_distribution = np.ones(self.num_positions) / (self.num_positions - 1)
        # self.guard2_distribution[0] = 0  # Exclude (0,0)

        # Initialize prisoner and guards
        self._initialize_prisoner()
        self._initialize_guards()

        # Set movement probabilities for guard 1
        self.set_guard_probabilities(1, {
            'up': 0.2,
            'down': 0.4,
            'left': 0.2,
            'right': 0.2
        })

        # Set movement probabilities for guard 2
        self.set_guard_probabilities(2, {
            'up': 0.4,
            'down': 0.1,
            'left': 0.2,
            'right': 0.3
        })

        # For each position in the grid, pre-compute probability of guards moving there, and distance to goal
        self._precompute_guard_risks()
        self._precompute_distances_to_goal()

        # Print initial positions
        print(f"Starting positions:")
        print(f"Guard 1: {self.guard1_pos}")
        print(f"Guard 2: {self.guard2_pos}")
        print(f"Prisoner: {self.prisoner_pos}")
        print(f"Goal: {self.goal_pos}")

    def _position_to_index(self, pos: Tuple[int, int]) -> int:
        """Convert (row, col) position to linear index."""
        return pos[0] * self.grid_size + pos[1]

    def _index_to_position(self, index: int) -> Tuple[int, int]:
        """Convert linear index to (row, col) position."""
        return (index // self.grid_size, index % self.grid_size)

    def _update_transition_matrices(self):
        """Update transition matrices based on guard probabilities."""
        # Clear matrices
        self.guard1_transition_matrix.fill(0)
        self.guard2_transition_matrix.fill(0)

        # Update for both guards
        for i in range(self.num_positions):
            row, col = self._index_to_position(i)

            # Calculate possible moves and their indices
            moves = {
                'up': (row - 1, col) if row > 0 else (row, col),
                'down': (row + 1, col) if row < self.grid_size - 1 else (row, col),
                'left': (row, col - 1) if col > 0 else (row, col),
                'right': (row, col + 1) if col < self.grid_size - 1 else (row, col)
            }

            # Assign probabilities for guard 1
            for direction, new_pos in moves.items():
                new_idx = self._position_to_index(new_pos)
                self.guard1_transition_matrix[i, new_idx] += self.guard1_probabilities.get(direction, 0)

            # Assign probabilities for guard 2
            for direction, new_pos in moves.items():
                new_idx = self._position_to_index(new_pos)
                self.guard2_transition_matrix[i, new_idx] += self.guard2_probabilities.get(direction, 0)


    def _initialize_prisoner(self) -> None:
        """Place prisoner at entry point."""
        self.prisoner_pos = (0, 0)
        self._update_grid()

    def _initialize_guards(self) -> None:
        """Randomly place guards on the grid."""
        available_positions = [
            (i, j) for i in range(4) for j in range(4)
            # if (i, j) != (0, 0)  # Avoid entry point
        ]

        # Randomly select positions for guards
        guard_positions = random.sample(available_positions, 2)

        self.guard1_pos = guard_positions[0]
        self.guard2_pos = guard_positions[1]
        self._update_grid()

    def set_guard_probabilities(self, guard_num: int, probabilities: Dict[str, float]) -> None:
        """Set movement probabilities for a guard."""
        required_directions = {'up', 'down', 'left', 'right'}
        if set(probabilities.keys()) != required_directions:
            raise ValueError("Must provide probabilities for up, down, left, right")
        if not abs(sum(probabilities.values()) - 1.0) < 0.0001:
            raise ValueError("Probabilities must sum to 1")

        if guard_num == 1:
            self.guard1_probabilities = probabilities
        elif guard_num == 2:
            self.guard2_probabilities = probabilities
        else:
            raise ValueError("Invalid guard number")

        self._update_transition_matrices()

    def _precompute_guard_risks(self):
        """Precompute risk contributions for all positions."""
        self.guard1_risk_contributions = self.guard1_distribution @ self.guard1_transition_matrix
        self.guard2_risk_contributions = self.guard2_distribution @ self.guard2_transition_matrix

    def _precompute_distances_to_goal(self):
        """Precompute Manhattan distances to the goal for all grid positions."""
        self.goal_distances = np.zeros((self.grid_size, self.grid_size))
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                self.goal_distances[x, y] = abs(x - self.goal_pos[0]) + abs(y - self.goal_pos[1])

    def _calculate_risk_score(self, position: Tuple[int, int]) -> float:
        """Calculate risk score for a given position."""
        pos_idx = self._position_to_index(position)

        # Use precomputed risk contributions
        guard1_risk = self.guard1_risk_contributions[pos_idx]
        guard2_risk = self.guard2_risk_contributions[pos_idx]

        return guard1_risk + guard2_risk


    def _attempt_move(self, current_pos: Tuple[int, int], direction: str) -> Tuple[int, int]:
        """Calculate new position based on direction, even if illegal."""
        row, col = current_pos
        if direction == 'up':
            return (row - 1, col)
        elif direction == 'down':
            return (row + 1, col)
        elif direction == 'left':
            return (row, col - 1)
        elif direction == 'right':
            return (row, col + 1)
        return current_pos

    def _is_move_legal(self, new_pos: Tuple[int, int]) -> bool:
        """Check if a move is within bounds."""
        return self._is_valid_position(new_pos)

    def get_best_move(self, current_pos: Tuple[int, int]) -> Tuple[int, int]:
        """Determine the safest move based on guard transition probabilities."""
        best_move = current_pos
        lowest_risk = float('inf')

        # Use precomputed distance
        current_dist = self.goal_distances[current_pos]

        # Check all possible moves
        for direction in self.directions:
            new_pos = self._attempt_move(current_pos, direction)
            if self._is_move_legal(new_pos):
                risk = self._calculate_risk_score(new_pos)

                # Update best move if this position has lower risk and is closer
                if risk < lowest_risk and current_dist > self.goal_distances[new_pos]:
                    lowest_risk = risk
                    best_move = new_pos

        return best_move

    def move_prisoner(self):
        """Move prisoner, return True if discovered."""
        # Get best move based on transition matrices
        new_pos = self.get_best_move(self.prisoner_pos)

        # Update prisoner position
        self.prisoner_pos = new_pos
        self._update_grid()

    def move_guard(self, guard_num: int) -> bool:
        """Move specified guard and return True if prisoner is discovered."""
        if guard_num == 1:
            current_pos = self.guard1_pos
            probs = self.guard1_probabilities
        else:
            current_pos = self.guard2_pos
            probs = self.guard2_probabilities

        if not probs:
            raise ValueError(f"Probabilities not set for guard {guard_num}")

        # Choose direction based on probabilities
        direction = random.choices(
            list(probs.keys()),
            list(probs.values())
        )[0]

        # Attempt move
        new_pos = self._attempt_move(current_pos, direction)

        # If move is illegal, stay in current position
        if not self._is_move_legal(new_pos):
            return current_pos == self.prisoner_pos

        # Update guard position
        if guard_num == 1:
            self.guard1_pos = new_pos
        else:
            self.guard2_pos = new_pos

        self._update_grid()
        return new_pos == self.prisoner_pos

    def _is_valid_position(self, position: Tuple[int, int]) -> bool:
        """Check if a position is within grid bounds."""
        row, col = position
        return 0 <= row < 4 and 0 <= col < 4

    def run_iteration(self) -> bool:
        """Run one iteration of movements. Return (continue_sim, result)."""
        self.iteration += 1

        if self.display_moves:
            print(f"\nIteration {self.iteration}:")

        # Move prisoner first
        self.move_prisoner()

        discovered_by_guard1 = self.move_guard(1)
        discovered_by_guard2 = self.move_guard(2)

        # Move guards and check for discovery
        if discovered_by_guard1 or discovered_by_guard2:
            if self.display_moves:
                print("\nYou have been caught by a guard!")
                self.display_grid()
            return False

        # Check if prisoner reached the goal
        if self.prisoner_pos == self.goal_pos:
            if self.display_moves:
                print("\nPrisoner reached the goal!")
                self.display_grid()
            return False

        if self.display_moves:
            self.display_grid()
            time.sleep(1)

        return True, "ongoing"

    def run_simulation(self) -> None:
        """Run simulation until prisoner escapes or is discovered."""
        print("Initial grid:")
        self.display_grid()

        while self.run_iteration():
            time.sleep(1)

    def _update_grid(self) -> None:
        """Update grid to reflect current positions."""
        # Clear grid
        self.grid = [[0 for _ in range(4)] for _ in range(4)]

        # Place entities on grid
        if self.prisoner_pos:
            self.grid[self.prisoner_pos[0]][self.prisoner_pos[1]] = 'P'
        if self.guard1_pos:
            row, col = self.guard1_pos
            self.grid[row][col] = 'G1' if self.grid[row][col] == 0 else f"{self.grid[row][col]}+G1"
        if self.guard2_pos:
            row, col = self.guard2_pos
            self.grid[row][col] = 'G2' if self.grid[row][col] == 0 else f"{self.grid[row][col]}+G2"

    def display_grid(self) -> None:
        """Print the current state of the grid."""
        for i, row in enumerate(self.grid):
            print(f"{' '.join(str(cell) for cell in row)}")  # Grid
        print(f"Guard 1: {self.guard1_pos}, Guard 2: {self.guard2_pos}, Prisoner: {self.prisoner_pos}")

# Example usage
if __name__ == "__main__":
    prison = Solution()

    # Run simulation until goal is reached or prisoner is discovered
    prison.run_simulation()

