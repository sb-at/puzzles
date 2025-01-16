# Puzzle "Somewhat Square Sudoku": https://www.janestreet.com/puzzles/current-puzzle/#fnref:1


class SudokuGrid:

    def __init__(self, legal_values=None):
        """ Initialize a 9x9 grid with specified legal values. List of values that can be placed in the grid.
            Defaults to [0,1,2,3,4,5,6,7,8,9]"""

        if legal_values is None:
            legal_values = list(range(10))
        self.legal_values = legal_values
        self.grid = [[None for _ in range(9)] for _ in range(9)]

    def set_legal_values(self, new_legal_values):
        # Check that values are unique
        if len(new_legal_values) != len(set(new_legal_values)):
            return False

        self.legal_values = new_legal_values

        for i in range(9):
            for j in range(9):
                if self.grid[i][j] not in new_legal_values:
                    self.grid[i][j] = None

        return True

    def display_grid(self):
        """Print the current state of the Sudoku grid"""
        for i, row in enumerate(self.grid):
            if i % 3 == 0 and i != 0:
                print("-" * 21)
            for j, num in enumerate(row):
                if j % 3 == 0 and j != 0:
                    print("|", end=" ")
                print(num if num is not None else " ", end=" ")
            print()

    def update_cell(self, row, col, value):
        """Update a cell in the grid with a given value and check if the grid is still valid"""
        if not (0 <= row <= 8 and 0 <= col <= 8):
            return False
        
        if value not in self.legal_values and value is not None:
            return False

        old_value = self.grid[row][col]
        self.grid[row][col] = value

        # Check only affected sections
        if self.is_valid_update(row, col):
            return True
        else:
            # Revert the change if it creates an invalid state
            self.grid[row][col] = old_value
            return False

    def is_valid_update(self, row, col):
        """ Check if the update at the specified position creates a valid state
        by only checking the affected row, column, and 3x3 box"""
        # Check row
        if not self.is_valid_section(self.grid[row]):
            return False

        # Check column
        column = [self.grid[i][col] for i in range(9)]
        if not self.is_valid_section(column):
            return False

        # Check 3x3 box
        box_row = (row // 3) * 3
        box_col = (col // 3) * 3
        box = []
        for i in range(3):
            for j in range(3):
                box.append(self.grid[box_row + i][box_col + j])
        if not self.is_valid_section(box):
            return False

        return True

    def is_valid_section(self, section):
        """section (list): List of numbers to check (row, column, sub-grid)"""
        filled_cells = [x for x in section if x is not None]
        return len(filled_cells) == len(set(filled_cells))