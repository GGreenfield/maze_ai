import numpy as np

class Cell:
    """A Cell object represents a 1x1px space on a maze which is not a wall,
    i.e. a valid space to be considered in the solution"""
    wall_dic = {"N": "S", "S": "N", "W": "E", "E": "W"}

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.visited = False
        self.walls = {"N": True, "S": True, "W": True, "E": True}
        self.fScore = np.inf

    def __str__(self):
        return "Cell: (" + str(self.x) + "," + str(self.y) + ")"

    def __lt__(self, other):
        return self.fScore < other.fScore

    def remove_wall(self, other, wall):
        """Removes a wall between two cells"""
        self.walls[wall] = False
        other.walls[Cell.wall_dic[wall]] = False
