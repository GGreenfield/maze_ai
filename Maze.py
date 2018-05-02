import Cell
import random
import numpy as np
from PIL import Image
import heapq


class Maze:
    """A Maze is a collection of connected nodes represented by black and white pixels
    (e.g. https://i.pinimg.com/originals/89/3e/44/893e44c9caee4a47a2df520f818483fc.jpg)"""

    def __init__(self, dimension):
        self.dimension = dimension  # dimension is the width x height of the maze
        self.maze_grid = [[Cell.Cell(x, y) for x in range(dimension)] for y in range(dimension)]
        self.start = Cell.Cell(0, 0)  # start cell of the maze
        self.maze_grid[0][0] = self.start
        self.end = Cell.Cell(dimension - 1, dimension - 1)  # end cell of the maze
        self.maze_grid[dimension - 1][dimension - 1] = self.end
        self.matrix = np.zeros([(2 * dim) + 1, (2 * dim) + 1], dtype=np.uint8)  # init matrix to all 0s

    def __str__(self):
        s = ""
        for i in range(self.dimension):
            for j in range(self.dimension):
                s += str(self.get_cell(i, j)) + ", "
            s += "\n"
        return s

    def get_cell(self, x, y):
        """returns Cell at specific index in self.maze_grid"""
        if x > self.dimension - 1:
            x = self.dimension - 1
        if y > self.dimension - 1:
            y = self.dimension - 1
        return self.maze_grid[y][x]

    def get_neighbors(self, x, y):
        """returns a list of unvisited neighbors of a cell at a specific index"""
        neighbors = []

        if 0 <= x < self.dimension and 0 <= y < self.dimension:
            # check north neighbor
            if not self.get_cell(x - 1, y).visited and 0 <= x - 1 < self.dimension:
                neighbors.append((self.get_cell(x - 1, y), "W"))
            # check south neighbor
            if not self.get_cell(x + 1, y).visited and 0 <= x + 1 < self.dimension:
                neighbors.append((self.get_cell(x + 1, y), "E"))
            # check east neighbor
            if not self.get_cell(x, y - 1).visited and 0 <= y - 1 < self.dimension:
                neighbors.append((self.get_cell(x, y - 1), "N"))
            # check west neighbor
            if not self.get_cell(x, y + 1).visited and 0 <= y + 1 < self.dimension:
                neighbors.append((self.get_cell(x, y + 1), "S"))

        if neighbors:
            return neighbors

        return None

    def gen_backtrack(self):
        """Generates a maze by visiting every node in random order,
        removing walls between Cells"""
        print("Creating maze with wall removal backtracking")
        n = self.dimension
        self.start.visited = True
        stack = []
        current_cell = self.get_cell(0, 0)
        visited_count = 1

        # loop until all Cells have been visited
        while visited_count < n * n:
            current_cell.visited = True
            neighbors = self.get_neighbors(current_cell.x, current_cell.y)
            # if no valid neighbors exist, backtrack until there are valid neighbors
            while not neighbors:
                current_cell = stack.pop()
                neighbors = self.get_neighbors(current_cell.x, current_cell.y)

            next_cell, direction = random.choice(neighbors)
            current_cell.remove_wall(next_cell, direction)
            stack.append(current_cell)
            current_cell = next_cell
            visited_count += 1

        # reset visited
        for row in maze.maze_grid:
            for cell in row:
                cell.visited = False


    def astar(self):
        """Returns path found from start node to end node"""
        visited = set((self.start.x, self.start.y))
        fringe = [self.start]
        heapq.heapify(fringe)
        cameFrom = {}
        gScore = {self.get_cell(i, j): np.inf for i in range(dim) for j in range(dim)}  # initially inf for all nodes
        gScore[self.start] = 0
        fScore = {self.get_cell(i, j): self.get_cell(i, j).fScore for i in range(dim) for j in
                  range(dim)}  # initially inf for all nodes
        fScore[self.start] = self.heuristic(self.start)

        while fringe:
            current = heapq.heappop(fringe)
            if current == self.end:
                return self.construct_path(cameFrom, current)
            visited.add((current.x, current.y))
            neighbors = self.get_neighbors(current.x, current.y)

            for neighbor in neighbors:
                for d in current.walls.keys():
                    if neighbor[0].walls[Cell.Cell.wall_dic[d]] == False == current.walls[d]:
                        if (neighbor[0].x, neighbor[0].y) in visited:
                            continue

                        if neighbor[0] not in fringe:
                            heapq.heappush(fringe, neighbor[0])

                        tentativeG = gScore[current] + 1
                        if tentativeG >= gScore[neighbor[0]]:
                            continue

                        cameFrom[neighbor[0]] = current
                        gScore[neighbor[0]] = tentativeG
                        neighbor[0].fScore = gScore[neighbor[0]] + self.heuristic(neighbor[0])
                        fScore[neighbor[0]] = gScore[neighbor[0]] + self.heuristic(neighbor[0])
        return -1

    def construct_path(self, came_from, current):
        path = [current]
        while current in came_from.keys():
            current = came_from[current]
            path.append(current)
        return path

    def make_matrix(self):
        """Constructs a 2-D matrix representing a Maze in which each Cell is surrounded
        by N,S,E,W walls between the neighboring cells"""
        for x in range(dim):
            for y in range(dim):
                self.matrix[(2 * x) + 1][(2 * y) + 1] = 1

                if maze.get_cell(x, y).walls['N']:
                    try:
                        self.matrix[(2 * x) + 1][(2 * y)] = 0
                    except IndexError:
                        pass
                elif not maze.get_cell(x, y).walls['N']:
                    try:
                        self.matrix[(2 * x) + 1][(2 * y)] = 1
                    except IndexError:
                        pass
                if maze.get_cell(x, y).walls['S']:
                    try:
                        self.matrix[(2 * x) + 1][(2 * y) + 2] = 0
                    except IndexError:
                        pass
                elif not maze.get_cell(x, y).walls['S']:
                    try:
                        self.matrix[(2 * x) + 1][(2 * y) + 2] = 1
                    except IndexError:
                        pass
                if maze.get_cell(x, y).walls['E']:
                    try:
                        self.matrix[(2 * x) + 2][(2 * y) + 1] = 0
                    except IndexError:
                        pass
                elif not maze.get_cell(x, y).walls['E']:
                    try:
                        self.matrix[(2 * x) + 2][(2 * y) + 1] = 1
                    except IndexError:
                        pass
                if maze.get_cell(x, y).walls['W']:
                    try:
                        self.matrix[(2 * x)][(2 * y) + 1] = 0
                    except IndexError:
                        pass
                elif not maze.get_cell(x, y).walls['W']:
                    try:
                        self.matrix[(2 * x)][(2 * y) + 1] = 1
                    except IndexError:
                        pass

    def update_matrix_path(self):
        path = self.astar()
        for point in path:
            self.matrix[(2 * point.x) + 1][(2 * point.y) + 1] = 2
            if not maze.get_cell(point.x, point.y).walls['N']:
                try:
                    self.matrix[(2 * point.x) + 1][(2 * point.y)] = 2
                except IndexError:
                    pass
            if not maze.get_cell(point.x, point.y).walls['S']:
                try:
                    self.matrix[(2 * point.x) + 1][(2 * point.y) + 2] = 2
                except IndexError:
                    pass
            if not maze.get_cell(point.x, point.y).walls['E']:
                try:
                    self.matrix[(2 * point.x) + 2][(2 * point.y) + 1] = 2
                except IndexError:
                    pass
            if not maze.get_cell(point.x, point.y).walls['W']:
                try:
                    self.matrix[(2 * point.x)][(2 * point.y) + 1] = 2
                except IndexError:
                    pass

    def draw_maze(self):
        """Draws a black or white pixel based on the value at each y, x in the matrix"""
        # mark entry and exit for image visualization
        self.matrix[1][0] = 2
        self.matrix[2 * dim - 1][2 * dim] = 3

        for y in range(imgY):
            for x in range(imgX):
                pixels[x, y] = colors[self.matrix[(2 * dim + 1) * y // imgY][(2 * dim + 1) * x // imgX]]

    # Manhattan distance
    def heuristic(self, current):
        return abs(current.x - self.end.x) + abs(current.y - self.end.y)

    # Chebyshev distance
    # def heuristic(self, current):
    #    return max(abs(current.x - self.end.x), abs(current.y - self.end.y))

    # Euclidian distance
    # def heuristic(self, current):
    #    return np.sqrt(np.power((current.x - self.end.x), 2) + np.power((current.y - self.end.y), 2))


if __name__ == "__main__":
    dim = 30  # fix drawing for large mazes
    maze = Maze(dim)
    maze.gen_backtrack()  # generate the maze
    maze.make_matrix()  # convert to np matrix
    imgX = 500
    imgY = 500
    img = Image.new("RGB", (imgX, imgY))
    pixels = img.load()
    colors = [(0, 0, 0), (255, 255, 255), (255, 0, 0), (0, 0, 255)]

    maze.draw_maze()  # output image
    img.save("maze_" + str(2 * dim) + "x" + str(2 * dim) + ".png", "PNG")
    maze.update_matrix_path() # update with solution path
    maze.draw_maze()  # output image with soln
    img.save("maze_" + str(2 * dim) + "x" + str(2 * dim) + "_solved.png", "PNG")



    print(maze.matrix)  # start = matrix[1][1], end = matrix[len-2][len-2]
    #soln_path = maze.astar()
    #print(soln_path)
    #for cell in reversed(soln_path):
    #   print(cell)


