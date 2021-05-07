import math
from heapq import *


class Map:

    # Default constructor
    def __init__(self):
        self.width = 0
        self.height = 0
        self.cells = []

    # Converting a string (with '#' representing obstacles and '.' representing free cells) to a grid
    def ReadFromString(self, cellStr, width, height):
        self.width = width
        self.height = height
        self.cells = [[0 for _ in range(width)] for _ in range(height)]
        cellLines = cellStr.split("\n")
        i = 0
        j = 0
        for l in cellLines:
            if len(l) != 0:
                j = 0
                for c in l:
                    if c == '.':
                        self.cells[i][j] = 0
                    elif c == '#':
                        self.cells[i][j] = 1
                    else:
                        continue
                    j += 1
                # TODO
                if j != width:
                    raise Exception("Size Error. Map width = ", j, ", but must be", width)

                i += 1

        if i != height:
            raise Exception("Size Error. Map height = ", i, ", but must be", height)

    # Initialization of map by list of cells.
    def SetGridCells(self, width, height, gridCells):
        self.width = width
        self.height = height
        self.cells = gridCells

    # Check if the cell is on a grid.
    def inBounds(self, i, j):
        return (0 <= j < self.width) and (0 <= i < self.height)

    # Check if thec cell is not an obstacle.
    def Traversable(self, i, j):
        return not self.cells[i][j]

    # Get a list of neighbouring cells as (i,j) tuples.
    # It's assumed that grid is 4-connected (i.e. only moves into cardinal directions are allowed)
    def GetNeighbors(self, i, j):
        neighbors = []
        delta = [[0, 1], [1, 0], [0, -1], [-1, 0]]

        for d in delta:
            if self.inBounds(i + d[0], j + d[1]) and self.Traversable(i + d[0], j + d[1]):
                neighbors.append((i + d[0], j + d[1]))

        return neighbors


class Node:
    def __init__(self, i=-1, j=-1, g=math.inf, h=math.inf, F=None, parent=None):
        self.i = i
        self.j = j
        self.g = g
        self.h = h
        if F is None:
            self.F = self.g + h
        else:
            self.F = F
        self.parent = parent

    def __eq__(self, other):
        return (self.i == other.i) and (self.j == other.j)

    def __lt__(self, other):
        return self.F < other.F or ((self.F == other.F) and (self.h < other.h)) \
               or ((self.F == other.F) and (self.h == other.h) and (self.k > other.k))


class YourOpen:
    def __init__(self):
        self.heap = []
        self.coord2node = {}

    def __iter__(self):
        return iter(self.coord2node.values())

    def __len__(self):
        return len(self.coord2node)

    def isEmpty(self):
        return len(self.coord2node) == 0

    def AddNode(self, item: Node):
        coord = (item.i, item.j)
        oldNode = self.coord2node.get(coord)

        if oldNode is None or item.g < oldNode.g:
            self.coord2node[coord] = item
            heappush(self.heap, item)

    def GetBestNode(self):
        bestNode = heappop(self.heap)
        coord = (bestNode.i, bestNode.j)

        while self.coord2node.pop(coord, None) is None:
            bestNode = heappop(self.heap)
            coord = (bestNode.i, bestNode.j)

        return bestNode


class YourClosed:
    def __init__(self):
        self.elements = set()

    def __iter__(self):
        return iter(self.elements)

    def __len__(self):
        return len(self.elements)

    # AddNode is the method that inserts the node to CLOSED
    def AddNode(self, item: Node, *args):
        self.elements.add((item.i, item.j))

    # WasExpanded is the method that checks if a node has been expanded
    def WasExpanded(self, item: Node, *args):
        return (item.i, item.j) in self.elements


# Creating a path by tracing parent pointers from the goal node to the start node
# It also returns path's length.
def MakePath(goal):
    length = goal.g
    current = goal
    path = []
    while current.parent:
        path.append(current)
        current = current.parent
    path.append(current)
    return path[::-1], length


def ManhattanDistance(i1, j1, i2, j2):
    # Implement Manhattan Distance as a heuristic
    return abs(i1 - i2) + abs(j1 - j2)


def CalculateCost(i1, j1, i2, j2):
    return math.sqrt((i1 - i2) ** 2 + (j1 - j2) ** 2)


def AStar(gridMap, iStart, jStart, iGoal, jGoal, heuristicFunction=ManhattanDistance):
    OPEN = YourOpen()
    CLOSED = YourClosed()
    pathIsFound = False
    lastNode = None

    h = heuristicFunction(iStart, jStart, iGoal, jGoal)
    start = Node(iStart, jStart, g=0, h=h, parent=None)
    goal = Node(iGoal, jGoal)

    OPEN.AddNode(start)

    k = 1

    while not OPEN.isEmpty():
        cur = OPEN.GetBestNode()
        CLOSED.AddNode(cur)

        if cur == goal:
            lastNode = cur
            pathIsFound = True
            break

        for i, j in gridMap.GetNeighbors(cur.i, cur.j):
            neighb = Node(i, j)

            if not CLOSED.WasExpanded(neighb):
                neighb.g = cur.g + CalculateCost(cur.i, cur.j, i, j)
                neighb.h = heuristicFunction(i, j, iGoal, jGoal)
                neighb.F = neighb.g + neighb.h
                neighb.k = k
                neighb.parent = cur
                OPEN.AddNode(neighb)
                k += 1

    return pathIsFound, lastNode, CLOSED, OPEN
