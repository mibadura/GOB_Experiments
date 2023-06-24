import matplotlib.pyplot as plt
import sys
from shapely.geometry import Point, Polygon, LineString
import time


class WorldState:

    def __init__(self, player_x, player_y, actions):
        self.x = player_x
        self.y = player_y
        self.actions = actions
        self.action_index = 0

    def copy(self):
        return WorldState(self.x, self.y, self.actions)

    def apply_action(self, action):
        self.x += action.dx
        self.y += action.dy

    def next_action(self):
        if self.action_index < len(self.actions):
            next_action = self.actions[self.action_index]
            self.action_index += 1
            return next_action, self.action_index-1
        else:
            return None, None

    def __hash__(self):
        return hash((self.x, self.y))

    def __str__(self):
        return f"({self.x}, {self.y})"

    def __eq__(self, other):
        if isinstance(other, WorldState):
            return self.x == other.x and self.y == other.y and self.action_index == other.action_index
        return False


class Goal:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def is_fulfilled(self, state):
        return self.x == state.x and self.y == state.y


class Action:
    def __init__(self, dx, dy):
        self.dx = dx
        self.dy = dy

    @staticmethod
    def get_cost():
        return 1

    @staticmethod
    def get_all_actions():
        return [
            Action(-1, 0),  # Move left
            Action(1, 0),   # Move right
            Action(0, -1),  # Move down
            Action(0, 1),   # Move up
        ]


class Heuristic:
    def __init__(self, goal):
        self.goal = goal

    def estimate(self, state):
        return abs(self.goal.x - state.x) + abs(self.goal.y - state.y)


class TranspositionTable:
    def __init__(self):
        self.visited = {}

    def has(self, state, depth):
        return state in self.visited and self.visited[state] <= depth

    def add(self, state, depth):
        if state not in self.visited or depth < self.visited[state]:
            print(f'\t\ttranspo:\tadding state {state} at depth: {depth}')
            self.visited[state] = depth


class Obstacles:
    def __init__(self):
        self.obstacle_polygons = [
            Polygon([(-5, 3), (-5, 8), (-3, 8), (-3, 6), (-1, 6), (-1, 4), (-3, 4), (-3, 3)]),
            Polygon([(-3, 1), (-3, 4), (-1, 4), (-1, 3), (2, 3), (2, 1)]),
            Polygon([(-1, 8), (-1, 11), (-11, 11), (-11, 8)])
        ]

    @staticmethod
    def is_point_on_polygon(point, polygon):
        point = Point(point)
        for line in zip(list(polygon.exterior.coords)[:-1], list(polygon.exterior.coords)[1:]):
            if Point(line[0]).distance(point) + Point(line[1]).distance(point) == LineString(line).length:
                return True
        return False

    def is_point_in_polygon(self, point):
        return any(polygon.contains(Point(point)) and not self.is_point_on_polygon(point, polygon) for polygon in self.obstacle_polygons)

    def get_obstacle_polygons(self):
        return self.obstacle_polygons


def plan_action(world_model, goal, heuristic, max_depth,  obstacles = None):
    cutoff = heuristic.estimate(world_model)

    while cutoff != float('inf'):
        transposition_table = TranspositionTable()
        print('\n', '-'*20, f'plan_action: new plan action - cutoff: {cutoff}', '-'*20, '\n')
        cutoff, actions = do_depth_first(world_model, goal, heuristic, transposition_table, max_depth, cutoff, obstacles=obstacles)
        if actions:
            print("plan_action: action plan returned")
            return actions
    return None


def do_depth_first(world_model, goal, heuristic, transposition_table, max_depth, cutoff, obstacles = None ):
    print(f'\n\tdo_depth_first: new do_depth_first')
    max_depth += 1

    states = [None] * (max_depth + 1)
    actions = [None] * max_depth
    costs = [0.0] * max_depth

    world_model.action_index = 0
    states[0] = world_model
    current_depth = 0

    smallest_cutoff = float('inf')

    while current_depth >= 0:

        cost = costs[current_depth] + heuristic.estimate(states[current_depth])
        print(f'\tdo_depth_first:\tdepth: {current_depth};\tstate {states[current_depth]};\tcost {cost};\tcutoff: {cutoff}')

        if obstacles.is_point_in_polygon((states[current_depth].x, states[current_depth].y)):
            print(f'\tdo_depth_first: in polygon! {(states[current_depth].x, states[current_depth].y)}')
            current_depth -= 1
            continue

        if goal.is_fulfilled(states[current_depth]):
            print('\n','-'*10,f' Goal {states[current_depth]} is fulfilled ','-'*10)
            return cutoff, actions

        if current_depth >= max_depth - 1:
            print(f'\tdo_depth_first: current_depth ({current_depth}) too deep and goal not reached. Decreasing depth')
            current_depth -= 1
            continue

        if cost > cutoff:
            if cost < smallest_cutoff:
                smallest_cutoff = cost
            current_depth -= 1
            continue

        next_action, next_action_idx = states[current_depth].next_action()
        if next_action:
            states[current_depth + 1] = states[current_depth].copy()
            states[current_depth + 1].apply_action(next_action)

            actions[current_depth] = next_action
            costs[current_depth + 1] = costs[current_depth] + next_action.get_cost()

            if not transposition_table.has(states[current_depth + 1], current_depth + 1):
                print(f'\n\tdo_depth_first: simulating move {states[current_depth]} -> {states[current_depth + 1]} | new move! adding to transpo table ')
                current_depth += 1
                transposition_table.add(states[current_depth], current_depth)
            else:
                print(f'\n\tdo_depth_first: simulating move {states[current_depth]} -> {states[current_depth + 1]} | already in transpo. trying next move ')

        else:
            current_depth -= 1
            print(f'\tdo_depth_first: no more actions: depth decreased')

    return smallest_cutoff, []


def main():
    orig_stdout = sys.stdout
    f = open('out.txt', 'w')
    sys.stdout = f

    actions = Action.get_all_actions()

    start = WorldState(-9, 0, actions)
    goal = Goal(-2, 11)

    heuristic = Heuristic(goal)
    obstacles = Obstacles()

    start_time = time.time()
    plan = plan_action(start, goal, heuristic, obstacles = obstacles, max_depth=30)
    end_time = time.time()
    print(f'The script execution took {end_time - start_time} seconds.')

    if plan:
        print("Plan found.")
    else:
        print("No plan found.")

    x_positions = [start.x]
    y_positions = [start.y]
    current_state = start

    sys.stdout = orig_stdout
    f.close()

    if plan:
        for action in plan:
            if action is None:
                break
            current_state = current_state.copy()
            current_state.apply_action(action)
            x_positions.append(current_state.x)
            y_positions.append(current_state.y)

        x_positions.append(goal.x)
        y_positions.append(goal.y)

        print(f'Plan length: {len(x_positions)-2}')
        plt.figure(dpi=200)

        polygons = obstacles.get_obstacle_polygons()
        for polygon in polygons:
            x, y = polygon.exterior.xy
            plt.fill(x, y, alpha=0.5, label='Obstacle')  # fill with colors

        plt.plot(x_positions, y_positions, '-', linewidth=2, label='Path')
        plt.scatter([start.x], [start.y], color='g', label='Start')
        plt.scatter([goal.x], [goal.y], color='r', label='Goal')
        plt.grid(True)
        plt.axis('equal')
        plt.xticks(range(-15, 10))
        plt.yticks(range(-3, 15))
        plt.legend()
        plt.show()


if __name__ == "__main__":
    main()
