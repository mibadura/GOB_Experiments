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
            Polygon([(-1, 8), (-1, 9), (-11, 9), (-11, 8)])
        ]

    @staticmethod
    def is_point_on_polygon(point, polygon):
        point = Point(point)
        for line in zip(list(polygon.exterior.coords)[:-1], list(polygon.exterior.coords)[1:]):
            if Point(line[0]).distance(point) + Point(line[1]).distance(point) == LineString(line).length:
                return True
        return False

    def is_point_in_polygon(self, point):
        return any(polygon.contains(Point(point)) and
                   not self.is_point_on_polygon(point, polygon) for polygon in self.obstacle_polygons)

    def get_obstacle_polygons(self):
        return self.obstacle_polygons


def plan_action(world_model, goal, heuristic, max_depth,  obstacles = None):
    cutoff = heuristic.estimate(world_model)
    goal_reached = False
    while cutoff != float('inf'):
        transposition_table = TranspositionTable()
        print('\n', '-'*20, f'plan_action: new plan action - cutoff: {cutoff}', '-'*20, '\n')
        cutoff, actions, goal_reached = do_depth_first(world_model, goal, heuristic, transposition_table, max_depth,
                                                       cutoff, obstacles=obstacles)
        if actions:
            if goal_reached:
                print(f'plan_action: goal reached')
                return actions, goal_reached
            else:
                print(f'plan_action: goal not reached')
                print(f'best plan so far:')
                for action in actions:
                    print(f'{action.dx, action.dy}')

    return actions, goal_reached


def do_depth_first(world_model, goal, heuristic, transposition_table, max_depth, cutoff, obstacles=None):
    print(f'\n\tdo_depth_first: new do_depth_first')
    max_depth += 1

    states = [None] * (max_depth + 1)
    actions = [None] * max_depth
    costs = [0.0] * max_depth

    world_model.action_index = 0
    states[0] = world_model
    current_depth = 0

    transposition_table.add(states[current_depth], current_depth)   # add the starting point to transposition table

    smallest_cutoff = float('inf')

    best_path = None  # New variable for best path
    best_cost = float('inf')  # New variable for best cost

    while current_depth >= 0:

        heuristic_estimate = heuristic.estimate(states[current_depth])
        cost = costs[current_depth] + heuristic_estimate
        print(f'\tdo_depth_first:\tdepth: {current_depth};\tstate {states[current_depth]};\tcost {cost};'
              f'\tcutoff: {cutoff}')

        if obstacles.is_point_in_polygon((states[current_depth].x, states[current_depth].y)):
            print(f'\tdo_depth_first: in polygon! {(states[current_depth].x, states[current_depth].y)}')
            current_depth -= 1
            continue

        if goal.is_fulfilled(states[current_depth]):
            print('\n','-'*10,f' Goal {states[current_depth]} is fulfilled ','-'*10)
            return cutoff, actions, True

        if heuristic_estimate < best_cost:
            best_cost = heuristic_estimate
            best_path = actions[:current_depth]

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
                print(f'\n\tdo_depth_first: simulating move {states[current_depth]} -> {states[current_depth + 1]} '
                      f'| new move! adding to transpo table ')
                current_depth += 1
                transposition_table.add(states[current_depth], current_depth)
            else:
                print(f'\n\tdo_depth_first: simulating move {states[current_depth]} -> {states[current_depth + 1]} '
                      f'| already in transpo. trying next move ')

            # Check is midpoint is in obstacle (prevents crossing 1-wide obstacles)
            midpoint = ((states[current_depth].x + states[current_depth - 1].x) / 2, (states[current_depth].y +
                                                                                      states[current_depth - 1].y) / 2)
            if obstacles.is_point_in_polygon(midpoint):
                print(f'\tdo_depth_first: midpoint in polygon! {midpoint}')
                current_depth -= 1
                continue

        else:
            current_depth -= 1
            print(f'\tdo_depth_first: no more actions: depth decreased')

    return smallest_cutoff, best_path, False


def main(_max_depth):
    main_start_time = time.time()
    orig_stdout = sys.stdout
    f = open('out.txt', 'w')
    sys.stdout = f

    actions = Action.get_all_actions()

    start = WorldState(0, 0, actions)
    goal = Goal(-5, 10)
    max_plan_loops = 10

    heuristic = Heuristic(goal)
    obstacles = Obstacles()

    goal_reached = False
    plan_loop_idx = 0
    full_plan = []

    x_positions = [start.x]
    y_positions = [start.y]
    starting_points_x = [start.x]
    starting_points_y = [start.y]
    current_state = start

    while not goal_reached and plan_loop_idx < max_plan_loops:

        start_time = time.time()
        single_plan, goal_reached = plan_action(start, goal, heuristic, obstacles=obstacles, max_depth=_max_depth)
        end_time = time.time()
        print(f'Plan_action execution took {end_time - start_time} seconds.')

        if single_plan and not goal_reached:
            print("Partial plan found.")
        elif single_plan and goal_reached:
            print("Final plan found.")
        else:
            print('No plan at all.')
            break

        if single_plan:
            for action in single_plan:
                if action is None:
                    break
                current_state = current_state.copy()
                current_state.apply_action(action)
                x_positions.append(current_state.x)
                y_positions.append(current_state.y)

        start = current_state.copy()
        starting_points_x.append(start.x)
        starting_points_y.append(start.y)

    if goal_reached:
        x_positions.append(goal.x)
        y_positions.append(goal.y)

    print(f'Plan length: {len(x_positions)-2}')
    plt.figure(dpi=200)

    polygons = obstacles.get_obstacle_polygons()
    for polygon in polygons:
        x, y = polygon.exterior.xy
        plt.fill(x, y, alpha=0.5, label='Obstacle')  # fill with colors

    plt.plot(x_positions, y_positions, '-', linewidth=2, label='Path')
    plt.scatter([starting_points_x], [starting_points_y], color='g', label='Startpoints')
    plt.scatter([goal.x], [goal.y], color='r', label='Goal')
    plt.grid(True)
    plt.axis('equal')
    plt.xticks(range(-15, 10))
    plt.yticks(range(-3, 15))
    plt.legend()
    # plt.show()
    plt.savefig(f'./PointSearchingFigs/PointSearching_GOAPwithIDAstar_MaxDepth-{_max_depth}.jpg', dpi=200)
    plt.close()

    sys.stdout = orig_stdout
    f.close()
    main_end_time = time.time()
    return main_end_time-main_start_time


if __name__ == "__main__":
    main(20)
