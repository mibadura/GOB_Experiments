import matplotlib.pyplot as plt
from matplotlib.path import Path
import sys
from shapely.geometry import Point, Polygon, LineString


class WorldState:

    def __init__(self, player_x, player_y, actions):
        self.x = player_x
        self.y = player_y
        self.actions = actions
        self.action_index = 0

    def copy(self):
        """
        Return a copy of the current state.
        """
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

    def get_cost(self):
        return 1

    @staticmethod
    def get_all_actions():
        """
        Return a list of all possible actions.
        """
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

    def has(self, state):
        """
        Check if the state is already in the transposition table.
        state: an instance of State.
        """
        return state in self.visited

    def add(self, state, depth):
        """
        Add a state to the transposition table or update its depth if it's already in the table.
        state: an instance of State.
        depth: depth of the state.
        """
        if state not in self.visited or depth < self.visited[state]:
            self.visited[state] = depth


class Obstacles:
    def __init__(self):
        self.obstacle_polygons = [
            Polygon([(-5, 3), (-5, 8), (-3, 8), (-3, 6), (-1, 6), (-1, 4), (-3, 4), (-3, 3)]),
            Polygon([(-3, 1), (-3, 4), (-1, 4),(-1, 3), (2, 3), (2, 1)]),
            Polygon([(3, 8), (3, 10), (-7, 10), (-7, 8)])
        ]

    def is_point_on_polygon(self, point, polygon):
        point = Point(point)
        for line in zip(list(polygon.exterior.coords)[:-1], list(polygon.exterior.coords)[1:]):
            if Point(line[0]).distance(point) + Point(line[1]).distance(point) == LineString(line).length:
                return True
        return False

    def is_point_in_polygon(self, point):
        """
        Check if a point is inside a polygon.

        Args:
        point (tuple): (x, y) pair representing the point.

        Returns:
        bool: True if the point is in the polygon and not on its boundary, False otherwise.
        """
        return any(polygon.contains(Point(point)) and not self.is_point_on_polygon(point, polygon) for polygon in self.obstacle_polygons)

    def get_obstacle_polygons(self):
        return self.obstacle_polygons


def plan_action(world_model, goal, heuristic, transposition_table, max_depth,  obstacles = None):
    """
    This function uses Iterative Deepening A* (IDA*) to plan actions.
    It returns the first best action found that meets the cutoff heuristic.
    """

    # Initial cutoff is the heuristic from the start model
    cutoff = heuristic.estimate(world_model)

    while cutoff != float('inf'):
        # Conduct a depth-limited depth-first search and update cutoff and action
        print(f'\n\n---PLAN ACTION--')
        print(f'Current cutoff: {cutoff}')
        cutoff, actions = do_depth_first(world_model, goal, heuristic, transposition_table, max_depth, cutoff, obstacles=obstacles)
        # If an action has been found, return it
        if actions:
            print("plan_action: action plan returned")
            return actions
    return None


def do_depth_first(world_model, goal, heuristic, transposition_table, max_depth, cutoff, obstacles = None ):
    print(f'\ndo_depth_first: New do_depth_first')
    max_depth += 1

    # if obstacles:
    #     print(obstacles.get_obstacle_polygons())
    """
    This function performs a depth-limited depth-first search with iterative deepening and returns the smallest cutoff
    encountered and the corresponding best action.
    """
    # Initialize storage for states at each depth, and actions and costs corresponding to them
    states = [None] * (max_depth + 1)
    actions = [None] * max_depth
    costs = [0.0] * max_depth

    # Set up the initial data
    world_model.action_index = 0
    states[0] = world_model
    current_depth = 0

    # Keep track of the smallest pruned cutoff
    smallest_cutoff = float('inf')

    # Iterate until all actions at depth zero are completed
    while current_depth >= 0:
        # If the goal is fulfilled by the current state, return the cutoff and the corresponding action
        if goal.is_fulfilled(states[current_depth]):
            print('-'*10,f' Goal {states[current_depth]} is fulfilled ','-'*10)
            return cutoff, actions


        # Calculate total cost of plan including heuristic estimate
        cost = costs[current_depth] + heuristic.estimate(states[current_depth])
        print('do_depth_first:',f'\tDepth: {current_depth};\tstate {states[current_depth]};\tcost {cost};\tcutoff: {cutoff}')

        if obstacles.is_point_in_polygon((states[current_depth].x, states[current_depth].y)):
            print(f'do_depth_first: in polygon! {(states[current_depth].x, states[current_depth].y)}')
            current_depth -= 1
            continue

        if current_depth >= max_depth - 1:
            print(f'do_depth_first: current_depth ({current_depth}) >= max_depth-1 and goal not reached. Decreasing depth')
            current_depth -= 1
            continue


        # If the cost exceeds the cutoff, move back up the tree and update smallest cutoff if necessary
        if cost > cutoff:
            if cost < smallest_cutoff:
                smallest_cutoff = cost
                # print('-'*current_depth,f'Smallest cutoff updated to {smallest_cutoff}')
            current_depth -= 1
            # print('-'*(current_depth+1),f'Cost {cost} bigger than cutoff {cutoff}. Decreasing depth to {current_depth}')
            continue

        # Try the next action
        next_action, next_action_idx = states[current_depth].next_action()
        if next_action:
            # print('-'*current_depth,f'Next action on depth {current_depth}')
            # Copy the current state and apply the action to the copy
            states[current_depth + 1] = states[current_depth].copy()
            # print('-'*current_depth,f"State[{current_depth + 1}.{next_action_idx}] ", states[current_depth + 1])
            states[current_depth + 1].apply_action(next_action)
            # print('-'*current_depth,f"State[{current_depth + 1}.{next_action_idx}] after action ", states[current_depth + 1])

            # Update action and cost lists
            actions[current_depth] = next_action
            costs[current_depth + 1] = costs[current_depth] + next_action.get_cost()

            # Process the new state if it hasn't been seen before
            if not transposition_table.has(states[current_depth + 1]):
                current_depth += 1
            else:
                print(f'do_depth_first: state {states[current_depth + 1]} alredy in transpo table')

            # Add the new state to the transposition table
            transposition_table.add(states[current_depth + 1], current_depth)

        else:
            # If there are no more actions to try, move back up the tree
            current_depth -= 1
            print(f'do_depth_first: no more actions: depth decreased')
            # print('-'*current_depth,f'No more actions, decreasing depth to {current_depth}')

    # If no action is found after searching all states, return the smallest cutoff encountered
    return smallest_cutoff, []


def main():

    #FOR SAVING OUTPUT TO TEXT FILE
    orig_stdout = sys.stdout
    f = open('out.txt', 'w')
    sys.stdout = f

    # Create actions
    actions = Action.get_all_actions()

    # Initial state
    start = WorldState(0, 0, actions)
    goal = Goal(-2, 7)

    # Heuristic function
    heuristic = Heuristic(goal)

    # Create transposition table
    transposition_table = TranspositionTable()

    obstacles = Obstacles()

    # Plan action
    plan = plan_action(start, goal, heuristic, transposition_table, obstacles = obstacles, max_depth=20)

    # If a plan was found, print it
    if plan:
        print("Plan found.")
    else:
        print("No plan found.")

    # Gather positions for plotting
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

        # Add goal position
        x_positions.append(goal.x)
        y_positions.append(goal.y)

        print(f'Plan length: {len(x_positions)-2}')

        plt.figure(dpi=200)  # Set dpi to 300

        # Create obstacle polygons
        polygons = obstacles.get_obstacle_polygons()
        for polygon in polygons:
            # Obtain x and y coordinates
            x, y = polygon.exterior.xy
            plt.fill(x, y, alpha=0.5, label='Obstacle')  # fill with colors

        # Create plot
        plt.plot(x_positions, y_positions, '-', linewidth=2, label='Path')
        plt.scatter([start.x], [start.y], color='g', label='Start')
        plt.scatter([goal.x], [goal.y], color='r', label='Goal')
        plt.grid(True)
        plt.axis('equal')
        plt.xticks(range(-10, 10))
        plt.yticks(range(-1, 15))
        plt.legend()
        plt.show()


if __name__ == "__main__":
    main()
