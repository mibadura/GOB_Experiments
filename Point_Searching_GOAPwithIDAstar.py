import matplotlib.pyplot as plt


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


def plan_action(world_model, goal, heuristic, transposition_table, max_depth):
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
        cutoff, actions = do_depth_first(world_model, goal, heuristic, transposition_table, max_depth, cutoff)
        # If an action has been found, return it
        if actions:
            return actions
    return None


def do_depth_first(world_model, goal, heuristic, transposition_table, max_depth, cutoff):
    print(f'\nNew do_depth_first')
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
            print('-'*10,' Goal is fulfilled ','-'*10)
            return cutoff, actions

        # If we're at maximum depth, move back up the tree
        # print(f'Curr depth before check: {current_depth}')
        if current_depth >= max_depth-1:
            current_depth -= 1
            continue

        # Calculate total cost of plan including heuristic estimate
        cost = costs[current_depth] + heuristic.estimate(states[current_depth])

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
                # print('-'*current_depth,f'Increasing depth to {current_depth}')
            else:
                print(f'State {states[current_depth + 1]} alredy in transpo table')

            # Add the new state to the transposition table
            transposition_table.add(states[current_depth + 1], current_depth)

        else:
            # If there are no more actions to try, move back up the tree
            current_depth -= 1
            # print('-'*current_depth,f'No more actions, decreasing depth to {current_depth}')

    # If no action is found after searching all states, return the smallest cutoff encountered
    return smallest_cutoff, []


def main():

    # Create actions
    actions = Action.get_all_actions()

    # Initial state
    start = WorldState(-2, 0, actions)
    goal = Goal(-4, 7)

    # Heuristic function
    heuristic = Heuristic(goal)

    # Create transposition table
    transposition_table = TranspositionTable()

    # Plan action
    plan = plan_action(start, goal, heuristic, transposition_table, max_depth=13)

    # If a plan was found, print it
    if plan:
        print("Plan found.")
    else:
        print("No plan found.")

    # Gather positions for plotting
    x_positions = [start.x]
    y_positions = [start.y]
    current_state = start
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

        print(f'x positions: {x_positions}')

        # Create plot
        plt.plot(x_positions, y_positions, '-', label='Path')
        plt.scatter([start.x], [start.y], color='g', label='Start')
        plt.scatter([goal.x], [goal.y], color='r', label='Goal')
        plt.grid(True)
        plt.legend()
        plt.show()


if __name__ == "__main__":
    main()
