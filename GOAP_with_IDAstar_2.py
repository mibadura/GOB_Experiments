import json
import sys
import matplotlib.pyplot as plt
import numpy as np
import math
import copy
import operator

"""
This is the declaration of the WorldModel class.
It represents the current state of the world, which includes both stats and goals.
"""


class WorldModel:

    """
    This dictionary maps strings of comparison operators to the actual Python operator functions.
    This makes it easy to evaluate the preconditions of an action,
    which are specified as strings in the input JSON file.
    """
    ops = {
        ">": operator.gt,
        ">=": operator.ge,
        "<": operator.lt,
        "<=": operator.le,
        "==": operator.eq,
        "!=": operator.ne,
    }

    """
    The __init__ method is the constructor of the WorldModel class.
    It takes a setup file as input, which is assumed to be a dictionary that contains the initial state of the world
    and the possible actions. The action_index variable is initialized to 0;
    this is used to iterate over the actions in the next_action method.
    """
    def __init__(self, _setup_file):
        self.setup = _setup_file
        self.action_index = 0

    """
    This method calculates how an action would change a specific goal. It iterates over the goals affected by the
    action and checks if the goal matches the given goal name.
    If it does, it calculates the change in the goal's value according to the effect specified by the action.
    This can either be an absolute value or a percentage of the current goal value.
    """
    def get_goal_change(self, _goal_name, _goal_value, _action):
        """Changed"""
        action_effect = 0
        for affectedGoal in _action["goalsChange"]:
            if _goal_name == affectedGoal["name"]:
                if isinstance(affectedGoal["value"], str) and '%' in affectedGoal["value"]:
                    percentage_value = float(affectedGoal["value"].strip('%')) / 100
                    action_effect -= math.ceil(_goal_value * percentage_value)
                else:
                    action_effect -= affectedGoal["value"]

        return action_effect

    """
    This method calculates the total discontentment in the current world state. It squares the value of each goal
    and adds them all up. The higher the discontentment, the further the world is from its ideal state.
    """
    def calculate_current_discontentment(self):
        discontentment = 0.0
        for goal_name, goal_value in self.setup["goals"].items():
            if goal_value >= 100:
                goal_value = 100
            elif goal_value < 0:
                goal_value = 0
            discontentment += pow(goal_value, 2)
        return discontentment

    """
    This method calculates what the total discontentment would be if a specific action was applied. It works
    similarly to calculate_current_discontentment,
    but it first adjusts the goal values according to the effect of the given action.
    """
    def calculate_discontentment(self, _action):
        discontentment = 0.0

        for goal_name, goal_value in self.setup["goals"].items():
            new_goal_value = goal_value + self.get_goal_change(goal_name, goal_value, _action)
            if new_goal_value >= 100:
                new_goal_value = 100
            elif new_goal_value < 0:
                new_goal_value = 0
            discontentment += pow(new_goal_value, 2)

        return discontentment

    """
    This method checks if the preconditions of an action are met in the current world state. It iterates over
    the preconditions of the action, evaluates them using the current stats,
    and returns False if any precondition is not met.
    """
    def preconditions_met(self, _action):
        preconditions_met_bool = True

        for idx, precondition in enumerate(_action["preconditions"]):

            if precondition["where"] == "stats":
                current_value = self.setup["stats"][precondition["what"]]
                precondition_value = precondition["value"]

                logical_test_result = self.ops[precondition["logical_test"]](current_value, precondition_value)

                if not logical_test_result:
                    preconditions_met_bool = False

        return preconditions_met_bool

    """
    This method retrieves the next action from the list of possible actions, checking if its preconditions are met.
    If they are, it returns the action; otherwise, it continues to the next action.
    If all actions have been checked, it indicates this by returning True for all_actions_checked.
    """
    def next_action(self):
        if self.action_index < len(self.setup["actions"]):
            all_actions_checked = False
            action = self.setup["actions"][self.action_index]
            self.action_index += 1
            if self.preconditions_met(action):
                return all_actions_checked, action, self.action_index
            else:
                print(action["name"], "precons not met")
                return all_actions_checked, None, None
        else:
            all_actions_checked = True
            return all_actions_checked, None, None

    """
    This method applies a given action to the world. It adjusts the goal values according to the effect of the action,
    and it also adjusts the stats if specified by the action.
    If the total discontentment after applying the action is given, it also updates this in the stats.
    The effect of the action is not applied directly to the goal values but rather subtracted from them because lower
    goal values mean less discontentment.
    """
    def apply_action(self, action, action_discontentment=None):
        print("Applying action",action["name"])
        for goal_change in action["goalsChange"]:
            goal_name = goal_change["name"]
            if goal_name in self.setup["goals"]:
                if isinstance(goal_change["value"], str) and '%' in goal_change["value"]:
                    percentage_value = float(goal_change["value"].strip('%')) / 100
                    self.setup["goals"][goal_name] -= math.ceil(self.setup["goals"][goal_name] * percentage_value)
                else:
                    self.setup["goals"][goal_name] -= goal_change["value"]
                if self.setup["goals"][goal_name] > 100:
                    self.setup["goals"][goal_name] = 100
                elif self.setup["goals"][goal_name] < 0:
                    self.setup["goals"][goal_name] = 0

        if "statsChange" in action:
            for stat_change in action["statsChange"]:
                stat_name = stat_change["name"]
                if stat_name in self.setup["stats"]:
                    self.setup["stats"][stat_name] += stat_change["value"]

        if action_discontentment:
            self.setup["stats"]["discontentment"] = action_discontentment

    def hash(self):
        """Generate a hash value representing the world model."""
        goal_hash = hash(frozenset(self.setup['goals'].items()))
        stat_hash = hash(frozenset(self.setup['stats'].items()))
        return hash((goal_hash, stat_hash))


class Heuristic:
    def __init__(self, weight=1.0):
        self.weight = weight

    def estimate(self, world_model):
        return self.weight * world_model.calculate_current_discontentment()
        # return 0


class TranspositionTable:
    """
    The TranspositionTable class serves as a cache mechanism to store world states that have already been evaluated.
    By storing these states, it avoids re-computation and duplication of efforts, thus speeding up the execution
    of the IDA* algorithm.

    Each world state is represented by a hash value in the transposition table, which is a one-dimensional array.
    The hash function is used to compute an index in this array where the state is stored, ensuring efficient retrieval.
    The size of the transposition table is defined at the time of its creation.
    """
    class Entry:
        """
        The Entry class represents a single item in the transposition table.

        Each entry consists of:
        - a 'hash_value', which is a unique identifier of a world state,
        - and a 'depth' value, which represents the depth in the search tree where this state was first encountered.
        """
        def __init__(self):
            self.hash_value = None
            self.depth = float('inf')

    def __init__(self, size):
        """
        Initialize a new TranspositionTable with a given size. All entries are initially empty.
        """
        self.entries = [self.Entry() for _ in range(size)]
        self.size = size

    def has(self, world_model):
        """
        Check if the TranspositionTable already has a specific world model.

        The method calculates the hash value of the world model, then checks the corresponding entry in the table.
        Returns True if this entry's hash value matches the world model's hash value, False otherwise.
        """
        # Check if the TranspositionTable already has this world model
        entry = self.entries[world_model.hash() % self.size]
        return entry.hash_value == world_model.hash()

    def add(self, world_model, depth):
        """
        Adds a new world model to the TranspositionTable or updates the depth of an existing one.

        First, it checks if the world model is already in the table. If it is, and its new depth is less than
        the one stored in the table, the depth in the table is updated.

        If the world model is not in the table, or its stored depth is larger than the new depth, the table entry
        is updated to hold the new world model's hash value and its corresponding depth.
        """
        # Add a new world model to the TranspositionTable or update the depth of an existing one
        entry = self.entries[world_model.hash() % self.size]

        if entry.hash_value == world_model.hash():
            if depth < entry.depth:
                entry.depth = depth
        elif depth < entry.depth:
            entry.hash_value = world_model.hash()
            entry.depth = depth


def do_depth_first(world_model, goal, transposition_table, heuristic, max_depth, cutoff):
    """
    This function performs a depth-limited depth-first search with iterative deepening and returns the smallest cutoff
    encountered and the corresponding best action.
    """
    # Initialize storage for world models at each depth, and actions and costs corresponding to them
    models = [None] * (max_depth + 2)
    actions = [None] * max_depth
    costs = [0.0] * (max_depth + 1)

    # Set up the initial data
    models[0] = world_model
    current_depth = 0

    # Keep track of the smallest pruned cutoff
    smallest_cutoff = float('inf')

    # Iterate until all actions at depth zero are completed
    while current_depth >= 0:
        print("-"*current_depth, "currrent depth",current_depth)
        # If the goal is fulfilled by the current world model, return the cutoff and the corresponding action
        if goal.is_fulfilled(models[current_depth]) and actions[0] is None:
            print("-"*current_depth, "Goal is fulfilled by the current model but no action chosen yet")
        elif goal.is_fulfilled(models[current_depth]) and actions[0]:
            print("-"*current_depth, "Goal is fulfilled by the current model and action is",actions[0]["name"])
            return cutoff, actions[0]

        # If we're at maximum depth, move back up the tree
        if current_depth >= max_depth:
            current_depth -= 1
            continue

        # Calculate total cost of plan including heuristic estimate
        cost = heuristic.estimate(models[current_depth]) + costs[current_depth]
        print("-"*current_depth, "cost:",cost)
        print("-"*current_depth, "cutoff:", cutoff)

        # If the cost exceeds the cutoff, move back up the tree and update smallest cutoff if necessary
        if cost > cutoff:
            if cost < smallest_cutoff:
                smallest_cutoff = cost
            current_depth -= 1
            print("-"*current_depth, "Decreasing depth")
            continue

        # Try the next action
        all_actions_checked = False
        next_action = None
        while not all_actions_checked and not next_action:
            print("-"*current_depth, "Looking for next action...")
            all_actions_checked, next_action, action_index = models[current_depth].next_action()
            print("-"*current_depth, "all_actions_checked, next_action, action_index",all_actions_checked, next_action, action_index)

        if next_action:
            # Copy the current model and apply the action to the copy
            models[current_depth + 1] = copy.deepcopy(models[current_depth])
            models[current_depth + 1].apply_action(next_action)

            # Update action and cost lists
            actions[current_depth] = next_action
            # costs[current_depth + 1] = costs[current_depth] + world_model.calculate_discontentment(next_action)#cost #assuming the cost here refers to discontentment calculated by the action
            costs[current_depth + 1] = current_depth + 1

            # Process the new state if it hasn't been seen before
            if not transposition_table.has(models[current_depth + 1]):
                current_depth += 1

            # #TEST
            # if models[current_depth + 1]:
            #     # Add the new model to the transposition table
            #     transposition_table.add(models[current_depth + 1], current_depth)

        else:
            # If there are no more actions to try, move back up the tree
            print("-"*current_depth, "Decreasing depth - no more actions to try")
            current_depth -= 1

    # If no action is found after searching all states, return the smallest cutoff encountered
    return smallest_cutoff, None


def plan_action(world_model, goal, heuristic, max_depth):
    """
    This function uses Iterative Deepening A* (IDA*) to plan actions.
    It returns the first best action found that meets the cutoff heuristic.
    """

    print("Inside plan_action() function...")  # Debug print

    # Initial cutoff is the heuristic from the start model
    cutoff = heuristic.estimate(world_model)

    # Create a TranspositionTable to avoid repeated states
    transposition_table = TranspositionTable(size=30)  # Choose an appropriate size here

    # Initialize chosen_action and chosen_action_discontentment
    chosen_action = None
    chosen_action_discontentment = None

    while cutoff < float('inf'):
        # print("Inside while loop in plan_action()...")  # Debug print
        # Conduct a depth-limited depth-first search and update cutoff and action
        print("Running do_depth_first")
        cutoff, action = do_depth_first(world_model, goal, transposition_table, heuristic, max_depth, cutoff)
        print("Chosen cutoff and action are", cutoff, action)
        # If an action has been found, return it
        if action:
            chosen_action = action
            chosen_action_discontentment = world_model.calculate_discontentment(action)
            break

    return chosen_action, chosen_action_discontentment


class Goal:
    """
    This class encapsulates the goal of minimizing discontentment in the world model.
    """

    def __init__(self, discontentment_threshold):
        """
        Initializes a new Goal instance.

        Parameters:
        discontentment_threshold (float): The threshold below which discontentment must fall for the goal to be considered fulfilled.
        """
        self.discontentment_threshold = discontentment_threshold

    def is_fulfilled(self, world_model):
        """
        Checks whether the goal is fulfilled in the given world model.

        Parameters:
        world_model (WorldModel): The world model to check.

        Returns:
        bool: True if the discontentment in the world model is below the threshold, False otherwise.
        """
        return world_model.calculate_current_discontentment() <= self.discontentment_threshold


def recurring_changes_update(goals_and_actions_json):

    for recurring_goal_name, recurring_goal_value in goals_and_actions_json["recurring_changes"]["changed_goals"].items():
        for goal_name, goal_value in goals_and_actions_json["goals"].items():
            if recurring_goal_name == goal_name:
                goal_value += recurring_goal_value
                goals_and_actions_json["goals"][goal_name] = goal_value


def plot_goals(goals_list):
    plt.figure()
    goal_keys = list(goals_list[0].keys())
    for key in goal_keys:
        goal_values = [d[key] for d in goals_list]
        plt.plot(goal_values, label=key)

    plt.title('Goals over time')
    plt.xlabel('Iteration')
    plt.ylabel('Value')
    plt.legend()
    plt.savefig('GOAP-IDAstar2-goals_changes.jpg', dpi=300)


def plot_stats(stats_list):
    plt.figure()
    stat_keys = list(stats_list[0].keys())
    for key in stat_keys:
        stat_values = [d[key] for d in stats_list]
        plt.plot(stat_values, label=key)

    plt.title('Stats over time')
    plt.xlabel('Iteration')
    plt.ylabel('Value')
    plt.legend()
    plt.savefig('GOAP-IDAstar2-stats_changes.jpg', dpi=300)


def main(iterations, filename="setup_with_percentages.json"):

    # Load setup file
    with open(filename, "r") as file:
        setup = json.load(file)

    # Create a world model from the setup
    world_model = WorldModel(setup)

    # Create a heuristic with a weight of 1.0
    heuristic = Heuristic(weight=1.0)

    # Maximum depth to search
    max_depth = 100

    # Lists to keep track of the state at each iteration
    all_chosen_actions = []
    stats_list = []
    goals_list = []

    # Run the iterations
    for i in range(iterations):
        print(f"Iteration {i+1} of {iterations}...")

        # Add the current state to the lists
        stats_list.append(copy.deepcopy(world_model.setup["stats"]))
        goals_list.append(copy.deepcopy(world_model.setup["goals"]))

        # Plan the next action using Iterative Deepening A* (IDA*)
        current_discontentment = world_model.calculate_current_discontentment()
        # Create a goal with a threshold discontentment value
        goal = Goal(discontentment_threshold=current_discontentment-1)

        chosen_action, chosen_action_discontentment = plan_action(world_model, goal, heuristic, max_depth)

        # If an action was chosen, apply it
        if chosen_action:
            print(f"Chosen action: {chosen_action['name']}")
            all_chosen_actions.append(chosen_action['name'])
            world_model.apply_action(chosen_action, chosen_action_discontentment)
        else:
            print("No action can be chosen. Stopping.")
            break

        # # Update the recurring changes
        # recurring_changes_update(world_model.setup)

    # Plot the changes in goals and stats over time
    plot_goals(goals_list)
    plot_stats(stats_list)

    print("All chosen actions:\n", all_chosen_actions)
    print("All stats:\n", stats_list)
