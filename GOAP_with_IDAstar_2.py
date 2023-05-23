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
                #print(action["name"], "precons not met")
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


def plan_action(world_model, max_depth):
    """
    This function is used to choose the best action to take, given the current state of the world. It uses a form of
    Depth-First Search (DFS) to simulate applying each possible action up to a certain depth (max_depth),
    and it chooses the action sequence that would result in the smallest total discontentment.
    """

    """
    These lines initialize arrays to store the world models, actions, and action indices for each level of the DFS.
    The world models represent the state of the world after applying each action sequence, and the actions are the
    actual action sequences. The action indices are used to keep track of which action to try next at each level.
    """
    models = [None] * (max_depth + 1)
    actions = [None] * max_depth
    action_indices = [0] * (max_depth + 1)


    """
    The initial world model and depth are set. The depth represents the number of actions in the current sequence.
    """
    models[0] = world_model
    current_depth = 0

    """
    The best action and its resulting discontentment are initialized.
    best_value is initially set to infinity so that any actual discontentment will be smaller.
    """
    best_action = None
    best_value = float('inf')

    """
    This is the main loop of the DFS. It continues as long as the current depth is not negative, 
    which means that there are still action sequences to try.
    """
    while current_depth >= 0:
        print("\t"*current_depth,"Curr depth", current_depth)

        """
        The total discontentment of the current world model is calculated.
        """
        current_value = models[current_depth].calculate_current_discontentment()
        print("\t" * current_depth, "Curr value", current_value)

        """
        If the maximum depth has been reached, it means that an action sequence of maximum length has been applied.
        In this case, it checks if the total discontentment is less than the best found so far, and if it is,
        it updates the best action and value. It then goes back one level in the DFS.
        """
        if current_depth >= max_depth:
            if current_value < best_value:
                best_value = current_value
                best_action = actions[0]
            current_depth -= 1
            continue

        """
        This loop continues until it finds an action whose preconditions are met, or until it has checked all possible
        actions at the current level. If an action's preconditions are met, it breaks the loop to apply the action.
        """
        all_actions_were_checked = False
        while not all_actions_were_checked:
            if action_indices[current_depth] < len(models[current_depth].setup["actions"]):
                next_action = models[current_depth].setup["actions"][action_indices[current_depth]]
                print("\t"*current_depth,"Evaluating action:", next_action['name'])
                action_indices[current_depth] += 1
                if models[current_depth].preconditions_met(next_action):
                    all_actions_were_checked = False
                    break
                else:
                    print("\t"*current_depth,next_action["name"], "precons not met")
                    pass
            else:
                all_actions_were_checked = True

        """
        If it found an action to apply, it creates a deep copy of the current world model,
        applies the action to the copy, and goes one level deeper in the DFS.
        """
        if not all_actions_were_checked:
            #print("Trying action", next_action["name"]," idx ",action_indices[current_depth]-1)
            models[current_depth + 1] = copy.deepcopy(models[current_depth])
            actions[current_depth] = next_action
            models[current_depth + 1].apply_action(next_action)
            current_depth += 1

        else:
            """
            If all actions have been checked and none of them could be applied, it goes back one level in the DFS.
            """
            action_indices[current_depth] = 0  # Reset action index for this depth
            current_depth -= 1

    """
    After exploring all possible action sequences up to the maximum depth, it returns the first action of
    the best sequence and the total discontentment resulting from the best sequence.
    """
    #print("Best action:",best_action["name"], "best value:",best_value)
    print("best_action",best_action['name'],"best_value",best_value)
    return best_action, best_value


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
    plt.savefig('GOAP-goals_changes.jpg', dpi=300)


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
    plt.savefig('GOAP-stats_changes.jpg', dpi=300)


def main(iterations):
    """

    :param iterations:
    :return:
    """

    """
    This opens a JSON file which contains the initial setup of the world, including the initial state of goals
    and actions. The file is read and its contents are parsed into a Python dictionary.
    """
    with open("setup_with_percentages.json", "r") as file:
        goals_and_actions_json = json.load(file)

    #print("Starting goals are", goals_and_actions_json["goals"])

    """
    These are lists to store the history of chosen actions, stats, and goals over all iterations.
    This is done so that the progression of the world model can be tracked and later analyzed or visualized.
    """
    all_chosen_actions = []
    stats_list = []
    goals_list = []
    goal_values = []

    """
    This is the main loop of the function, which repeats the process of action planning and execution
    for a given number of iterations.
    """
    for i in range(iterations):
        #print("\nRound", i)
        #print("Goals:", goals_and_actions_json["goals"])
        print(f"Loading... {(100/iterations)*i}%")

        """
        These lines append the current state of the goals and stats to the respective lists.
        They are copied so that the lists will contain the state at each iteration,
        not just references to the final state.
        """
        goals_list.append(goals_and_actions_json["goals"].copy())
        stats_list.append(goals_and_actions_json["stats"].copy())

        """
        This creates a new instance of the WorldModel class, initializing it with the current state of the world.
        """
        world_model = WorldModel(goals_and_actions_json)

        """
        This calls the plan_action function to select the best action to perform based on the current world model.
        The chosen action and its resulting discontentment are returned.
        """
        chosen_action, chosen_action_discontentment = plan_action(world_model, 1)

        """
        This checks if an action was chosen. If no action could be
        chosen (which would mean that no action's preconditions were met), the program terminates.
        """
        if chosen_action:
            #print("Chosen action:\t", chosen_action)
            all_chosen_actions.append(chosen_action["name"])

            """
            These lines update the state of the world according to recurring changes and the chosen action.
            The recurring_changes_update function applies changes that happen every iteration,
            and apply_action applies the changes resulting from the chosen action.
            """
            recurring_changes_update(goals_and_actions_json)
            world_model.apply_action(chosen_action, chosen_action_discontentment)

            """
            These lines update the state of the world for the next iteration, based on the new state of the world model.
            """
            goal_values.append(list(goals_and_actions_json["goals"].values()))
            goals_and_actions_json["goals"] = world_model.setup["goals"]
            goals_and_actions_json["stats"] = world_model.setup["stats"]
            #print("Goals are now:\t", goals_and_actions_json["goals"])
        else:
            print("No action can be chosen. Closing the game.")
            sys.exit()

    """
    After all iterations have been completed, these lines create and save plots of the goal values and stats over time,
    using the lists that were compiled during the iterations.
    """
    plot_goals(goals_list)
    plot_stats(stats_list)

    print("All chosen acitons:\n",all_chosen_actions)
    print("All stats:\n", stats_list)
    # print("All goals:\n", goal_values)
