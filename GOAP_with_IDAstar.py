import hashlib
import json
import sys
import matplotlib.pyplot as plt
import numpy as np
import math
import copy
import operator

class WorldModel:

    ops = {
        ">": operator.gt,
        ">=": operator.ge,
        "<": operator.lt,
        "<=": operator.le,
        "==": operator.eq,
        "!=": operator.ne,
    }

    def __init__(self, _setup_file):
        self.setup = _setup_file
        self.action_index = 0

    def get_goals(self):
        return self.setup['goals']

    def get_stats(self):
        return self.setup['stats']

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

    def calculate_current_discontentment(self):
        discontentment = 0.0
        for goal_name, goal_value in self.setup["goals"].items():
            if goal_value >= 100:
                goal_value = 100
            elif goal_value < 0:
                goal_value = 0
            discontentment += pow(goal_value, 2)
        return discontentment

    def calculate_discontentment(self, _action):
        """Changed"""
        discontentment = 0.0

        for goal_name, goal_value in self.setup["goals"].items():
            new_goal_value = goal_value + self.get_goal_change(goal_name, goal_value, _action)
            if new_goal_value >= 100:
                new_goal_value = 100
            elif new_goal_value < 0:
                new_goal_value = 0
            discontentment += pow(new_goal_value, 2)

        return discontentment

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

    def heuristic(self, _action):
        """
        Estimate of the cost to reach the goal from the given node.
        The lower the value, the better.
        """
        return self.calculate_discontentment(_action)

    def hash_model(self):
        """
        Generate a hash representation of the current state of the world model.
        """
        # We'll use the goals and stats as the basis for our hash
        state = {
            "goals": self.setup["goals"],
            "stats": self.setup["stats"],
        }

        # We'll use the json library to convert our state to a string
        # and the hashlib library to generate a hash of this string
        state_str = json.dumps(state, sort_keys=True)
        state_hash = hashlib.sha256(state_str.encode()).hexdigest()

        return state_hash


class TranspositionTable:

    def __init__(self):
        self.entries = {}
        self.size = 0

    def has(self, world_model):
        hash_value = hash(world_model)
        return hash_value in self.entries

    def add(self, world_model, depth):
        hash_value = hash(world_model)
        if hash_value in self.entries:
            if depth < self.entries[hash_value]:
                self.entries[hash_value] = depth
        else:
            self.entries[hash_value] = depth


# def plan_action(world_model, goal, heuristic, max_depth):
#     cutoff = heuristic.estimate(world_model)
#     transposition_table = TranspositionTable()
#
#     while cutoff >= 0:
#         cutoff, action = do_depth_first(world_model, goal, transposition_table, heuristic, max_depth, cutoff)
#
#         if action:
#             return action
#
#     return None

def plan_action(world_model, max_depth):
    best_action = None
    min_discontentment = float('inf')
    actions = world_model.setup["actions"]
    for action in actions:
        if world_model.preconditions_met(action):
            temp_world_model = copy.deepcopy(world_model)
            temp_world_model.apply_action(action)
            temp_discontentment = temp_world_model.calculate_discontentment(action)
            if temp_discontentment < min_discontentment:
                min_discontentment = temp_discontentment
                best_action = action
    return best_action, min_discontentment  # Return both the action and the discontentment


def do_depth_first(world_model, current_depth, cutoff, models, actions, action_indices):
    g = world_model.calculate_current_discontentment()  # Cost to reach current node

    # Estimate the cost to reach the goal using the next action's heuristic
    _, next_action, _ = world_model.next_action()
    if next_action is None:  # If no more actions, set heuristic to 0
        h = 0
    else:
        h = world_model.heuristic(next_action)  # Heuristic cost to reach goal from current node

    f = g + h  # Total estimated cost

    if f > cutoff:
        return f
    if h == 0:  # If heuristic is 0, we have reached the goal
        return 'FOUND'

    min = float('inf')

    while True:
        all_actions_were_checked, next_action, _ = models[current_depth].next_action()
        if all_actions_were_checked:
            return min

        if next_action is not None:
            models[current_depth + 1] = copy.deepcopy(models[current_depth])
            actions[current_depth] = next_action
            models[current_depth + 1].apply_action(next_action)
            t = do_depth_first(models[current_depth + 1], current_depth + 1, cutoff, models, actions, action_indices)

            if t == 'FOUND':
                return 'FOUND'
            if t < min:
                min = t


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
    plt.savefig('IDAstar-goals_changes.jpg', dpi=300)


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
    plt.savefig('IDAstar-stats_changes.jpg', dpi=300)

def main(iterations):
    with open("setup_with_percentages.json", "r") as file:
        setup = json.load(file)

    all_chosen_actions = []
    stats_list = []
    goals_list = []
    goal_values = []

    for i in range(iterations):
        print(f"Loading... {(100/iterations)*i}%")
        # Create a new instance of the world model for each iteration
        world_model = WorldModel(setup)
        # Record the initial state of the world
        goals_list.append(setup["goals"].copy())
        stats_list.append(setup["stats"].copy())
        # Plan and apply the chosen action
        chosen_action, action_discontentment = plan_action(world_model, 3)
        if chosen_action:
            all_chosen_actions.append(chosen_action["name"])
            recurring_changes_update(setup)
            world_model.apply_action(chosen_action, action_discontentment)
            goal_values.append(list(world_model.get_goals().values()))
            # Update the setup for the next iteration
            setup["goals"] = world_model.setup["goals"]
            setup["stats"] = world_model.setup["stats"]
        else:
            print("No action can be chosen. Closing the game.")
            sys.exit()

    plot_goals(goals_list)
    plot_stats(stats_list)

    print("All chosen acitons:\n", all_chosen_actions)
    print("All stats:\n", stats_list)
    # print("All goals:\n", goal_values)
