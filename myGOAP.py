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


def plan_action(world_model, max_depth):
    models = [None] * (max_depth + 1)
    actions = [None] * max_depth
    action_indices = [0] * (max_depth + 1)

    models[0] = world_model
    current_depth = 0

    best_action = None
    best_value = float('inf')

    while current_depth >= 0:
        print("\t"*current_depth,"Curr depth", current_depth)
        current_value = models[current_depth].calculate_current_discontentment()
        print("\t" * current_depth, "Curr value", current_value)

        if current_depth >= max_depth:
            if current_value < best_value:
                best_value = current_value
                best_action = actions[0]
            current_depth -= 1
            continue

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


        if not all_actions_were_checked:
            #print("Trying action", next_action["name"]," idx ",action_indices[current_depth]-1)
            models[current_depth + 1] = copy.deepcopy(models[current_depth])
            actions[current_depth] = next_action
            models[current_depth + 1].apply_action(next_action)
            current_depth += 1
        else:
            action_indices[current_depth] = 0  # Reset action index for this depth
            current_depth -= 1

    # while current_depth >= 0:
    #     current_value = models[current_depth].calculate_current_discontentment()
    #
    #     if current_depth >= max_depth or action_indices[current_depth] == len(models[current_depth].setup["actions"]):
    #         if current_value < best_value:
    #             best_value = current_value
    #             best_action = actions[0]
    #         current_depth -= 1
    #         continue
    #
    #     # Find the next valid action
    #     while action_indices[current_depth] < len(models[current_depth].setup["actions"]):
    #         next_action = models[current_depth].setup["actions"][action_indices[current_depth]]
    #         action_indices[current_depth] += 1
    #         if models[current_depth].preconditions_met(next_action):
    #             break
    #     else:
    #         # All actions have been checked, so go back to the outer loop
    #         action_indices[current_depth] = 0  # Reset action index for this depth
    #         current_depth -= 1
    #         continue
    #
    #     # Apply the valid action found and increase the depth
    #     models[current_depth + 1] = copy.deepcopy(models[current_depth])
    #     actions[current_depth] = next_action
    #     models[current_depth + 1].apply_action(next_action)
    #     current_depth += 1

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
    plt.savefig('GOAP-goals_changes.jpg')


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
    plt.savefig('GOAP-stats_changes.jpg')


def main(iterations):
    with open("setup_with_percentages.json", "r") as file:
        goals_and_actions_json = json.load(file)

    #print("Starting goals are", goals_and_actions_json["goals"])

    all_chosen_actions = []
    stats_list = []
    goals_list = []
    goal_values = []

    for i in range(iterations):
        #print("\nRound", i)
        #print("Goals:", goals_and_actions_json["goals"])
        print(f"Loading... {(100/iterations)*i}%")
        goals_list.append(goals_and_actions_json["goals"].copy())
        stats_list.append(goals_and_actions_json["stats"].copy())
        world_model = WorldModel(goals_and_actions_json)
        chosen_action, chosen_action_discontentment = plan_action(world_model, 1)
        if chosen_action:
            #print("Chosen action:\t", chosen_action)
            all_chosen_actions.append(chosen_action["name"])
            recurring_changes_update(goals_and_actions_json)
            world_model.apply_action(chosen_action, chosen_action_discontentment)
            goal_values.append(list(goals_and_actions_json["goals"].values()))
            goals_and_actions_json["goals"] = world_model.setup["goals"]
            goals_and_actions_json["stats"] = world_model.setup["stats"]
            #print("Goals are now:\t", goals_and_actions_json["goals"])
        else:
            #print("No action can be chosen. Closing the game.")
            sys.exit()

    plot_goals(goals_list)
    plot_stats(stats_list)

    print("All chosen acitons:\n",all_chosen_actions)
    print("All stats:\n", stats_list)
    print("All goals:\n", goal_values)
