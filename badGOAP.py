import json
import sys
import matplotlib.pyplot as plt
import numpy as np
import math
import copy


class WorldModel:
    def __init__(self, _setup_file):
        self.setup = _setup_file
        self.action_index = 0

    def calculate_discontentment(self):
        discontentment = 0.0
        for goal_name, goal_value in self.setup["goals"].items():
            discontentment += pow(goal_value, 2)
        return discontentment

    def next_action(self):
        if self.action_index < len(self.setup["actions"]):
            action = self.setup["actions"][self.action_index]
            self.action_index += 1
            return action
        else:
            return None

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

    models[0] = world_model
    current_depth = 0

    best_action = None
    best_value = float('inf')

    while current_depth >= 0:
        current_value = models[current_depth].calculate_discontentment()

        if current_depth >= max_depth:
            if current_value < best_value:
                best_value = current_value
                best_action = actions[0]

            current_depth -= 1
            continue

        next_action = models[current_depth].next_action()
        if next_action:
            models[current_depth + 1] = copy.deepcopy(models[current_depth])
            actions[current_depth] = next_action
            models[current_depth + 1].apply_action(next_action)
            current_depth += 1
        else:
            current_depth -= 1

    return best_action, best_value


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

    print("Starting goals are", goals_and_actions_json["goals"])

    all_chosen_actions = []
    stats_list = []
    goals_list = []
    goal_values = []

    for i in range(iterations):
        print("\nRound", i)
        print("Goals:", goals_and_actions_json["goals"])
        goals_list.append(goals_and_actions_json["goals"].copy())
        stats_list.append(goals_and_actions_json["stats"].copy())
        world_model = WorldModel(goals_and_actions_json)
        chosen_action, chosen_action_discontentment = plan_action(world_model, 1)
        if chosen_action:
            print("Chosen action:\t", chosen_action)
            all_chosen_actions.append(chosen_action["name"])
            world_model.apply_action(chosen_action, chosen_action_discontentment)
            goal_values.append(list(goals_and_actions_json["goals"].values()))
            goals_and_actions_json["goals"] = world_model.setup["goals"]
            goals_and_actions_json["stats"] = world_model.setup["stats"]
            print("Goals are now:\t", goals_and_actions_json["goals"])
        else:
            print("No action can be chosen. Closing the game.")
            sys.exit()

    plot_goals(goals_list)
    plot_stats(stats_list)

