import json
import operator
import sys
import matplotlib.pyplot as plt
import numpy as np
import math


current_top_action = {}

with open("setup_with_percentages.json", "r") as file:
    goalsAndActionsJson = json.load(file)

ops = {
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
}


def get_goal_change(_goal_name, _goal_value, _action):
    """
    Get the effect of an action on a chosen goal
    :param _goal_value: A single goal value
    :param _goal_name: A single goal name
    :param _action: A single action JSON
    :return: Returns the effects of an action on the goal specified in the args
    """
    action_effect = 0
    for affectedGoal in _action["goalsChange"]:
        if _goal_name == affectedGoal["name"]:
            if isinstance(affectedGoal["value"], str) and '%' in affectedGoal["value"]:
                percentage_value = float(affectedGoal["value"].strip('%')) / 100
                action_effect -= math.ceil(_goal_value * percentage_value)
            else:
                action_effect -= affectedGoal["value"]

    # print("Action",_action["name"], "will change",_goal_name["name"],"by",action_effect )
    return action_effect


def calculate_discontentment(_action, _all_goals):
    """
    Loops through all the goals and calculates their values after a chosen action. Calculates the second power of the
    new goal value and adds it to the overall discontentment. Return this calculated discontentment.
    :param _action: A single action JSON
    :param _all_goals: All goals JSON
    :return: Integer discontentment value
    """
    discontentment = 0.0

    for goal_name, goal_value in _all_goals.items():
        new_goal_value = goal_value + get_goal_change(goal_name, goal_value, _action)
        if new_goal_value >= 100:
            new_goal_value = 100
        elif new_goal_value < 0:
            new_goal_value = 0
        discontentment += pow(new_goal_value, 2)

    return discontentment


def preconditions_met(_json_stats, _json_action):
    preconditions_met_bool = True

    for idx, precondition in enumerate(_json_action["preconditions"]):

        if precondition["where"] == "stats":
            current_value = _json_stats[precondition["what"]]
            precondition_value = precondition["value"]
            logical_test_str = precondition["logical_test"]

            logical_test_result = ops[precondition["logical_test"]](current_value, precondition_value)

            if not logical_test_result:
                preconditions_met_bool = False

            # print(f"""Checking stat #{idx} - {precondition["what"]}, current value: \t{current_value}""")
            # print(f"""Logical test: {precondition["what"]} {logical_test_str} {precondition_value} - Result: {logical_test_result}""")

    # if preconditions_met_bool:
    #     print("Success - conditions met")
    # else:
    #     print("Fail - not all conditions met")

    return preconditions_met_bool


def choose_action(_all_actions, _all_goals):
    """
    Loops through all actions and chooses the action which lowers discontentment the most
    :param _all_actions: All actions JSON
    :param _all_goals: All goals JSON
    :return: Returns the best action JSON
    """
    global current_top_action
    best_action = None
    best_value = float('inf')
    # best_action = _all_actions[0]
    # best_value = calculate_discontentment(best_action, _all_goals)

    for action in _all_actions[0:]:
        this_value = calculate_discontentment(action, _all_goals)
        are_preconditions_met = preconditions_met(goalsAndActionsJson["stats"], action)

        if are_preconditions_met:
            print("\tAction", action["name"], "discontentment", this_value)
        else:
            print("Action", action["name"], "discontentment", this_value)

        if this_value < best_value and are_preconditions_met:
            best_value = this_value
            best_action = action
            # print("Best action updated")

    current_top_action = best_action

    if best_action:
        return best_action, best_value
    else:
        print("No action can be chosen. Closing the game.")
        sys.exit()


def update_stats(_json_stats, _json_action, _chosen_action_discontentment):
    global goalsAndActionsJson

    goalsAndActionsJson["stats"]["discontentment"] = _chosen_action_discontentment
    stats_change = _json_action["statsChange"]
    if stats_change:
        for stat_change in stats_change:
            for stat_key, stat in _json_stats.items():
                if stat_change["name"] == stat_key:
                    goalsAndActionsJson["stats"][stat_key] = _json_stats[stat_key] + stat_change["value"]


def update_goals(_current_top_action):
    """
    Updates the JSON (does not update the JSON file) after using the best action
    :param _current_top_action: The global current top action
    :return:
    """
    global goalsAndActionsJson

    for goal_name, goal_value in goalsAndActionsJson["goals"].items():
        new_goal_value = goal_value + get_goal_change(goal_name, goal_value, _current_top_action)
        if 0 < new_goal_value < 100:
            goalsAndActionsJson["goals"][goal_name] = new_goal_value
        elif new_goal_value >= 100:
            goalsAndActionsJson["goals"][goal_name] = 100
        else:
            goalsAndActionsJson["goals"][goal_name] = 0


def recurring_changes_update():
    global goalsAndActionsJson

    for recurring_goal_name, recurring_goal_value in goalsAndActionsJson["recurring_changes"]["changed_goals"].items():
        for goal_name, goal_value in goalsAndActionsJson["goals"].items():
            if recurring_goal_name == goal_name:
                goal_value += recurring_goal_value
                goalsAndActionsJson["goals"][goal_name] = goal_value


def main(_iterations):
    print("Starting goals are", goalsAndActionsJson["goals"])
    all_chosen_actions = []
    stats_list = []
    goals_list = []
    goal_values = []
    for i in range(_iterations):
        print("\nRound", i)
        print("Stats:", goalsAndActionsJson["stats"])
        stats_list.append(goalsAndActionsJson["stats"].copy())
        goals_list.append(goalsAndActionsJson["goals"].copy())
        chosen_action, chosen_action_discontentment = choose_action(goalsAndActionsJson["actions"], goalsAndActionsJson["goals"])
        print("Chosen action:\t", chosen_action)
        all_chosen_actions.append(chosen_action["name"])
        print("Goals before:\t", goalsAndActionsJson["goals"])
        recurring_changes_update()
        update_goals(current_top_action)
        goal_values.append(list(goalsAndActionsJson["goals"].values()))
        update_stats(goalsAndActionsJson["stats"], current_top_action, chosen_action_discontentment)
        print("Goals are now:\t", goalsAndActionsJson["goals"])

    print("all_chosen_actions", all_chosen_actions)
    print("stats_list", stats_list)
    goal_values = np.transpose(goal_values)  # transposing extracted goal values so that every goal is in its own array

    # Plot the stats values
    plt.figure()
    stat_keys = list(goalsAndActionsJson["stats"].keys())
    for key in stat_keys[0:]:
        stat_values = [d[key] for d in stats_list]
        plt.plot(stat_values, label=key)

    plt.title('Stats over time')
    plt.xlabel('Iteration')
    plt.ylabel('Value')
    plt.legend()
    plt.savefig('stats_changes.jpg')

    # Plot the goal values
    plt.figure()
    goal_keys = list(goalsAndActionsJson["goals"].keys())
    for key, values in zip(goal_keys, goal_values):
        plt.plot(values, label=key)

    plt.title('Goals over time')
    plt.xlabel('Iteration')
    plt.ylabel('Value')
    plt.legend()
    plt.savefig('goals_changes.jpg')

    print("All chosen acitons:\n", all_chosen_actions)
    print("All stats:\n", stats_list)
    print("All goals:\n", goal_values)

if __name__ == '__main__':
    main()
