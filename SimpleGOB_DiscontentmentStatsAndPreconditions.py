import json
import operator
import sys

current_top_action = {}

with open("goalsAndActions_expanded.json", "r") as file:
    goalsAndActionsJson = json.load(file)

ops = {
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
}


def get_goal_change(_goal, _action):
    """
    Get the effect of an action on a chosen goal
    :param _goal: A single goal JSON
    :param _action: A single action JSON
    :return: Returns the effects of an action on the goal specified in the args
    """
    action_effect = 0
    for affectedGoal in _action["goalsChange"]:

        if _goal["name"] == affectedGoal["name"]:
            action_effect -= affectedGoal["value"]

    # print("Action",_action["name"], "will change",_goal["name"],"by",action_effect )
    return action_effect


def calculate_discontentment(_action,_all_goals):
    """
    Loops through all the goals and calculates their values after a chosen action. Calculates the second power of the
    new goal value and adds it to the overall discontentment. Return this calculated discontentment.
    :param _action: A single action JSON
    :param _all_goals: All goals JSON
    :return: Integer discontentment value
    """
    discontentment = 0.0

    for goal in _all_goals:
        new_goal_value = goal["value"] + get_goal_change(goal, _action)
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
        return best_action
    else:
        print("No action can be chosen. Closing the game.")
        sys.exit()


def update_stats(_json_stats, _json_action):
    global goalsAndActionsJson

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

    for idx, goal in enumerate(goalsAndActionsJson["goals"]):
        goal_name = goal["name"]
        new_goal_value = goal["value"] + get_goal_change(goal, _current_top_action)
        if new_goal_value>0:
            goalsAndActionsJson["goals"][idx]["value"] = new_goal_value
        else:
            goalsAndActionsJson["goals"][idx]["value"] = 0


def main():
    print("Starting goals are", goalsAndActionsJson["goals"])
    all_chosen_actions = []
    for i in range(50):
        print("\nRound", i)
        print("Stats:", goalsAndActionsJson["stats"])
        chosen_action = choose_action(goalsAndActionsJson["actions"], goalsAndActionsJson["goals"])
        print("Chosen action:\t", chosen_action)
        all_chosen_actions.append(chosen_action["name"])
        print("Goals before:\t", goalsAndActionsJson["goals"])
        update_goals(current_top_action)
        update_stats(goalsAndActionsJson["stats"], current_top_action)
        print("Goals are now:\t", goalsAndActionsJson["goals"])

    print(all_chosen_actions)


if __name__ == '__main__':
    main()

