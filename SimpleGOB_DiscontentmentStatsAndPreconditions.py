import json
import operator

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
    discontentment = 0

    for goal in _all_goals:
        new_goal_value = goal["value"] + get_goal_change(goal, _action)
        discontentment += pow(new_goal_value, 2)

    print("Action",_action["name"],"discontentment", discontentment)
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

            print(f"""Checking stat #{idx} - {precondition["what"]}, current value: \t{current_value}""")
            print(f"""Logical test: {precondition["what"]} {logical_test_str} {precondition_value} - Result: {logical_test_result}""")

    if preconditions_met_bool:
        print("Success - conditions met")
    else:
        print("Fail - not all conditions met")
def choose_action(_all_actions, _all_goals):
    """
    Loops through all actions and chooses the action which lowers discontentment the most.
    :param _all_actions: All actions JSON
    :param _all_goals: All goals JSON
    :return: Returns the best action JSON
    """
    global current_top_action
    best_action = _all_actions[0]
    best_value = calculate_discontentment(best_action, _all_goals)

    for action in _all_actions[1:]:
        this_value = calculate_discontentment(action, _all_goals)
        if this_value < best_value:
            best_value = this_value
            best_action = action

    current_top_action = best_action
    return best_action


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
    for i in range(30):
        print("\nRound", i)
        print("Chosen action:\t", choose_action(goalsAndActionsJson["actions"], goalsAndActionsJson["goals"]))
        print("Goals before:\t", goalsAndActionsJson["goals"])
        update_goals(current_top_action)
        print("Goals are now:\t", goalsAndActionsJson["goals"])
        preconditions_met(goalsAndActionsJson["stats"], goalsAndActionsJson["actions"][0])


if __name__ == '__main__':
    main()

