import json

current_top_action = {}

with open("goalsAndActions.json", "r") as file:
    goalsAndActionsJson = json.load(file)


def get_goal_change(_goal, _action):
    action_effect = 0
    for affectedGoal in _action["goalsChange"]:

        if _goal["name"] == affectedGoal["name"]:
            action_effect -= affectedGoal["value"]

    # print("Action",_action["name"], "will change",_goal["name"],"by",action_effect )
    return action_effect


def calculate_discontentment(_action,_all_goals):
    discontentment = 0

    for goal in _all_goals:
        new_goal_value = goal["value"] + get_goal_change(goal, _action)
        discontentment += pow(new_goal_value, 2)

    print("Action",_action["name"],"discontentment", discontentment)
    return discontentment


def choose_action(_all_actions, _all_goals):
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
    global goalsAndActionsJson

    for idx, goal in enumerate(goalsAndActionsJson["goals"]):
        goal_name = goal["name"]
        new_goal_value = goal["value"] + get_goal_change(goal, _current_top_action)
        if new_goal_value>0:
            goalsAndActionsJson["goals"][idx]["value"] = new_goal_value
        else:
            goalsAndActionsJson["goals"][idx]["value"] = 0


if __name__ == '__main__':

    print("Starting goals are", goalsAndActionsJson["goals"])
    for i in range(10):
        print("\nRound",i)
        print("Chosen action:\t", choose_action(goalsAndActionsJson["actions"], goalsAndActionsJson["goals"]))
        print("Goals before:\t", goalsAndActionsJson["goals"])
        update_goals(current_top_action)
        print("Goals are now:\t", goalsAndActionsJson["goals"])
