import json

#opening the JSON file containing possible goals and actions
with open("goalsAndActions.json", "r") as file:
    goalsAndActionsJson = json.load(file)

current_top_goal = {}
current_top_action = {}


def choose_goal(_goal_json):
    """
    Function choosing the most important goal from the goal file. Sets the global current_top_goal
    :param _goal_json: Part of the goalsAndActions JSON file which contains goals
    """
    global current_top_goal
    # Find the most important goal
    top_goal = _goal_json[0]

    for goal in _goal_json:
        if goal["value"] > top_goal["value"]:
            top_goal = goal

    current_top_goal = top_goal
    print("Current top goal set as:", current_top_goal)


def get_goal_change(_goal, _action):
    """
    Get and integer by how much this action can change this goal
    :param _goal: One goal form the Goals Json
    :param _action: One action from the Action Json
    :return: Returns and integer by how much a specified action can change a specified goal, returns 0 if action cannot change this goal
    """
    if _goal["name"] == _action["goalsChange"][0]["name"]:
        return _action["goalsChange"][0]["value"]
    else:
        return 0


def choose_action(_action_json):
    global current_top_action
    top_action = _action_json[0]
    best_utility = get_goal_change(current_top_goal, _action_json[0])

    for action in _action_json:
        if get_goal_change(current_top_goal, action) > best_utility:
            best_utility = get_goal_change(current_top_goal, action)
            top_action = action

    current_top_action = top_action
    print("Current top action set as:", current_top_action)


def update_goals(_current_top_goal, _current_top_action):
    """
    dfhfhgfgh
    :param _current_top_goal: fgfgjh
    :param _current_top_action: fgjg
    :return:
    """
    global goalsAndActionsJson
    goal_index = 0
    _goal_name = _current_top_goal["name"]

    for i, goal in enumerate(goalsAndActionsJson["goals"]):
        if goal["name"] == _current_top_goal["name"]:
            goal_index = i
            break
    goalsAndActionsJson["goals"][goal_index]["value"] -= _current_top_action["goalsChange"][0]["value"]


if __name__ == '__main__':
    for i in range(10):
        print("\nRound", i)
        choose_goal(goalsAndActionsJson["goals"])
        choose_action(goalsAndActionsJson["actions"])
        update_goals(current_top_goal, current_top_action)