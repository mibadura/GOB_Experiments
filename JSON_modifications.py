import json

base_json_path = "goalsAndActions_expanded.json"
output_json_name = "output_json.json"
mockup_action = json.loads("""{"name":"Go-Crazy",
            "goalsChange":[
                {"name":"Be-Healthy","value":-100},
                {"name":"Level-Up","value":-100},
                {"name":"Do-Many-Quests","value":-100},
                {"name":"Gather-Resources","value":-100},
                {"name":"Make-Friends","value":-100},
                {"name":"Learn-Skills","value":-100},
                {"name":"Gain-Experience","value":-100},
                {"name":"Earn-Money","value":-100}
            ]}""")

with open(base_json_path, "r") as file:
    loaded_json = json.load(file)


def save_output(_loaded_json,_output_json_name):
    with open(_output_json_name, "w") as f:
        json.dump(_loaded_json, f,  indent=4)


def remove_action(_loaded_json, action_name):
    action_found = False
    for idx, action in enumerate(_loaded_json["actions"]):
        if action["name"] == action_name:
            del _loaded_json["actions"][idx]
            action_found = True

    if action_found:
        print("Removed action:", action_name)
    else:
        print("Action",action_name,"not found")
    save_output(_loaded_json, output_json_name)


def add_action(_loaded_json, action_definition):
    action_already_there = False
    for idx, action in enumerate(_loaded_json["actions"]):
        if action["name"] == action_definition["name"]:
            action_already_there = True
            break

    if action_already_there:
        print("Action already there")
        pass
    else:
        _loaded_json["actions"].append(action_definition)
        print("Added action:", action_definition["name"])

    save_output(_loaded_json, output_json_name)


remove_action(loaded_json, "Fight-Monster")
remove_action(loaded_json, "Explore-Dungeon")
remove_action(loaded_json, "Explore-Dungeonxxx")
add_action(loaded_json, mockup_action)