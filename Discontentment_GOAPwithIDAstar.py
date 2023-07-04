import matplotlib.pyplot as plt
import sys
from shapely.geometry import Point, Polygon, LineString
import time
import operator
import json
import copy


class WorldState:
    MAX_GOAL_VALUE = 100
    MIN_GOAL_VALUE = 0

    def __init__(self, _stats, _goals, _actions):
        self.stats = _stats
        self.goals = _goals
        self.actions = _actions
        self.action_index = 0
        self.discontentment = self.calculate_current_discontentment()

    def copy(self):
        return WorldState(copy.deepcopy(self.stats), copy.deepcopy(self.goals), copy.deepcopy(self.actions))

    def calculate_current_discontentment(self):
        discontentment = 0.0
        for goal_name, goal_value in self.goals.items():
            discontentment += pow(goal_value, 2)
        return discontentment

    def apply_action(self, action):
        for goal_change in action.single_action["goalsChange"]:
            goal_name = goal_change["name"]
            if goal_name in self.goals:

                self.goals[goal_name] -= goal_change["value"]

                if self.goals[goal_name] > self.MAX_GOAL_VALUE:
                    self.goals[goal_name] = self.MAX_GOAL_VALUE
                elif self.goals[goal_name] < self.MIN_GOAL_VALUE:
                    self.goals[goal_name] = self.MIN_GOAL_VALUE

        if "statsChange" in action.single_action:
            for stat_change in action.single_action["statsChange"]:
                stat_name = stat_change["name"]
                if stat_name in self.stats:
                    self.stats[stat_name] += stat_change["value"]

        new_calculated_discontentment = self.calculate_current_discontentment()
        # print(f'\t\tworldState: new_calculated_discontentment = {new_calculated_discontentment}')
        self.discontentment = new_calculated_discontentment

    def next_action(self):
        if self.action_index < len(self.actions):
            next_action = self.actions[self.action_index]
            self.action_index += 1
            return next_action, self.action_index-1
        else:
            return None, None

    def __hash__(self):
        goal_hash = hash(frozenset(self.goals.items()))
        stat_hash = hash(frozenset(self.stats.items()))
        return hash((goal_hash, stat_hash))

    def __str__(self):
        return f"({self.goals}, {self.stats})"

    def __eq__(self, other):
        if isinstance(other, WorldState):
            return (self.goals == other.goals and
                    self.stats == other.stats and
                    self.action_index == other.action_index and
                    self.discontentment == other.discontentment)
        return False


class Obstacles:

    def __init__(self):
        self.ops = {
            ">": operator.gt,
            ">=": operator.ge,
            "<": operator.lt,
            "<=": operator.le,
            "==": operator.eq,
            "!=": operator.ne
        }

    def preconditions_met(self, _action, _stats):
        preconditions_met_bool = True

        for idx, precondition in enumerate(_action.single_action["preconditions"]):

            if precondition["where"] == "stats":
                current_value = _stats[precondition["what"]]
                precondition_value = precondition["value"]

                logical_test_result = self.ops[precondition["logical_test"]](current_value, precondition_value)

                if not logical_test_result:
                    print(f'\t\tobstacles: failed {precondition["what"]} {precondition["logical_test"]} {precondition["value"]}')
                    preconditions_met_bool = False

        return preconditions_met_bool


class Goal:
    def __init__(self, _target_discontentment):
        self.target_discontentment = _target_discontentment

    def is_fulfilled(self, world_state):
        return self.target_discontentment >= world_state.discontentment


class Action:

    def __init__(self, _single_action, _all_actions):
        self.single_action = _single_action
        self.all_actions = _all_actions

    @staticmethod
    def get_cost():
        return 1

    def get_name(self):
        return self.single_action['name']

    def get_all_actions(self):
        return [Action(action_dict, self.all_actions) for action_dict in self.all_actions]


class Heuristic:
    def __init__(self, goal):
        self.goal = goal

    def estimate(self, state):
        return state.discontentment


class TranspositionTable:
    def __init__(self):
        self.visited = {}

    def has(self, state, depth):
        return state in self.visited and self.visited[state] <= depth

    def add(self, state, depth):
        if state not in self.visited or depth < self.visited[state]:
            print(f'\t\ttranspo:\tadding state {state} at depth: {depth}')
            self.visited[state] = depth


def plan_action(world_model, goal, heuristic, max_depth,  obstacles = None):
    cutoff = heuristic.estimate(world_model)
    goal_reached = False
    while cutoff != float('inf'):
        transposition_table = TranspositionTable()
        print('\n', '-'*20, f'plan_action: new plan action - cutoff: {cutoff}', '-'*20, '\n')
        cutoff, actions, goal_reached = do_depth_first(world_model, goal, heuristic, transposition_table, max_depth,
                                                       cutoff, obstacles=obstacles)
        if actions:
            if goal_reached:
                print(f'plan_action: goal reached')
                return actions, goal_reached
            else:
                print(f'plan_action: goal not reached')
                print(f'best plan so far:')
                for action in actions:
                    print(f'{action.get_name()}')

    return actions, goal_reached


def do_depth_first(world_model, goal, heuristic, transposition_table, max_depth, cutoff, obstacles=None):
    print(f'\n\tdo_depth_first: new do_depth_first')
    max_depth += 1

    states = [None] * (max_depth + 1)
    actions = [None] * max_depth
    costs = [0.0] * max_depth

    world_model.action_index = 0
    states[0] = world_model
    current_depth = 0
    print(f'\tdo_depth_first: starting worldModel: {states[0]}')
    transposition_table.add(states[current_depth], current_depth)   # add the starting point to transposition table

    smallest_cutoff = float('inf')

    best_path = None  # New variable for best path
    best_cost = float('inf')  # New variable for best cost

    while current_depth >= 0:

        heuristic_estimate = heuristic.estimate(states[current_depth])
        cost = costs[current_depth] + heuristic_estimate
        print(f'\tdo_depth_first:\tdepth: {current_depth};\tstate discontentment {states[current_depth].discontentment};\tcost {cost};'
              f'\tcutoff: {cutoff}')

        if goal.is_fulfilled(states[current_depth]):
            print('\n','-'*10,f' Goal {goal.target_discontentment} is fulfilled. Current discontentment is {states[current_depth].discontentment}','-'*10)
            return cutoff, actions, True

        if heuristic_estimate < best_cost:
            print(f'\tdo_depth_first: heuristic_estimate {heuristic_estimate} lower than best_cost {best_cost}')
            best_cost = heuristic_estimate
            best_path = actions[:current_depth]

        if current_depth >= max_depth - 1:
            print(f'\tdo_depth_first: current_depth ({current_depth}) too deep and goal not reached. Decreasing depth')
            current_depth -= 1
            continue

        if cost > cutoff:
            print(f'\tdo_depth_first: cost bigger than cutoff ({cost} > {cutoff})')
            if cost < smallest_cutoff:
                smallest_cutoff = cost
            current_depth -= 1
            continue

        next_action, next_action_idx = states[current_depth].next_action()
        if next_action:
            print(f'\tdo_depth_first: action, action_idx: {next_action.get_name(), next_action_idx}')

            states[current_depth + 1] = states[current_depth].copy()
            states[current_depth + 1].apply_action(next_action)

            actions[current_depth] = next_action
            costs[current_depth + 1] = costs[current_depth] + next_action.get_cost()

            if not transposition_table.has(states[current_depth + 1], current_depth + 1):
                print(f'\n\tdo_depth_first: simulating move {states[current_depth].discontentment} -> {states[current_depth + 1].discontentment} '
                      f'| new move! adding to transpo table ')
                current_depth += 1
                transposition_table.add(states[current_depth], current_depth)
            else:
                print(f'\n\tdo_depth_first: simulating move {states[current_depth].discontentment} -> {states[current_depth + 1].discontentment} '
                      f'| already in transpo. trying next move ')

            if not obstacles.preconditions_met(next_action, states[current_depth-1].stats):
                print(f'\tdo_depth_first: preconditions not met')
                current_depth -= 1
                continue

        else:
            current_depth -= 1
            print(f'\tdo_depth_first: no more actions: depth decreased')

    return smallest_cutoff, best_path, False


def main(_max_depth):
    main_start_time = time.time()
    orig_stdout = sys.stdout
    f = open('out_discontentment.txt', 'w')
    sys.stdout = f

    setup_filename = 'setup_2.json'

    # Load setup file
    with open(setup_filename, "r") as file:
        setup_file = json.load(file)

    actions = Action(None, setup_file['actions']).get_all_actions()

    start = WorldState(setup_file['stats'], setup_file['goals'], actions)
    max_plan_loops = 20

    obstacles = Obstacles()

    goal_reached = False
    plan_loop_idx = 0

    start.discontentment = start.calculate_current_discontentment()

    list_disconts = [start.discontentment]
    list_actions = []
    current_state = start.copy()

    while not goal_reached and plan_loop_idx < max_plan_loops:

        goal = Goal(list_disconts[0]*0.2)
        heuristic = Heuristic(goal)
        print(f'\tmain: goal discontentment = {goal.target_discontentment}')

        start_time = time.time()
        single_plan, goal_reached = plan_action(start, goal, heuristic, obstacles=obstacles, max_depth=_max_depth)
        end_time = time.time()
        print(f'Plan_action execution took {end_time - start_time} seconds.')

        if single_plan and not goal_reached:
            print("Partial plan found.")
        elif single_plan and goal_reached:
            print("Final plan found.")
            print(list_actions)
        else:
            print('No plan at all.')
            break

        if single_plan:
            for action_idx, action in enumerate(single_plan):
                if action is None:
                    break
                current_state = current_state.copy()
                current_state.apply_action(action)
                list_disconts.append(current_state.discontentment)
                list_actions.append(action.get_name())
                print(f'Action #{action_idx}: {action.get_name()}')

        start = current_state.copy()
        plan_loop_idx += 1


    print(f'Plan length: {len(list_disconts)-2}')
    # plt.figure(dpi=200)
    #
    # plt.plot(list_disconts, y_positions, '-', linewidth=2, label='Path')
    # plt.scatter([starting_points_x], [starting_points_y], color='g', label='Startpoints')
    # for i, txt in enumerate(starting_points_x):
    #     plt.text(starting_points_x[i] + 0.25, starting_points_y[i] + 0.25, str(i), fontsize=8, color='g')
    # plt.scatter([goal_x_points], [goal_y_points], color='r', label='Goal')
    # for i, txt in enumerate(goal_x_points):
    #     plt.text(goal_x_points[i] + 0.25, goal_y_points[i] + 0.25, str(i), fontsize=8, color='r')
    # plt.grid(True)
    # plt.axis('equal')
    # plt.xticks(range(-15, 10))
    # plt.yticks(range(-3, 15))
    # plt.legend()
    # plt.show()
    # # plt.savefig(f'./DiscontentmentFigs/Discontentment_GOAPwithIDAstar_MaxDepth-{_max_depth}.jpg', dpi=200)
    # # plt.close()

    sys.stdout = orig_stdout
    f.close()
    main_end_time = time.time()
    return main_end_time-main_start_time
#
# def main(max_depth):
#
#     filename = 'setup_2.json'
#
#     # Load setup file
#     with open(filename, "r") as file:
#         setup = json.load(file)
#
#     actions = Action(setup['actions'])
#
#     print(type(actions.get_all_actions()))


if __name__ == "__main__":
    main(3)
