import Point_Searching_GOAPwithIDAstar

if __name__ == '__main__':
    min_maxDepth = int(input("Set the smallest max_depth value"))
    max_maxDepth = int(input("Set the biggest max_depth value"))

    for i in range(min_maxDepth, max_maxDepth+1):
        print(f'Running PointSearching with max_depth = {i}, '
              f'time[s]: {Point_Searching_GOAPwithIDAstar.main(i,should_goal_move=False)}')