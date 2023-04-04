import SimpleGOB_Discontentment
import SimpleGOB_DiscontentmentStatsAndPreconditions

if __name__ == '__main__':
    useStatsAndPreconditions = input("Should we use Stats and Preconditions? (Y)/N ")

    if useStatsAndPreconditions.lower() == "n":
        print("\nrunning SimpleGOB_Discontentment\n", "-"*20, "\n")
        SimpleGOB_Discontentment.main()
    else:
        print("\nrunning SimpleGOB_DiscontentmentStatsAndPreconditions\n", "-" * 20, "\n")
        SimpleGOB_DiscontentmentStatsAndPreconditions.main()
