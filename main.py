import SimpleGOB_Discontentment
import SimpleGOB_DiscontentmentStatsAndPreconditions
import SimpleGOAP

if __name__ == '__main__':
    useStatsAndPreconditions = input("Should we use Stats and Preconditions? (Y)/N ")

    if useStatsAndPreconditions.lower() == "n":
        print("\nrunning SimpleGOB_Discontentment\n", "-"*20, "\n")
        SimpleGOB_Discontentment.main()
    else:
        print("\nrunning SimpleGOAP\n", "-" * 20, "\n")
        SimpleGOAP.main(100)
