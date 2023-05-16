import SimpleGOB_Discontentment
import SimpleGOB_DiscontentmentStatsAndPreconditions
import SimpleGOAP
import badGOAP
import myGOAP

if __name__ == '__main__':
    useStatsAndPreconditions = input("Should we use Stats and Preconditions? (Y)/N ")

    if useStatsAndPreconditions.lower() == "n":
        print("\nrunning SimpleGOAP\n", "-"*20, "\n")
        SimpleGOAP.main(100)
    else:
        print("\nrunning GOAP\n", "-" * 20, "\n")
        myGOAP.main(100)
