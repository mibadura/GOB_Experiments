import SimpleGOB_Discontentment
import SimpleGOB_DiscontentmentStatsAndPreconditions
import SimpleGOAP
import badGOAP
import myGOAP
import GOAP_with_IDAstar_2

if __name__ == '__main__':
    whichCode = input("Should we use Stats and Preconditions? (Y)/N ")

    if whichCode.lower() == "n":
        print("\nrunning SimpleGOAP\n", "-"*20, "\n")
        SimpleGOAP.main(300)
    elif whichCode.lower() == "*":
        GOAP_with_IDAstar_2.main(300)
    else:
        print("\nrunning GOAP\n", "-" * 20, "\n")
        myGOAP.main(300)
