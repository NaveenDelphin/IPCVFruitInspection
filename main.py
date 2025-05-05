from firstTask import run1 
from secondTask import run2 
from finalTask import run3

def main():
    # Calling the functions from the other files
    while(1):
        taskNo = int(input("Enter the Task no. (1,2 or 3) you want to perform:"))
        if(taskNo == 1):
            run1()
            taskNo = 0
            

        elif(taskNo == 2):
            run2()
            taskNo = 0
            

        elif(taskNo == 3):
            run3()
            taskNo = 0
        

if __name__ == "__main__":
    main()

