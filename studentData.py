import pandas as pd
student_data = [
   [3,85,75,6,'Pass'],
    [2,60,55,5,'Fail'],
    [4,90,80,7,'Pass'],
    [1,40,35,4,'Fail'],
    [3.5,88,78,6.5,'Pass'],
    [2.2,65,50,5.5,'Fail'],
    [4.5,95,88,7.2,'Pass'],
    [1.5,50,45,5,"Fail"]
 ]
pd = pd.DataFrame(student_data,columns=['StudyHours','Attendence','PreviousGrade','SleepHours','Performance'])
print(pd)
pd.to_csv("student_Data.csv")# creating csv file