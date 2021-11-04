#%%
## Task Two

#Import necessary packages
import numpy as np
import matplotlib.pyplot as plt

#Define the x,y plane
x = np.linspace(0,5)
y = np.linspace(0,5)

#Plot the constaints given
plt.plot(x,1/2*(3-x),label='x + 2y >= 3')
plt.plot(x,5-2*x, label='2x + y >= 5')

#Plot versions of the objective function to identify the minimum
plt.plot(x,1/2*(10-2*x), label = 'Objective function f=10', linestyle = 'dotted')
plt.plot(x,1/2*(5.35-2*x), label = 'Objective function f=10', linestyle = 'dotted')
#Use a different line style to differentiate 

#Label the axes
plt.xlabel('x')
plt.ylabel('y')

#Title the graph
plt.title("Plotting Linear Constraints")

#Set limits of axes
plt.xlim(0,5)
plt.ylim(0,5)

#Add a legend to indentify the lines
plt.legend()
#Add text to identify the Feasibility Region
plt.text(3,3, "Feasibility Region")
plt.show()
#%%
##Task Four

#Identify our objective function
f = [-1,2,3,1]

#Form an array of our constraints
A = [[1,-1,-2,-1],[2,0,5,-4],[-2,1,0,1]]

#Form an array of our solutions
b = [4,2,1]

#Indentify the bounds on our decision variables
x1_bounds = (0,None)
x2_bounds = (0,None)
x3_bounds = (0,None)
x4_bounds = (0,None)

#Find the maximum of our objective function by minimising the negative of the function
from scipy.optimize import linprog
res = linprog(f, A_ub=A, b_ub=b, bounds=(x1_bounds, x2_bounds, x3_bounds, x4_bounds),options={"disp": True})

#Now print our results
print(res)
#%%
##Task Five
#Identify our objective function
f = [4,12,1]

#Form an array of our constraints
A = [[-1,-4,1],[-2,-2,-1]]

#Form an array of our solutions
b = [-2,-2]

#Indentify the bounds on our decision variables
y1_bounds = (0,None)
y2_bounds = (0,None)
y3_bounds = (0,None)

#Find the maximum of our objective function by minimising the negative of the function
from scipy.optimize import linprog
dual = linprog(f, A_ub=A, b_ub=b, bounds=(y1_bounds, y2_bounds, y3_bounds),options={"disp": True})

#Now print our results
print(dual)

print("")

#Identify our objective function
f = [-2,-2]

#Form an array of our constraints
A = [[1,2],[4,2],[-1,1]]

#Form an array of our solutions
b = [4,12,1]

#Indentify the bounds on our decision variables
x1_bounds = (0,None)
x2_bounds = (0,None)


#Find the maximum of our objective function by minimising the negative of the function
from scipy.optimize import linprog
orig = linprog(f, A_ub=A, b_ub=b, bounds=(x1_bounds, x2_bounds),options={"disp": True})

#Now print our results
print(orig)
#%%
##Task Six

#Import neccessary libraries
from random import weibullvariate as wb
import numpy as np
import math
import matplotlib.pyplot as plt

#Generate 100,000 random numbers
nsample = 100000

data = [] #create an empty list

for _ in range(nsample): 
    value = wb(5,20)
    data.append(value) #add the weibull values to this list
    
def weibull(v,a,b): #define a new function
    weib = (b/a)*((v/a)**(b-1))*(math.exp(-(v/a)**b)) #use the PDF
    return weib #returns the probability of the wind blowing with a given speed

v_vec = np.linspace(3,6,100000) #define the space

p = [] #create an empty list

for v in v_vec: 
    p1 = weibull(v,5,20) #use the new function to work out the probability 
    p.append(p1) #add the probability to the list
    
plt.xlim = (3,6) #set x limit for the histogram
plt.ylim = (0,1.5) #set the y limit for the histogram
    
plt.hist(data, density=True,bins=25) 
#plot histogram, density = true normalises the data
#and change the bins to make it more accurate
plt.plot(v_vec,p,label='PDF',linestyle = 'dashed') #add the PDF
plt.xlabel('Wind Speed (v)') #add label to x-axis
plt.ylabel('Density') #add label to y'axis
plt.legend() #add a legend for the labels
plt.show() #show the histogram and PDF on the same plot

PV = [] #create an empty list

def power(i,Pm,vA,C): #define a new function
    powr = Pm/(1+(math.exp((vA-i)/C))) #use the given formula for power
    return powr #returns the power of the wind turbine

for i in range(nsample):
    PV1 = power(data[i],4.5,5,0.5) #use the new function to work out the power
    PV.append(PV1) #add this value to the new list

plt.hist(PV, color = 'pink',bins=25)
plt.xlabel('Wind Speed (v)')
plt.ylabel('Density')
plt.show()
meanPV = np.mean(PV)
print(f'\n The mean power from the simulation is {meanPV:.4f} MW')
#%%
##Task Seven

#Import necessary libraries
import numpy as np
import math
from timeit import default_timer as timer

def function(x1,x2,x3,x4,x5): #create a new function
    sum1 = 0 #create new variable and set to 0
    sum2 = 0 #create new variable and set to 0
    x = [x1,x2,x3,x4,x5] #create a new array for x
    a = [1,0.5,0.2,0.2,0.2] #create an array of a values given
    for i in range(0,5): #create a for loop for outside sum
        for j in range(0,5): #create a for loop for inside sum
            if j == i: #create if statment to prevent i=j
                sum2 = sum2 #if i=j don't change the sum
            else: 
                sum2 += x[j] #if i=j is not true then do the inside sum
        sum1 += 0.5*a[i]*x[i]**2*(2+math.sin(sum2)) #using given formula for outside sum
    return math.exp(sum1) #return the exponentional of the outside sum

Nsample = 10**6
x = np.zeros(Nsample) #create an array of zeroes of size Nsample

start = timer()  #starting timing

for i in range(0,Nsample):
    rx1 = np.random.rand() #create a random value for x1 between 0 and 1
    rx2 = np.random.rand() #create a random value for x2 between 0 and 1
    rx3 = np.random.rand() #create a random value for x3 between 0 and 1
    rx4 = np.random.rand() #create a random value for x4 between 0 and 1
    rx5 = np.random.rand() #create a random value for x5 between 0 and 1
    x[i] = function(rx1,rx2,rx3,rx4,rx5)
    
meanr = np.mean(x); #mean of the function value with the random x1-5
err = np.std(x)/math.sqrt(Nsample) #error of the mean

end = timer() #stop timing

timetaken = end - start #calculates the time taken to compute the Monte Carlo estimate

print(f"\nThe Monte Carlo estimate is {meanr:.6f} +/- {err:.6f} to 6 decimal places")
print(f"It took {timetaken:.2f} seconds to compute this estimate")

Terr = 10**-5 #target error is 10^-5
Tsample= np.round((np.std(x)/Terr)**2) #this calculates the target sample

#estimated time to achive required error is Tsample/Nsample
est_time = (Tsample/Nsample)*timetaken

print(f"\nThe expected number of samples needed to solve I5 with error of 10^-5 is {Tsample:.0f}")
print(f"and it is expected to take {est_time:.2f} seconds")

#%%
##Task Eight

#Import necessary libraries
import numpy as np
import math
import scipy.integrate as integrate
import matplotlib.pyplot as plt


#Using scipy.integrate.quad to find I
integrand = lambda x: math.exp((math.sin(3*(x**2))))
lim = ([0,1])

Ip = integrate.quad(integrand,lim[0],lim[1]) #calculate integral
value = Ip[0] #quad approximate value of integral
error = Ip[1] #error of approximation
print('\nscipy.integrate value: ',value,'\nerror: +/-',error)
print('\n\n')

nplot = [1000,10000,100000,1000000] #array of sample sizes

def Imc(x): #define function that will be integrated
    sum1 = math.exp((math.sin(3*(x**2))))
    return(sum1)

error0S = [] #create an array for errors of standard Monte Carlo
error2S = [] #create an array for errors of 2 sets of S.S.
error4S = [] #create an array for errors of 4 sets of S.S.

for n in nplot:
    I0S = [] #array of solutions of Monte Carlo method for each n
    
    for i in range(n):
        rand0S = np.random.rand()

        I0S.append(Imc(rand0S))
        
    sample0S = np.mean(I0S)
        
    error0 = np.std(I0S)/math.sqrt(n) #error of the mean of 1st sample

    error0S.append(error0)
    print('Monte Carlo with ',n,' samples in 0 sets: ',sample0S,'\nerror: +/-',error0)


print('\n\n')

#Using Monte Carlo with 2 equal sets of statified sampling and a range of sample size's to find I
for n in nplot:
    I1 = [] #array of solutions of 2 sets of S.S. between 0 and 0.5
    I2 = [] #array of solutions of 2 sets of S.S. between 0.5 and 1
    
    n12 = int(n*1/2) #account for sample size being doubled
    
    for i in range(n12):
        rand1 = np.random.rand()*0.5 #random dist. form 0-0.5
        rand2 = np.random.rand()*0.5+0.5 #random dist. form 0.5-1
        I1.append(1/2*Imc(rand1))
        I2.append(1/2*Imc(rand2))

    sample1 = np.mean(I1)
    sample2 = np.mean(I2)
        
    sample = sample1 + sample2
    error1 = np.std(I1)/math.sqrt(n12) #error of the mean of 1st sample
    error2 = np.std(I2)/math.sqrt(n12) #error of the mean of 2st sample
    
    error = math.sqrt(error1**2+error2**2) #error of the mean of the full sample
    error2S.append(error)
    print('Monte Carlo with ',n,' samples in 2 sets: ',sample,'\nerror: +/-',error)
    
print('\n\n')

#Using Monte Carlo with 4 equal sets of statified sampling and a range of sample size's to find I
for n in nplot:
    I1 = [] #array of solutions of 4 sets of S.S. between 0 and 0.25
    I2 = [] #array of solutions of 4 sets of S.S. between 0.25 and 0.5
    I3 = [] #array of solutions of 4 sets of S.S. between 0.5 and 0.75
    I4 = [] #array of solutions of 4 sets of S.S. between 0.75 and 1
    
    n24 = int(n*1/4) #account for sample size being quadrupled 
    
    for i in range(n24):
        rand1 = np.random.rand()*0.25 #random dist. from 0-0.25
        rand2 = np.random.rand()*0.25+0.25 #random dist. from 0.25-0.5
        rand3 = np.random.rand()*0.25+0.50 #random dist. from 0.5-0.75
        rand4 = np.random.rand()*0.25+0.75 #random dist. from 0.75-1
        I1.append(1/4*Imc(rand1))
        I2.append(1/4*Imc(rand2))
        I3.append(1/4*Imc(rand3))
        I4.append(1/4*Imc(rand4))

    sample1 = np.mean(I1)
    sample2 = np.mean(I2)
    sample3 = np.mean(I3)
    sample4 = np.mean(I3)
    sample = sample1 + sample2 + sample3 + sample4
        
    error1 = np.std(I1)/math.sqrt(n24) #error of the mean of 1 st. sample
    error2 = np.std(I2)/math.sqrt(n24) #error of the mean of 2 st. sample
    error3 = np.std(I3)/math.sqrt(n24)
    error4 = np.std(I4)/math.sqrt(n24)
    error = math.sqrt(error1**2+error2**2+error3**2+error4**2) #error of the mean of the full sample
    error4S.append(error)
    
    print('Monte Carlo with ',n,' samples in 4 sets: ',sample,'\nerror: +/-',error)

    

plt.loglog(nplot,error0S,label='Standard Monte Carlo',color='orange')
plt.loglog(nplot,error2S,label='Monte Carlo 2 S.S.',color='pink')
plt.loglog(nplot,error4S,label='Monte Carlo 4 S.S',color='cornflowerblue')
plt.xlabel('Size of sample (n)')
plt.ylabel('Error in each step')
plt.title('Log Log plot of errors against n')
plt.legend(loc="lower left")
plt.show()

#%%
##Task Nine

#Import neccessary libraries
from scipy.optimize import linprog
import sqlite3

conn = sqlite3.connect('order_list.db') #creat connection to database

c = conn.cursor() #creat cursor object to execute from

c.execute("""SELECT * FROM order_summary""") #select all the information from the table order_summary

H = c.fetchall() #creat a varible H which holds that information in a tuple of arrays
order=[]
for i in range(4): 
    order.append(H[i][1]) #appending the relevent information about the orders only to a new variable order

#  objective function  minimize
f = [400,400,400,400,500,500,500,500,2,2,2,2] #objective function

# A x < b
A = [[0,0,0,0,0,0,0,0,-1,0,0,0],[0,0,0,0,0,0,0,0,0,-1,0,0],\
     [0,0,0,0,0,0,0,0,0,0,-1,0],[0,0,0,0,0,0,0,0,0,0,0,-1]]

b = [0,0,0,0]

#  equality constraints
A_eq = [[1,0,0,0,1,0,0,0,-1,0,0,0],[0,1,0,0,0,1,0,0,1,-1,0,0 ],\
        [0,0,1,0,0,0,1,0,0,1,-1,0],[0,0,0,1,0,0,0,1,0,0,1,-1]]

#using that varible of information extracted from the data base we can input it directly in to the code
b_eq = [order[0]-10,order[1],order[2],order[3]] 

##
xb = (0, order[0])
yb = (0, 10000)
ib = (None, None)

# Find minimum

res = linprog(f, A_ub=A, b_ub=b, A_eq=A_eq, b_eq=b_eq, \
              bounds=(xb,xb,xb,xb, yb,yb,yb,yb, ib,ib,ib,ib ), options={"disp": False})

print(res)

print("The minimum cost  = " , res.fun )

#%%
##Task Ten

import sqlite3
import numpy as np


##create connnection the the database HR_info.db
connection = sqlite3.connect("""HR_info.db""")
##create cursor object
cursor = connection.cursor()

##defining 'results' as the entiretly of the data from HR_info table
cursor.execute("""SELECT * FROM HR_info """) 
results = cursor.fetchall() 

##defining 'result' as the entiretly of the data from flu_info table
cursor.execute("""SELECT * FROM flu_info """) 
result = cursor.fetchall() 


##Using the join command to define 'info' as the staff_number, age, days_off and salary of all who contracted the flu
cursor.execute("""SELECT HR_info.staff_number,age,days_off,salary FROM HR_info INNER JOIN flu_info ON HR_info.staff_number = flu_info.staff_number""")
info = cursor.fetchall()



cost_flu = 0 #total cost to company of days off due to flu
cost_jab = 0 #total cost to the company to vaccinate all staff under 65 years
days = 0 #counting the cummaltive days off due to flu
for i in info:
    cost_flu += i[2]*i[3]*(1/365)
    days += i[2]
    if i[1]<65:
        cost_jab += 13

print('total cost due to flue: £',np.round(cost_flu,2))
print('with ',days,' days taken off overall')
print('total cost to treat all staff under 65: £',cost_jab)
print('\n\n')
print('The company saves £',np.round(cost_flu,2)-cost_jab,'by vaccinating its staff')
print('\n\n')

##creating a new table flu_shot to hold data on whether a staff memeber has been vaccinated
cursor.execute('''CREATE TABLE IF NOT EXISTS flu_shot(staff_number integer PRIMARY KEY, flu_jab integer)''')
##'insert' in a funtion to add the staff number and flu info the flu_shot database
def insert(entity):
    cursor.execute('''INSERT OR REPLACE INTO flu_shot(staff_number,flu_jab) VALUES(?,?)''', entity)
##'x' is an empty array of size 1,2: (0,0)
x = np.zeros([1,2])
xs = []
##for every iteration this adds the staff number and either a 0 or 1 to signife if they have recived a flu shot or not
for i in range(30):
    print('has staff id ',results[i][0],' recived a flue shot? 0=no, 1 =yes: ')
    xs = input('yes/no: ')
    x[0][0] = i
    x[0][1] = xs
    insert((x[0][0],x[0][1]))
    
##saving the changes to the database
connection.commit()

##checking that the new table has been updated properly and contains the proper info
cursor.execute("""SELECT * FROM flu_shot""")
res = cursor.fetchall()
for r in res:
    print(r)
    
##closing the connection
connection.close()

#%%
##Task Thirteen

#import necessary libraries
from scipy.optimize import linprog
import matplotlib.pyplot as plt
from scipy import stats as stat
import numpy as np

f = [1, 1] #our objective function
hit = 0 #counter of total number of succesfull terminations
miss = 0 #counter of total number of unseccesfull terminations
investment = [] #empty list for investment values
for i in range(1000):
    #a random value taken from the normal distribution mean=-0.1, std=0.03
    norm1 = np.random.normal(-0.1,0.03) 
    # a random value taken from the normal distribution mean=-0.4, std=0.1
    norm2 = np.random.normal(-0.4,0.1)
    
    A = [[ -0.12, -0.04],[norm1, norm2]] 
    b = [-600, -1000]   
    x0_bounds = (None, None)
    x1_bounds = (None, None)
    
    #we use the Try: command to implement a test which allows acceptions to errors
    try:
        res = linprog(f, A_ub=A, b_ub=b, bounds=(x0_bounds, x1_bounds ),
                      options={"disp": False})    
    #we use the except command to tell python to ignore errors of 
    #type ValueError and to continue running the simulation
    except (ValueError):
        #we add 1 to the value of failed simulations
        miss+=1  
    #res.success holds the information on whether the termination is successfull 
    #if true we add the invesment estimation to a list  
    if res.success == True:
        hit+=1
        #the recommened ammount to be invested is appended to a list
        investment.append(res.fun) 

mean = np.round(np.mean(investment),2)
std = np.round(np.std(investment),2)
mode = np.round(stat.mode(investment, axis= None),2)
percent = (miss/(hit+miss))*100

print(f'\nFrom 1000 samples the optimisation failed {percent:.2f}% of the time')
print('\nThe mean, mode and standard deviation of the recommended invested amount is:')
print(f'Mean: {mean:.2f}')
print('Mode:', *mode[0], sep = " ")
print(f'Standard deviation: {std}')

#plot histogram of data
plt.hist(investment,bins='auto')
plt.title('Histogram of objective function')
plt.xlabel('Investment value')
plt.ylabel('Frequency')

#%%
##Task Fifteen

#import necessary libraries
import pulp

#initialising a linear programming (lp) problem to minimise an objective funtion using pulp library
my_lp_problem = pulp.LpProblem("My LP Problem", pulp.LpMinimize)

#defining the 8 camera locations (x1-8) as an integer value (0-1) for all variables
x1 = pulp.LpVariable('x1', lowBound=0, cat='Integer')
x2 = pulp.LpVariable('x2', lowBound=0, cat='Integer')
x3 = pulp.LpVariable('x3', lowBound=0, cat='Integer')
x4 = pulp.LpVariable('x4', lowBound=0, cat='Integer')
x5 = pulp.LpVariable('x5', lowBound=0, cat='Integer')
x6 = pulp.LpVariable('x6', lowBound=0, cat='Integer')
x7 = pulp.LpVariable('x7', lowBound=0, cat='Integer')
x8 = pulp.LpVariable('x8', lowBound=0, cat='Integer')

#defining the objective funtion
my_lp_problem += x1+x2+x3+x4+x5+x6+x7+x8

#Adding the list of all possible contraints for the objective fuction
my_lp_problem += x1+x2>=1
my_lp_problem += x2+x3>=1
my_lp_problem += x4+x5>=1
my_lp_problem += x7+x8>=1
my_lp_problem += x6+x7>=1
my_lp_problem += x2+x6>=1
my_lp_problem += x1+x6>=1
my_lp_problem += x4+x7>=1
my_lp_problem += x2+x4>=1
my_lp_problem += x5+x8>=1
my_lp_problem += x3+x5>=1

#printing the problem objective and constraints
print(my_lp_problem)

#solveing the lp problem
my_lp_problem.solve()
#confirming that this is indeed the optimal solution
print('This solution is: ',pulp.LpStatus[my_lp_problem.status])

#printing those variables that have a solution equal to one as those are the locations 
#of the camera's that are most optimal
for variable in my_lp_problem.variables():
    if variable.varValue == 1:
        print ("{} = {}".format(variable.name, variable.varValue))

