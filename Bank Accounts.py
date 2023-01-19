# Shahnam Khazforoosh - Student No. - 201120887

#Creating a Basic and Premium Bank account

#Importing allowed libarires

import random
from random import randint
import datetime 
from datetime import date
from datetime import timedelta

#Creating the Basic account class
class BasicAccount():
        acNum = 1
        Initialoverdraft = 0
        #Creating a random 16 integer digit card Number
        cardNum = random.randint(1**16, 10**16)
        #Setting the card expiry date to three years from todays date and shortening
        #it into month and year
        cardExp = date.today() + timedelta(days = 1095)
        cardExp = cardExp.strftime("%m/%Y")
        #Defining the users name
        #Defining the initialiser
        def __init__(self, acName, openingBalance):
        #Setting the variables to be part of this specific class using "self." function
        #and giving these variables a condition, e.g. being a float, str etc.
            self.acName = str(acName)
            self.BasicAccount = BasicAccount
            self.openingBalance = float(openingBalance)
            self.balance = float(openingBalance)
            self.acNum = BasicAccount.acNum
        #Adds 1 everytime a new account is created to acNum
            BasicAccount.acNum += 1

        #Defining my __str__ function, so whenever the user goes into the account
        #this will be the first thing that gets printed to the screen.
        def __str__(self):
          return "Welcome {self.acName}, your available balance is £{self.balance}".format(self=self)

        #Defining the deposit function
        def deposit(self, value):
            #Ensuring that value cannot be a negative amount
            if value < 0:
                return print("Deposit value must be a positive integer, please try again or use the withdrawal option if you wish to withdraw")
            else:
            #Adding amount that was deposited to the balance
                self.balance = self.balance + value
            return print("£" + str(value), "has been deposited")

        #Defining the withdraw function
        def withdraw(self, value):
        #Ensuring that withdraw amount is not a negative value
            if value < 0:
                return print("Can not withdraw £" + str(value))
            else:
            #Ensuring that the value is not greater than the balance available
            #due to this being a basic account with no overdraft
                if value < self.balance:
            #balance is updated
                    self.balance = self.balance - value
                    return print(str(self.acName) + " has withdrawn £" + str(value) + ". New balance is £" + str(self.balance))
                elif value > self.balance:
                    return print("Can not withdraw £" + str(value))


        #Defining the getAvailableBalance
        def getAvailableBalance(self):
            return self.balance


        #Defining the getBalance function
        def getBalance(self):
            return self.balance


        #Defining the printBalance function
        def printBalance(self):
            return print("£" + str(self.balance))

        #Defining the getName function
        def getName(self):
            return self.acName

        #Defining the getAcNum function
        def getAcNum(self):
        #Changing self.acNum into a string
            self.acNum = str(self.acNum)
            return self.acNum

        #Defining the issueNewCard function
        def issueNewCard(self):
        #Using another method to create another 16 digit card number and new expiry date
            self.cardExp = 0
            #Creates the expiry date which is 3 years from today
            expirydate = date.today() + timedelta(days = 1095)
            #Gets the month and year of the expiry date as separate integers
            month = int(expirydate.strftime("%m"))
            year = int(expirydate.strftime("%y"))
            #Creates a tuple with month and year added itno it
            self.cardExp = tuple([month, year])
            #Clearing the cardNum value
            self.cardNum = int(self.cardNum) - int(self.cardNum)
            #Setting a random 16 digit card number
            self.cardNum = str(random.randint(1**16, 10**16))
            return self.cardNum

        #Defining the closeAccount function
        def closeAccount(self):
        #Checks to see if balance is negative
            if self.balance < 0:
                return False
            else:
        #Clears the balance
                self.balance = self.balance - self.balance
                print(self.balance, "Has been withdrawn")
                return True
            

class PremiumAccount(BasicAccount):
        #Adds one everytime a Premium account is added to acNum
        acNum = BasicAccount.acNum + 1
        #Allows an overdraft for the accout
        overdraft = True
        #Creating a random 16 integer digit card Number
        cardNum = str(random.randint(1**16, 10**16))
        #Setting the card expiry date to three years from todays date and shortening
        #it into month and year
        cardExp = date.today() + timedelta(days = 1095)
        cardExp = cardExp.strftime("%m, %Y")
        cardExp = tuple(cardExp)
        def __init__ (self, acName, openingBalance, Initialoverdraft, overdraft = True):
        #Using super function to call variables from parent class "BasicAccount"
            super().__init__(acName, openingBalance)
        #Setting the variables to themselves using the self. function 
        #to be recalled just for this clas    
            self.Initialoverdraft = Initialoverdraft
            self.Overdraft = Initialoverdraft
            self.acName = str(acName)
            self.overdaft = bool(overdraft)
            self.openingBalance = float(openingBalance)
            self.balance = float(openingBalance)
            self.acNum = PremiumAccount.acNum
            BasicAccount.acNum += 1
    
        def __str__(self):
          return "Welcome {self.acName}, your balance is {self.balance} and you have an Initial overdraft of {self.Initialoverdraft}".format(self=self)

        #Defining the deposit function
        def deposit(self,value):
            #Ensuring deposit amount isn't a negative value
            if value > 0:        
            #Checking to see if balance is in overdraft or not and if it is, to also update overdraft balance
                if self.balance > 0:
                    #Updating balance
                    self.balance = self.balance + value
                    return print("£" + str(value), "has been deposited")
                else:
                    #Updating balance and overdraft
                    self.balance = self.balance + value
                    if self.Initialoverdraft + value > self.Overdraft:
                        self.Initialoverdraft = self.Initialoverdraft + self.Overdraft
                        return print("£" + str(value), "has been deposited")
                    else:
                        #Ensuring overdraft does not exceed allowed amount
                        self.Initialoverdraft = self.Initialoverdraft + value
                    return print("£" + str(value), "has been deposited")
            else:
                return print("Deposit value must be a positive integer and not 0, please try again or use the withdrawal option if you wish to withdraw")

        #Defining the withdraw function
        def withdraw(self, value):
            #Checking to see if withdrawal is a negative amount
            if value < 0:
                return print("Can not withdraw -£" + str(abs(value)))
            else:
                #Checking to see if withdraw amount is less than balance
                if value < self.balance:
                    #Updating balance
                    self.balance = self.balance - value
                    return print(str(self.acName) + " has withdrawn £" + str(value) + ". New balance is £" + str(self.balance))
                #Checking if value is greater than balance
                elif value > self.balance:
                    #Checking to see if user is already in overdraft
                    if self.balance > 0:
                        #Checking to see if the withdrawal amount is greater
                        #than the allowed amount including their overdraft
                        if (self.Initialoverdraft + self.balance) - value < 0:
                            return print("Can not withdraw £" + str(value))
                        #Checking to see if total amount that the user can use (Overdraft included)
                        # is greater than the withdrawal amount
                        elif (self.Initialoverdraft + self.balance) - value >= 0:
                            #setting a random variable as the withdrawl amount
                            newvalue = value - self.balance
                            #updating balance
                            self.balance = self.balance - value
                            #updating the overdraft
                            self.Initialoverdraft = self.Initialoverdraft - newvalue
                            return print(str(self.acName) + " has withdrawn £" + str(value) + ". New balance is -£" + str(abs(self.balance)))
                        #Checking if the total amount available is less than the withdrawl amount
                        #as it would not be allowed
                        if self.Initialoverdraft + self.balance < value:
                            return print("Can not withdraw £" + str(value))
                    #Checking to see if balance is orignally already in overdraft
                    elif self.balance < 0:
                        #Checking if overdraft allowance is less than value
                        if (self.Initialoverdraft - value) < 0:
                            return print("Can not withdraw £" + str(value))
                        else:
                            #updating balance
                            self.balance = self.balance - value
                            #updating overdraft
                            self.Initialoverdraft = self.Initialoverdraft - value
                            return print(str(self.acName) + " has withdrawn £" + str(value) + ". New balance is -£" + str(abs(self.balance)))

            #Defines the getAvailableBalance
        def getAvailableBalance(self):
            return self.balance + self.Initialoverdraft

            #Defining the setOverdraft limit function
        def setOverdraftLimit(self,limit):
            self.Overdraft = limit
            return print("Overdraft limit has been updated to £" + str(self.Overdraft))

            #Tried Using parent getBalance function using Super() but gave errors in test
            #So, just had to re-write it and not use the Super() function
        def getBalance(self):
            return self.balance

            #Defining the printBalance function
        def printBalance(self):
            return self.balance, self.Initialoverdraft

            #Using parent getName function using Super()
        def getName(self):
            super().getName()

            #Using parent getAcNUM function using Super()
        def getAcNum(self):
            super().getAcNum()

            #Using parent issueNewCard function using Super()
        def issueNewCard(self):
            #Using super function to get issueNewCard fucntion from parent class
            super().issueNewCard()

            #Defining the closeAccount function
        def closeAccount(self):
            #Ensuring that user is not in debt with the bank
            if self.balance < 0:
                return False, print("Can not close account due to customer being overdrawn by £", self.balance)
            else:
                #Clears the balance
                self.balance = self.balance - self.balance
                print(self.balance, "Has been withdrawn")
                return True



