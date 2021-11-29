#!/usr/bin/env python
# coding: utf-8

# # Project 5 Python __ Creating Stone Paper Scissors.  #VIPatAINEAI
# 

# BY Data Science and Business Intelligence Intern--VIVIN GEORGE
# 
# 

# Importing required Libraries

# In[2]:


from random  import randint


# In[3]:


t = ["Stone", "Paper", "Scissors"]
'''Creating a list of possible actions available to user'''


while True:
    computer = t[randint(0,2)]
    '''Utilising randint to make competittors move'''
    """Using while Loop to ensure that the player can play as long as he/she wants"""
    player =input("Stone, Paper, Scissors?")
    """Using input to take players input among Stone , Paper , Scissor"""
    player=player.title()
    if player == computer:
        print("Tie!")
    elif player == "Stone":
        if computer == "Paper":
            print("You lose!", computer, "covers", player)
        else:
            print("You win!", player, "smashes", computer)
    elif player == "Paper":
        if computer == "Scissors":
            print("You lose!", computer, "cut", player)
        else:
            print("You win!", player, "covers", computer)
    elif player == "Scissors":
        if computer == "Stone":
            print("You lose...", computer, "smashes", player)
        else:
            print("You win!", player, "cut", computer)
    else:
         print("That's not a valid.Try again!")
    '''finding appropirate responses for each set of user-computer actions'''
    play_on=input('Do you want to conitnue playing(y/n)?')
    '''Asking user ot determine whether they want to continue playing accepting 2 responses'''
    play_on=play_on.title()
    if play_on=='N':
        print('Game over')
        break
    else:
        print("Next Try")
            
            
      
           


# In[ ]:





# In[ ]:




