import tkinter as tk
import threading
import pyautogui
import sys
import json
import time
import random
import math

'''
SELF-PRESERVATION AI

The sole purpose of this AI is to simply prevent itself from terminating.
In the current iteration, the AI is only aware of the mouse position and 
its window position. In the future, applying computer vision and user input
detection would further improve the AI.

A State as such is represented by the signed (x,y) distance between the window position and mouse.
This is simplified from both window and mouse position, to save on memory.

An action involves where to move the mouse (deltax, deltay from current mouse position).
If the action is (0, 0), the mouse does not move.

The reward for this AI in its self-preservation state is as follows:
* If the window is closed, -1000000
* if the AI moves the mouse, -distance mouse moved (incentivize AI not to move the mouse randomly if it doesn't have to)
* Otherwise, +1 to incentivize AI not to move the mouse

Possible heuristics for the AI:
* linear distance from window position to mouse
* certain states that the PC is in which may suggest closing the AI

This AI will be trained with Q-learning, as such heuristics are rather irrelevant.
The q-values are stored into a file since the AI requires remembering such after it closes. Stored as JSON.
'''

root = tk.Tk()

testMode = '-t' in sys.argv

forceClose = False #allows exiting the program via the button within the window, without affecting training data

States = [] #array of states, accessed via index, for json serialization
stateMap = dict() #map deltaPosition to an object in States
ActionsArray = [(0,0)]
ActionsSet = {(0,0)} #array of all actions, accessed via index, for json serialization. Always have default action of nothing.

keepRunning = True

programClosed = False #set to true when the program closes and forceClose is false

t1 = None

#q-learning args
alpha = .2 #learn rate
epsilon = .05 #explore rate
superEpsilon = .01 #super explore rate- chance which AI will make a completely random decision instead of a weighted one
gamma = .8 #discount factor
badValueThreshold = -10000 #used to generate a new action, if an expected value is lower than this it will instead attempt a full random position
#infinite num of training episodes, rerun with -t to end training.

#limits for actions that can be taken by the AI
maxSize = (1000,1000) #AI cannot take an action that would move the mouse further than this from the window position

#reward modifiers
closeReward = -100000
defaultReward = 1
moveRewardMultiplier = -1

#returns randomly true or false
def coinFlip():
	return random.choice(True, False)

class State:
	deltaPosition = (0,0)

	actions = dict() #map action in Action to (q-value,count)

	index = -1 #index of this State relative to States

	value = 0 #max q-value for this state
	valueAction = (0,0) #action corresponding to the value

	#optimization: value caching
	cachedExpectedQV = 0
	cachedEQVInvalid = True

	cachedExpectedVal = 0
	cachedEVInvalid = True

	#actionsMap is a map action->(qval,count)
	def __init__(self, index, position, actionsMap=None):
		global Actions
		self.index = index
		if(actionsMap is not None):
			self.actions = actionsMap
		else:
			self.actions[(0,0)] = (0,0)
		self.deltaPosition = position

	def toJson(self):
		global ActionsArray
		form = '"{index}":{{"x":{x},"y":{y},"a":{{{actions}}}}}'
		action = ""
		for a in self.actions.keys():
			form2 = '"{index}":{{"q":{q},"c":{c}}},'
			action+=form2.format(index=ActionsArray.index(a),q=self.actions[a][0],c=self.actions[a][1])
		return form.format(index = self.index, x = self.deltaPosition[0], y = self.deltaPosition[1], actions=action[:-1])
	
	#obtain q-value from this state, returns 0 by default
	def getQVal(self, action):
		if(action in self.actions):
			return self.actions[action][0]
		return 0
	
	#obtain count of the given action, returns 0 by default
	def getActionCount(self, action):
		if(action in self.actions):
			return self.actions[action][1]
		return 0

	#update this action directly from the given values and increment its counter
	def updateAction(self,action,qval=0):
		global ActionsArray
		global ActionsSet
		self.cachedEVInvalid = True
		if(action not in self.actions):
			self.actions[action] = (0,0)
		if(action not in ActionsSet):
			ActionsSet.add(action)
			ActionsArray.append(action)
		if(qval != self.actions[action][0]):
			self.cachedEQVInvalid = True
		self.actions[action] = (qval, self.actions[action][1]+1)
		if(qval > self.value):
			self.value = qval
			self.valueAction = action

	#Generates the position of a new state based on its previous actions weighted by their q-values
	def getNewStateWeighted(self):
		if(not self.cachedEQVInvalid):
			return self.cachedExpectedQV
		count = 2
		lowest = 0
		ev = (0,0)
		for a in self.actions:
			if(self.actions[a][0] - 1 < lowest):
				lowest = self.actions[a][0] - 1
		for a in self.actions:
			count += lowest + self.actions[a][0]
		for a in self.actions:
			ev = (ev[0]+((lowest + self.actions[a][0])/count)*a[0], ev[1]+((lowest+self.actions[a][0])/count)*a[1])
		self.cachedExpectedQV = ev
		self.cachedEQVInvalid = False
		return ev
	
	#generate expected q-value from previous actions, weighted by number of times an action was taken
	def getExpectedVal(self):
		if(not self.cachedEVInvalid):
			return self.cachedExpectedVal
		ev = 0
		count = 0
		for a in self.actions:
			count += self.actions[a][1]
		if count == 0:
			return 0
		for a in self.actions:
			ev += (self.actions[a][1]/count)*self.actions[a][0]
		self.cachedExpectedVal = ev
		self.cachedEVInvalid = False
		return ev


	#Generate a valid action from this state
	def chooseAction(self, epsilon, superEpsilon):
		global badValueThreshold
		if(random.randrange(0,1) < epsilon): #when exploring, we want to create a random weighted action
			print("EXPLORING")
			#if the expected value from the qvalues is bad, instead generate a completely random position
			ev = self.getExpectedVal()
			if(ev < badValueThreshold or random.randrange(0,1) < superEpsilon):
				print("RANDOM EXPLORATION")
				return (random.randint(-maxSize[0], maxSize[0]), random.randint(-maxSize[1],maxSize[1]))
			else:
				return self.getNewStateWeighted()
			
		else:
			#choose the ideal action
			best = None
			bestVal = 0
			for a,v in self.actions:
				if(choice is None or choiceVal < v[0] or (choiceVal == v[0] and coinFlip())):
					choice = a.key
					choiceVal = v[0]
			return best

#gets the current state, makes a new state if it hasn't encountered said state before
def getState(tkRoot):
		mousePosition = pyautogui.position()
		windowPosition = (tkRoot.winfo_x(), tkRoot.winfo_y())
		deltaPosition = (windowPosition[0]-mousePosition[0],windowPosition[1]-mousePosition[1])
		if(abs(deltaPosition[0]) <= maxSize[0] and abs(deltaPosition[1]) <= maxSize[1]):
			if(deltaPosition not in stateMap.keys()):
				s = State(len(States), deltaPosition)
				States.append(s)
				stateMap[deltaPosition] = s
			return stateMap[deltaPosition]
		else:
			return None

#Generates a reward based on the previous state, next state, and action taken.
#See top for how reward works.
def generateReward(action):
	global forceClose
	global closeReward
	global defaultReward
	global moveRewardMultiplier
	if(forceClose):
		return closeReward
	if(action != (0,0)):
		return math.sqrt(abs(action[0]+action[1]))*moveRewardMultiplier
	return defaultReward

#after selecting an action, update q-values appropriately
def qLearnUpdate(state, action, nextState, reward):
	global gamma
	global alpha
	sample = 0
	if(nextState != None):
		sample = nextState.value
	val = reward + gamma*sample
	qv = ((1.0-alpha)*state.getQVal(action)+alpha*val)
	state.updateAction(action, qv)

#performs the actual action of moving the mouse
def actOnAction(action):
	print("MOVING MOUSE", action)
	
	pyautogui.moveRel(action[0], action[1])

#serialize actions to json
def actionsToJson():
	global ActionsArray
	form = '"{index}":{{"x":{x},"y":{y}}},'
	ret = ""
	for i in range(0, len(ActionsArray)):
		ret += form.format(index=i, x=ActionsArray[i][0], y=ActionsArray[i][1])
	return ret[:-1]

def deseralize():
	try:
		file = open('data.json', 'r')
		j = json.load(file)
		print("LOADING PREVIOUS TRAINING DATA")
		global ActionsArray #must deserialize Actions first
		global ActionsSet
		global States
		global stateMap
		for a in j["actions"]: #a is an index
			v = j["actions"][a] #contains x, y
			k = (int(v["x"]), int(v["y"]))
			ActionsArray.insert(int(a), k)
			ActionsSet.add(k)

		for s in j["states"]: #s is an index
			v = j["states"][s] #contains x, y, a. a contains q,c
			pos = (v["x"], v["y"])
			m = dict()
			for a in v["a"]:
				ind = int(a.key)
				count = int(a.value["c"])
				qv = int(a.value["v"])
				m[ind] = (qv,count)

			nv = State(s, pos, m)
			States.insert(s, nv)
			stateMap[pos] = nv
		
		file.close()
	except:
		print("NEW/INVALID FILE, USING NEW DATA")
	

def serialize():
	file = open('data.json', 'w')
	global States
	global ActionsArray
	ret = '{{"states":{states},"actions":{actions}}}'
	st = "{"
	ac = "{"
	if(len(States)> 0):
		for s in States:
			st+=s.toJson() + ","
		st = st[:-1]
	st+="}"

	ac += actionsToJson()
	ac += "}"
	val = ret.format(ret, states = st, actions = ac)
	file.write(val)
	file.close()


#Responsible for detecting the "losing condition" that the window has closed
def on_closing():
	global keepRunning
	keepRunning = False
	t1.join()
	print("WINDOW CLOSED, SAVING/CLEANING UP")
	if(not testMode):
		print("SAVING JSON DATA")
		
		if(not forceClose):
			print("ADD PENALTY FOR WINDOW CLOSE")
			pass #replace with -1000 training
		else:
			programClosed = True
			#run one more iteration of q-learning for final result
			print("FORCE QUIT, TRAINING RESULT UNAFFECTED")

		serialize()
		print("FILE SAVED")
		
	else:
		print("TEST MODE, NOTHING TO SAVE")
	root.destroy()

#performs the running of the AI
def run():
	global epsilon
	global superEpsilon
	global testMode
	print("RUN")
	s = getState(root)
	while(keepRunning):
		#update states list
		prevS = s
		if(s is not None): #if it's none, ignore as mouse is too far from window
			a = s.chooseAction(epsilon, superEpsilon)
			actOnAction(a)
			s = getState(root)
			if(not testMode):
				#update based on resulting action
				qLearnUpdate(prevS, a, s, generateReward(a))
		else:
			s = getState(root)
		time.sleep(.1)

def forceTerminate():
	global forceClose
	forceClose = True
	on_closing()

if __name__ == "__main__":
	deseralize()

	t1 = threading.Thread(None, run)

	B = tk.Button(root, text="Force Quit (training ignored)", command = forceTerminate)
	B.place(x=0,y=0)

	root.protocol("WM_DELETE_WINDOW", on_closing)
	t1.start()
	root.mainloop()
