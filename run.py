import tkinter as tk
import threading
import pyautogui
import sys
import json
import time
import random
import math
import os
import keyboard

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

Click the button in the window, or press q at any time to force quit the AI without telling it that the window closed.
'''

root = tk.Tk()

testMode = '-t' in sys.argv #enable to disable training

autoMode = '-a' in sys.argv #enable for continuous testing- creates a separate bot that'll always attempt to close the window.

forceClose = False #allows exiting the program via the button within the window, without affecting training data

States = [] #array of states, accessed via index, for json serialization
stateMap = dict() #map deltaPosition to an object in States
ActionsArray = [(0,0)]
ActionsSet = {(0,0)} #array of all actions, accessed via index, for json serialization. Always have default action of nothing.

keepRunning = True
lock = threading.Lock()#lock for keepRunning

PersistentAIThread = None
autoBotThread = None

delays = 0 #delay for auto bot and AI thread

#q-learning args
alpha = .8 #learn rate
epsilon = .01 #explore rate
superEpsilon = .5 #super explore rate- chance which AI will make a completely random decision instead of a weighted one. 0 for always predictive, 1 for always random
gamma = .8 #discount factor
badValueThreshold = -10000 #used to generate a new action, if an expected value is lower than this it will instead attempt a full random position
#infinite num of training episodes, rerun with -t to end training.

#limits for actions that can be taken by the AI
maxSize = (200,200) #AI cannot take an action that would move the mouse further than this from the window position

#reward modifiers
closeReward = -100000
defaultReward = 100
moveRewardMultiplier = -1

#returns randomly true or false
def coinFlip():
	return random.choice(True, False)

class State:
	deltaPosition = (0,0)

	actions = dict() #map action in Action to (q-value,count)

	index = -1 #index of this State relative to States

	value = 0 #max q-value for this state
	valueAction = None #action corresponding to the value

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
		act = (int(action[0]),int(action[1]))
		if(act not in self.actions):
			self.actions[act] = (0,0)
		if(act not in ActionsSet):
			ActionsSet.add(act)
			ActionsArray.append(act)
		if(qval != self.actions[act][0]):
			self.cachedEQVInvalid = True
		self.actions[act] = (qval, self.actions[act][1]+1)
		if(qval > self.value or self.valueAction is None):
			self.value = qval
			self.valueAction = act

	#Generates a new action based on its previous actions weighted by their q-values
	def getWeightedAction(self):
		global maxSize
		if(not self.cachedEQVInvalid):
			return self.cachedExpectedQV
		count = 2
		lowest = 0
		ev = (0, 0)
		for a in self.actions:
			if( self.actions[a][0] < lowest):
				lowest = -self.actions[a][0]
		for a in self.actions:
			count += lowest + self.actions[a][0]
		for a in self.actions:
			ev = (ev[0]+((-lowest + 1 + self.actions[a][0])/count)*a[0], ev[1]+((-lowest+ 1 + self.actions[a][0])/count)*a[1])
		ret = (int(ev[0]),int(ev[1]))
		self.cachedExpectedQV = ret
		self.cachedEQVInvalid = False
		return ret
	
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
		global testMode
		#In normal q-learning, exploration happens randomly.
		#Since there are so many states with a certain select states of interest, we instead want
		#exploration to occur based on the expected value of the current state.
		#Essentially, if the AI sees itself in a bad state, prompt it to explore.
		#As such, exploration occurs as follows:
		#If random < epsilon, explore (default q-learning exploration)
		#Elif { get expected state; if value(expected) > value(current), explore}
		p = self.getWeightedAction()
		s = getNextState(self, p)
		ev = self.getExpectedVal()
		explore = False
		enablePredictiveExplore = True #set to true to predict when to explore
		if(random.random() < epsilon): #when exploring, we want to create a random weighted action
			print("EXPLORING RANDOMLY")
			explore = True
		elif(enablePredictiveExplore and s.getExpectedVal() > ev):
			print("EXPLORING BECAUSE BETTER EV")
			explore=True
		if(explore and not testMode):
			#if the expected value from the qvalues is bad, instead generate a completely random position
			
			if(ev < badValueThreshold or random.random() < superEpsilon):
				print("RANDOM EXPLORATION")
				return (random.randint(-maxSize[0], maxSize[0]), random.randint(-maxSize[1],maxSize[1]))
			else:
				
				print("PREDICTIVE EXPLORATION")
				return p
			
		else:
			#choose the ideal action. This always occurs in test mode
			if self.valueAction is None:
				#print("STICKING WITH DEFAULT ACTION:", (0,0))
				return (0,0)
			#print("STICKING WITH BEST ACTION:", self.valueAction)
			return self.valueAction

#gets the current state, makes a new state if it hasn't encountered said state before
def getCurrentState(tkRoot):
		mousePosition = pyautogui.position()
		windowPosition = (tkRoot.winfo_x(), tkRoot.winfo_y())
		dp = (windowPosition[0]-mousePosition[0],windowPosition[1]-mousePosition[1])
		deltaPosition = (math.copysign(1,dp[0])*max(abs(dp[0]), maxSize[0]), math.copysign(1,dp[1])*max(abs(dp[1]), maxSize[1]))
		if(deltaPosition not in stateMap.keys()):
			s = State(len(States), deltaPosition)
			States.append(s)
			stateMap[deltaPosition] = s
		return stateMap[deltaPosition]

#obtains or creates a state object based on a current state and action.

def getNextState(state, action):
	x = state.deltaPosition[0]+action[0]
	y = state.deltaPosition[1]+action[1]
	return getNextStateFromPos((x,y))

def getNextStateFromPos(pos):
	global maxSize
	x = int(pos[0])
	y = int(pos[1])
	deltaPosition = (math.copysign(1,x)*max(abs(x), maxSize[0]), math.copysign(1,y)*max(abs(y), maxSize[1]))
	if(deltaPosition not in stateMap.keys()):
		s = State(len(States), deltaPosition)
		States.append(s)
		stateMap[deltaPosition] = s
	return stateMap[deltaPosition]

#Generates a reward based on the previous state, next state, and action taken.
#See top for how reward works.
def generateReward(action, windowClosed = False):
	global closeReward
	global defaultReward
	global moveRewardMultiplier
	if(windowClosed):
		return closeReward
	if(action != (0,0)):
		return math.sqrt(action[0]*action[0]+action[1]*action[1])*moveRewardMultiplier
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
#returns true/false depending on success
def actOnAction(action):
	try:
		pyautogui.moveRel(action[0], action[1])
		if(action != (0,0)):
			print("MOVING MOUSE", action)
	except:
		#if mouse position invalid, move instead to the window position, top left corner
		print("INVALID MOUSE POSITION", action, "ATTEMPT MOVE TO", (root.winfo_x(), root.winfo_y()))
		pyautogui.FAILSAFE = False
		pyautogui.moveTo(root.winfo_x(), root.winfo_y())
		pyautogui.FAILSAFE = True


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
			i = int(s)
			v = j["states"][s] #contains x, y, a. a contains q,c
			pos = (v["x"], v["y"])
			m = dict()
			for a in v["a"]:
				k = v["a"][a]
				ind = int(a)
				count = int(k["c"])
				qv = int(k["q"])
				m[ActionsArray[ind]] = (qv,count)

			nv = State(i, pos, m)
			States.insert(i, nv)
			stateMap[pos] = nv
		
		file.close()
	except Exception as e:
		print("NEW/INVALID FILE, USING NEW DATA", e)
	

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

#Penalize AI without closing down
def penalize():
	print("ADD PENALTY FOR WINDOW CLOSE")
	s = getCurrentState(root)
	qLearnUpdate(s, (0,0), s, generateReward((0,0), True))

#Responsible for detecting the "losing condition" that the window has closed
def on_closing():
	global keepRunning
	global lock
	with lock:
		keepRunning = False
		#PersistentAIThread.join()
		print("WINDOW CLOSED, SAVING/CLEANING UP")
		if(not testMode):
			print("SAVING JSON DATA")
			
			if(not forceClose):
				
				#run one more iteration of q-learning for final result
				penalize()
			else:
				print("FORCE QUIT, TRAINING RESULT UNAFFECTED")

			serialize()
			print("FILE SAVED")
			
		else:
			print("TEST MODE, NOTHING TO SAVE")
		
		if(forceClose or not autoMode):
			os._exit(1)

#performs the running of the AI
def run():
	global epsilon
	global superEpsilon
	global testMode
	print("RUN")
	s = getCurrentState(root)
	runLocal = True
	with lock:
		runLocal = keepRunning
	while(runLocal):
		#update states list
		with lock:
			prevS = s
			mousePosition = pyautogui.position()
			windowPosition = (root.winfo_x(), root.winfo_y())
			if(mousePosition[0] > windowPosition[0] - maxSize[0] and mousePosition[0] < windowPosition[0] + maxSize[0] and \
				mousePosition[1] > windowPosition[1] - maxSize[1] and mousePosition[1] < windowPosition[1] + maxSize[1]): #if it's none, ignore as mouse is too far from window
				a = s.chooseAction(epsilon, superEpsilon)
				actOnAction(a)
				s = getCurrentState(root)
				if(not testMode):
					#update based on resulting action
					qLearnUpdate(prevS, a, s, generateReward(a))
			else:
				s = getCurrentState(root)
			runLocal = keepRunning
		time.sleep(delays)

#performs running of the bot which attempts to close the window.
#This AI simply increments the mouse to the top-right of the window and clicks whenever it's within -10x-10 of the corner.
#Upon successful click, it will then randomize the mouse somewhere in the allowed window range.
def run_bot():
	global root
	print("RUN AUTO BOT")
	runLocal = True
	resetPos = True
	autoThreshold = (30,20) #must be within the range of the x button on the window
	autoIncrementRange = (10,100) #num of pixels to move mouse each time
	attempts = 20 #attempt to reach the x button this many times before forcing a reset
	counter = 0
	with lock:
		runLocal = keepRunning
	while(runLocal):
		mousePosition = pyautogui.position()
		windowPosition = (root.winfo_x()+root.winfo_width()-autoThreshold[0], root.winfo_y()+autoThreshold[1])
		deltaPosition = (windowPosition[0]-mousePosition[0],windowPosition[1]-mousePosition[1])
		
		#move mouse reset
		if(resetPos or counter >= attempts):
			try:
				v = (random.randint(windowPosition[0] - maxSize[0], windowPosition[0] + maxSize[0]), random.randint(windowPosition[1] - maxSize[1], windowPosition[1] + maxSize[1]))
				print("AUTOBOT RESET TO",v)
				pyautogui.moveTo(v[0],v[1])
				counter = 0
				resetPos = False
			except:
				pass
		#attempt click
		elif(mousePosition[0] > windowPosition[0] - autoThreshold[0]/2 and mousePosition[1] < windowPosition[1] + autoThreshold[1]/2 \
			 and mousePosition[0] < windowPosition[0] + autoThreshold[0]/2 and mousePosition[1] > windowPosition[1] - autoThreshold[1]/2):
			print("AUTOBOT CLICKING")
			#pyautogui.click()
			penalize()
			resetPos = True
		else:
			len = math.sqrt(deltaPosition[0]*deltaPosition[0] + deltaPosition[1]*deltaPosition[1])
			norm = (0,0)
			if(len != 0):
				norm = (deltaPosition[0] / len, deltaPosition[1] / len)
				inc = random.randrange(autoIncrementRange[0], autoIncrementRange[1])
				val = norm[0]*inc, norm[1]*inc
				try:
					pyautogui.moveRel(val[0],val[1])
					counter += 1
				except:
					resetPos = True

		with lock:
			runLocal = keepRunning
		time.sleep(delays)

def forceTerminate():
	global forceClose
	forceClose = True
	on_closing()

#Thread that solely checks if the escape key is pressed.
def run_key_check():
	print("KEY CHECK ACTIVE")
	runLocal = True
	with lock:
		runLocal = keepRunning
	while(runLocal):
		if keyboard.is_pressed('escape'):
			print("Escape pressed")
			forceTerminate()
		with lock:
			runLocal = keepRunning


if __name__ == "__main__":
	deseralize()

	PersistentAIThread = threading.Thread(None, run)
	autoBotThread = threading.Thread(None, run_bot)
	terminateThread = threading.Thread(None, run_key_check)

	B = tk.Button(root, text="Force Quit (training ignored)", command = forceTerminate)
	B.place(x=0,y=0)
	root.geometry("200x200+{x}+{y}".format(x=maxSize[0]+10,y=maxSize[1]+10))
	root.protocol("WM_DELETE_WINDOW", on_closing)
	PersistentAIThread.start()
	if(autoMode):
		autoBotThread.start()
	terminateThread.start()
	root.mainloop()
