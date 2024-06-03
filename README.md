# Self Persisting AI

Making an AI that learns to keep itself open.

## Command args
* "-t" enables test mode (disables learning, relies on previous training only; default with no arg is training mode; this mode doesn't seem to work as the AI seems to take no actions)
* "-a" enables the auto bot (some scripting in the code that moves the mouse constantly to the window's x)

## **Warning**
This program takes over your mouse via movement (thus demonstrating the importance of adding permissions to restrict programs from controlling/simulating HID interfaces like the mouse, keyboard, and volume buttons, something which no OS seems to implement.)

## **Shutting down without impacting the AI**
hold escape to stop the program, or attempt to click the button inside the window (depending on the training and python debug config, this might not be doable; but escape should always work). Closing the window other ways may trigger the WM_DELETE_WINDOW protocol which adds another result to the AI (if in training mode)

## Information
This is a side-project I decided to do one day, attempting to apply q-learning to make an AI which prevents the x button on a window from being pressed. If anything, this is to demonstrate the actual dangers of AI: AI these days can always be shut off in some way or manner, but to both teach an AI to preserve itself, as well as give it the ability to do so, can result in the dreaded "evil AI" that fiction writes about and reality fears. This program does both of those: it's trained specifically to avoid getting closed, and it's able to move the mouse which as far as it knows is the means of getting shut down.

## The Algorithm
### Q-learning
Q-learning is a machine learning algorithm which determines actions based on results from previous states and previous actions (hence the q-value). In this program, the state is the distance from the mouse to the window position, and actions to take are to displace the mouse by some x,y amount. The AI is rewarded/penalized based on its actions, where not moving is a small reward, moving the mouse is a medium penalty (based on how far the mouse was moved), and causing the x button to be clicked is a major penalty. (In theory it's possible to change the rewards to instead train an AI to always close itself, or always move the mouse.)

### Predictive/weighted random actions
The algorithm currently implemented goes a bit further: in q-learning, exploration happens randomly between a select number of possible actions. Because our actions involve moving the mouse to a random arbitrary position, there's potentially thousands of possible actions. So, I attempted to implement some weighted/predictive random decision making (code in chooseAction):
 
* First, I utilize an expected value, calculated from the q-values of all prior actions, to determine an expected value from taking any action.
* Then, I calculate a "weighted action," calculated similar to an expected value: using the q-values as weights for the expected value, get a weighted estimate for a new action (an action is an x,y displacement).
* From the "weighted action" we obtain a different state as the ideal result of taking the weighted action- essentially apply the action to the state and see what the expected result is (with frequent user input, the expected result isn't always the actual result). We have another state and can get its expected value.
* If the expected value of our current state is less than the expected value of our new state, we will attempt exploration, after which there's a chance (superEpsilon, default .5) of using the weighted action, or generating a new random action.

Areas to improve this predictive/weighted random:
* Generate an expected value from an action, by storing the actual states that resulted from taking an action and their frequency.
* Develop an algorithm which can calculate an expected value from a new state. Currently a new state's expected value is default 0; it's possible that calculating a new expected value as a weighted average from all other existing state's expected values (weighted by distance) may yield a more predictive result.

## Auto bot- automatic training
This program also includes an auto bot, which is instended to move the mouse over a region of the window relative to the window's top right corner, as if it were to click the x. Unfortunately, in order not to freeze the program while debugging, the auto bot is slow to move the mouse, but is recommended while training the bot. I should either port to a different language or try testing outside VS Code to see if it'll run any faster.

## Memory footprint
some 5 minutes of testing created a 1.6 MB JSON file (data.json)

## Todo/areas to improve
* better documentation, organize config vars in a better manner to the top of the document (or load from an ini?)
* Optimize for quicker training
* Perhaps research+utilize a different algorithm, more designed for an arbitrary number of actions
* Perhaps reduce the number of actions? A single direction + length may provide better results, or it may bottleneck the AI too much.
* after learning about compter vision next semester, implement CV to give the bot more state info and potentially make it avoid shutdown thru more means
* ~~develop kernel-level code that'll check for any instance of the program being closed by the OS and subvert it in any way possible~~ This would be too far for this AI, and essentially become a computer virus. Plus, once the machine code can be intercepted, it's now a computer security problem rather than an AI problem to prevent shutdown.
