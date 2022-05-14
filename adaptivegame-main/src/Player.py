import numpy as np
import json
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow.keras.initializers as initializers


def construct_q_network(state_dim: int, action_dim: int) -> keras.Model:
    inputs = layers.Input(shape=(state_dim,))  # input dimension
    hidden1 = layers.Dense(
        24, activation="relu", kernel_initializer=initializers.he_normal()
    )(inputs)
    q_values = layers.Dense(
        action_dim, kernel_initializer=initializers.Identity() , activation="linear",
    )(hidden1)

    deep_q_network = keras.Model(inputs=inputs, outputs=[q_values])

    return deep_q_network

def mean_squared_error_loss(o_q_value: tf.Tensor, q_value: tf.Tensor) -> tf.Tensor:
    loss = (o_q_value - q_value) ** 2

    return loss

class RemotePlayerStrategy:
    def __init__(self, **kwargs):
        self.nextAction = "0"
        self.sendData = kwargs["sender"]
        self.getData = kwargs["getter"]
        self.name = kwargs["name"]

    def setObservations(self, ownObject, fieldDict):
        self.nextAction = "0"
        self.sendData(json.dumps({"type":"gameData", "payload":fieldDict}), ownObject.name)

    def getNextAction(self):
        newaction = self.getData(self.name)
        if newaction is None:
            return self.nextAction
        else:
            return newaction

    def reset(self):
        self.nextAction = "0"
        data = "something"
        while data is not None:
            data = self.getData(self.name)

class DummyStrategy:
    def __init__(self, **kwargs):
        self.nextAction = "0"

    def setObservations(self, ownObject, fieldDict):
        ownObject.active=False

    def getNextAction(self):
        return "0"

    def reset(self):
        pass

class RandBotStrategy:
    def __init__(self, **kwargs):
        self.nextAction = 0

    def setObservations(self, ownObject, fieldDict):
        pass

    def getNextAction(self):
        actdict = {0: "0", 1: "+", 2: "-"}
        r = np.random.randint(0, 3, 2)
        action = ""
        for act in r:
            action += actdict[act]

        return action

    def reset(self):
        self.nextAction = "0"

class NaiveStrategy:
    def __init__(self, **kwargs):
        self.nextAction = "0"
        self.oldpos = None
        self.oldcounter = 0

    def getRandomAction(self):
        actdict = {0: "0", 1: "+", 2: "-"}
        r = np.random.randint(0, 3, 2)
        action = ""
        for act in r:
            action += actdict[act]

        return action

    def setObservations(self, ownObject, fieldDict):
        if self.oldpos is not None:
            if tuple(self.oldpos) == tuple(ownObject.pos):
                self.oldcounter += 1

        self.oldpos = ownObject.pos.copy()

        values = np.array([field["value"] for field in fieldDict["vision"]])
        values[values > 3] = 0
        values[values < 0] = 0
        if np.max(values) == 0 or self.oldcounter >= 3:
            self.nextAction = self.getRandomAction()
            self.oldcounter = 0
        else:
            idx = np.argmax(values)
            actstring = ""
            for i in range(2):
                if fieldDict["vision"][idx]["relative_coord"][i] == 0:
                    actstring += "0"
                elif fieldDict["vision"][idx]["relative_coord"][i] > 0:
                    actstring += "+"
                elif fieldDict["vision"][idx]["relative_coord"][i] < 0:
                    actstring += "-"

            self.nextAction = actstring

    def getNextAction(self):
        return self.nextAction

    def reset(self):
        self.nextAction = "0"

class NaiveHunterStrategy:
    def __init__(self, **kwargs):
        self.nextAction = "0"
        self.oldpos = None
        self.oldcounter = 0

    def getRandomAction(self):
        actdict = {0: "0", 1: "+", 2: "-"}
        r = np.random.randint(0, 3, 2)
        action = ""
        for act in r:
            action += actdict[act]

        return action

    def setObservations(self, ownObject, fieldDict):
        if self.oldpos is not None:
            if tuple(self.oldpos) == tuple(ownObject.pos):
                self.oldcounter += 1
            else:
                self.oldcounter = 0
        if ownObject.active:
            self.oldpos = ownObject.pos.copy()

        vals = []
        for field in fieldDict["vision"]:
            if field["player"] is not None:
                if tuple(field["relative_coord"]) == (0, 0):
                    if 0 < field["value"] <= 3:
                        vals.append(field["value"])
                    elif field["value"] == 9:
                        vals.append(-1)
                    else:
                        vals.append(0)
                elif field["player"]["size"] * 1.1 < ownObject.size:
                    vals.append(field["player"]["size"])
                else:
                    vals.append(-1)
            else:
                if 0 < field["value"] <= 3:
                    vals.append(field["value"])
                elif field["value"] == 9:
                    vals.append(-1)
                else:
                    vals.append(0)

        values = np.array(vals)
        # print(values, fieldDict["vision"][np.argmax(values)]["relative_coord"], values.max())
        if np.max(values) <= 0 or self.oldcounter >= 3:
            self.nextAction = self.getRandomAction()
            self.oldcounter = 0
        else:
            idx = np.argmax(values)
            actstring = ""
            for i in range(2):
                if fieldDict["vision"][idx]["relative_coord"][i] == 0:
                    actstring += "0"
                elif fieldDict["vision"][idx]["relative_coord"][i] > 0:
                    actstring += "+"
                elif fieldDict["vision"][idx]["relative_coord"][i] < 0:
                    actstring += "-"

            self.nextAction = actstring

    def getNextAction(self):
        return self.nextAction

    def reset(self):
        self.nextAction = "0"

def getVisionSum(vision, startx, starty, endx, endy):
    sum = 0
    for x in range(startx, endx):
        for y in range(starty, endy):
            sum += vision[x][y]
    return sum

class MyDeepQStrategy:
    def __init__(self, **kwargs):
        self.nextAction = "0"
        self.oldpos = None
        self.oldcounter = 0

        self.q_network = construct_q_network(12, 8)
        self.opt = tf.keras.optimizers.Adam(learning_rate=0.1)
        self.last_size = 5
        self.exploration_rate = 0
        self.sizeGainCtr = 0
        self.sizes = []
        self.running = True
        self.prev_state = None
        self.gamma = 1
        self.reward = [0]
        self.prev_actions = [0,0,0,0,0,0,0,0,0,0]
        self.prev_q_value = 0
        self.starting = True
        self.q_network.trainable = True
        self.closest = [0,0,0,0,0,0,0,0]
        self.prevPos = []
        self.prev_action = 0
        self.reward = 0

        
    def save_q_network(self):
        tf.keras.models.save_model(self.q_network, "D:/BME/MSc/1. felev/Adaptív rendszerek modellezése/Projektfeladat/adaptivegame-main/src/my_model")

    def getRandomAction(self):
        actdict = {0: "0", 1: "+", 2: "-"}
        r = np.random.randint(0, 3, 2)
        action = ""
        for act in r:
            action += actdict[act]

        return action

    def checkClosest(self, field, value):
        #if value == 0:
        #    value = 0.1
        if field["relative_coord"][0] == -1 and field["relative_coord"][1] == -1:
            self.closest[0] = value
        elif field["relative_coord"][0] == -1 and field["relative_coord"][1] == 0:
            self.closest[1] = value
        elif field["relative_coord"][0] == -1 and field["relative_coord"][1] == 1:
            self.closest[2] = value
        elif field["relative_coord"][0] == 0 and field["relative_coord"][1] == -1:
            self.closest[3] = value
        elif field["relative_coord"][0] == 0 and field["relative_coord"][1] == 1:
            self.closest[4] = value
        elif field["relative_coord"][0] == 1 and field["relative_coord"][1] == -1:
            self.closest[5] = value
        elif field["relative_coord"][0] == 1 and field["relative_coord"][1] == 0:
            self.closest[6] = value
        elif field["relative_coord"][0] == 1 and field["relative_coord"][1] == 1:
            self.closest[7] = value

    def setObservations(self, ownObject, fieldDict):
        if self.oldpos is not None:
            if tuple(self.oldpos) == tuple(ownObject.pos):
                self.oldcounter += 1
            else:
                self.oldcounter = 0
        if ownObject.active:
            self.oldpos = ownObject.pos.copy()

        vals = []
        visionVals2D = [[0 for j in range(9)] for i in range(9)]
        xpp = 0
        xmm = 0
        ypp = 0
        ymm = 0

        

        for field in fieldDict["vision"]:
            if field["player"] is not None:
                if tuple(field["relative_coord"]) == (0, 0):
                    if 0 < field["value"] <= 3:
                        vals.append(field["value"])
                        #visionVals2D[field["relative_coord"][0]][field["relative_coord"][1]] = field["value"]
                    elif field["value"] == 9:
                        vals.append(-1)
                        #visionVals2D[field["relative_coord"][0]][field["relative_coord"][1]] = -1
                    else:
                        vals.append(0)
                        #visionVals2D[field["relative_coord"][0]][field["relative_coord"][1]] = 0
                elif field["player"]["size"] * 1.1 < ownObject.size:
                    self.checkClosest(field, field["player"]["size"])
                    if field["relative_coord"][0] > 0:
                        xpp += field["player"]["size"] / (abs(field["relative_coord"][0]))
                    elif field["relative_coord"][0] < 0:
                        xmm += field["player"]["size"] / (abs(field["relative_coord"][0]))
                    
                    if field["relative_coord"][1] > 0:
                        ypp += field["player"]["size"] / (abs(field["relative_coord"][1]))
                    elif field["relative_coord"][1] < 0:
                        ymm += field["player"]["size"] / (abs(field["relative_coord"][1]))
                else:
                    self.checkClosest(field, -100)
                    if field["relative_coord"][0] > 0:
                        xpp += -100 / (abs(field["relative_coord"][0]))
                    elif field["relative_coord"][0] < 0:
                        xmm += -100 / (abs(field["relative_coord"][0]))
                    
                    if field["relative_coord"][1] > 0:
                        ypp += -100 / (abs(field["relative_coord"][1]))
                    elif field["relative_coord"][1] < 0:
                        ymm += -100 / (abs(field["relative_coord"][1]))
            else:
                if ownObject.pos[0] + field["relative_coord"][0] < 0 or ownObject.pos[0] + field["relative_coord"][0] > 39:
                    continue
                if ownObject.pos[1] + field["relative_coord"][1] < 0 or ownObject.pos[1] + field["relative_coord"][1] > 39:
                    continue

                if 0 < field["value"] <= 3:
                    self.checkClosest(field, field["value"])
                    if field["relative_coord"][0] > 0:
                        xpp += field["value"] / (abs(field["relative_coord"][0]))
                    elif field["relative_coord"][0] < 0:
                        xmm += field["value"] / (abs(field["relative_coord"][0]))
                    
                    if field["relative_coord"][1] > 0:
                        ypp += field["value"] / (abs(field["relative_coord"][1]))
                    elif field["relative_coord"][1] < 0:
                        ymm += field["value"] / (abs(field["relative_coord"][1]))
                elif field["value"] == 9:
                    self.checkClosest(field, -1)
                    if field["relative_coord"][0] > 0:
                        xpp += -0.3 / (abs(field["relative_coord"][0])**2)
                    elif field["relative_coord"][0] < 0:
                        xmm += -0.3 / (abs(field["relative_coord"][0])**2)
                    
                    if field["relative_coord"][1] > 0:
                        ypp += -0.3 / (abs(field["relative_coord"][1])**2)
                    elif field["relative_coord"][1] < 0:
                        ymm += -0.3 / (abs(field["relative_coord"][1])**2)
                else:
                    self.checkClosest(field, 0)
                    #if field["relative_coord"][0] > 0:
                    #    xpp += 0 / (abs(field["relative_coord"][0])**2)
                    #elif field["relative_coord"][0] < 0:
                    #    xmm += 0 / (abs(field["relative_coord"][0])**2)
                    
                    #if field["relative_coord"][1] > 0:
                        #ypp += 0 / (abs(field["relative_coord"][1])**2)
                    #elif field["relative_coord"][1] < 0:
                        #ymm += 0 / (abs(field["relative_coord"][1])**2)
        
        #closeby_multiplier = 10

        #xmyp = getVisionSum(visionVals2D, 0, 3, 6, 9)/closeby_multiplier + visionVals2D[3][5]
        #xpyp = getVisionSum(visionVals2D, 6, 9, 6, 9)/closeby_multiplier + visionVals2D[5][5]
        #xpym = getVisionSum(visionVals2D, 6, 9, 0, 3)/closeby_multiplier + visionVals2D[5][3]
        #xmym = getVisionSum(visionVals2D, 0, 3, 0, 3)/closeby_multiplier + visionVals2D[3][3]
        
        #xpp = getVisionSum(visionVals2D, 6, 9, 3, 6)/closeby_multiplier + visionVals2D[6][4]
        #xmm = getVisionSum(visionVals2D, 0, 3, 3, 6)/closeby_multiplier + visionVals2D[2][4]
        #ypp = getVisionSum(visionVals2D, 3, 6, 0, 3)/closeby_multiplier + visionVals2D[4][6]
        #ymm = getVisionSum(visionVals2D, 3, 6, 0, 3)/closeby_multiplier + visionVals2D[4][2]

        state = tf.constant([[self.closest[0], self.closest[1], self.closest[2], self.closest[3], self.closest[4], self.closest[5], self.closest[6], self.closest[7], xmm,xpp,ymm,ypp]])
        #state.trainable = True

        #if not ownObject.active:
        #    return

        #if self.starting:
        #    self.prev_state = tf.identity(state)

        with tf.GradientTape() as tape:
            #tape.watch(state)
            #next_q_values = self.q_network(self.prev_state)
            #next_action = np.argmax(next_q_values[0])
            #next_q_value = next_q_values[0, next_action]

            #observed_q_value = self.reward + (self.gamma * next_q_value)
            #loss_value = mean_squared_error_loss(observed_q_value, self.prev_q_value)
            #self.q_network.loss = loss_value

            if not self.starting:
                #grads = tape.gradient(loss_value, self.q_network.trainable_variables)
                #self.opt.apply_gradients(zip(grads, self.q_network.trainable_variables))
                self.starting = True
            else:
                self.starting = False

            
            #self.prev_state = tf.identity(state)

            q_values = self.q_network(state)

            epsilon = np.random.rand()
            if epsilon <= self.exploration_rate:
                action = np.random.choice(8)
            else:
                action = np.argmax(q_values)

            #self.size = ownObject.size
            #sizeDiff = self.size - self.last_size
            #if sizeDiff > 0:
            #    self.sizeGainCtr = 0
            #else:
            #    self.sizeGainCtr = self.sizeGainCtr + 1
            #self.last_size = self.size

            #self.prevPos.append(ownObject.pos)
            #if len(self.prevPos) > 5:
            #    self.prevPos.pop(0)

            rel = 0
            if action == 1:
                actstring = "++"
                relX = 1
                relY = 1
                rel = 7
            elif action == 2:
                actstring = "+0"
                relX = 1
                relY = 0
                rel = 6
            elif action == 3:
                actstring = "+-"
                relX = 1
                relY = -1
                rel = 5
            elif action == 4:
                actstring = "0+"
                relX = 0
                relY = 1
                rel = 4
            #elif action == 5:
            #    actstring = "00"
            #    relX = 0
            #    relY = 0
            #    self.reward = 0
            elif action == 5:
                actstring = "0-"
                relX = 0
                relY = -1
                rel = 3
            elif action == 6:
                actstring = "-+"
                relX = -1
                relY = 1
                rel = 2
            elif action == 7:
                actstring = "-0"
                relX = -1
                relY = 0
                rel = 1
            elif action == 8:
                actstring = "--"
                relX = -1
                relY = -1
                rel = 0
            else:
                actstring = "--"
                relX = -1
                relY = -1
                rel = 0

            #self.prev_actions.append(relX)
            #self.prev_actions.append(relY)

            #if(len(self.prev_actions) > 10):
            #    self.prev_actions.pop(0)
            #    self.prev_actions.pop(1)

            #newPos = [ownObject.pos[0] + relX, ownObject.pos[1] + relY]

            

            #if sizeDiff > 0:
            #    self.reward = sizeDiff
            #elif sizeDiff < 0:
            #    self.reward = -10
            #elif self.oldcounter > 0:
            #    self.reward = -1
            #else:
            #    self.reward = 0

            self.reward = self.closest[rel]
            #if self.reward == 0:
            #    self.reward = 0.1
            
            #for pPos in self.prevPos:
            #    if tuple(pPos) == tuple(newPos) and self.reward == 0:
            #        self.reward = -0.5

            #if self.reward == 0:
            #    if action == 5:
            #        self.reward = 0
            #    elif action == 2:
            #        self.reward = xpp / 100
            #    elif action == 8:
            #        self.reward = xmm / 100
            #    elif action == 4:
            #        self.reward = ypp / 100
            #    elif action == 6:
            #        self.reward = ymm / 100
            #    elif action == 1:
            #        self.reward = xpp*ypp / 10000
            #    elif action == 3:
            #        self.reward = xpp*ymm / 10000
            #    elif action == 7:
            #        self.reward = xmm*ypp / 10000
            #    else:
            #        self.reward = xmm*ymm / 10000
                
                
                
            
            #if self.oldcounter > 2:
            #    self.reward = -1

            if ownObject.active:
                q_value = q_values[0,action]
                loss_value = (self.reward - q_value) ** 2
                grads = tape.gradient(loss_value, self.q_network.trainable_variables)
                self.opt.apply_gradients(zip(grads, self.q_network.trainable_variables))

            #self.reward = [self.reward]
            #self.prev_q_value = np.copy(q_values[0, action])
            #self.prev_action = action


            

            self.nextAction = actstring


    def getNextAction(self):
        return self.nextAction

    def reset(self):
        self.nextAction = "0"

class Player:
    strategies = {"randombot": RandBotStrategy, "naivebot": NaiveStrategy, "naivehunterbot": NaiveHunterStrategy,
                  "remoteplayer": RemotePlayerStrategy, "dummy":DummyStrategy, "mydeepq":MyDeepQStrategy}

    def __init__(self, name, playerType, startingSize, **kwargs):
        self.name = name
        self.playerType = playerType
        self.pos = np.zeros((2,))
        self.size = startingSize
        kwargs["name"] = name
        self.strategy = Player.strategies[playerType](**kwargs)
        self.active = True

    def die(self):
        self.active = False
        print(self.name + " died!")

    def reset(self):
        self.active = True

