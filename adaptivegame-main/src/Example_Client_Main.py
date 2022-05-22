#encoding: utf-8

import time

from sqlalchemy import true
from Client import SocketClient
import json
import numpy as np
import tensorflow as tf
import tensorflow.python.keras as keras
import tensorflow.python.keras.layers as layers
import tensorflow.python.keras.initializers as initializers

import matplotlib.pyplot as plt


def get_reward(bandit: float) -> tf.Tensor:
        reward = tf.random.normal([1], mean=bandit, stddev=1, dtype=tf.dtypes.float32)

        return reward

def construct_q_network(state_dim: int, action_dim: int) -> keras.Model:
    inputs = layers.Input(shape=(state_dim,))  # input dimension
    hidden1 = layers.Dense(
        50, activation="relu", kernel_initializer=initializers.he_normal()
    )(inputs)
    hidden2 = layers.Dense(
        50, activation="relu", kernel_initializer=initializers.he_normal()
    )(hidden1)
    q_values = layers.Dense(
        action_dim, kernel_initializer=initializers.Zeros(), activation="linear"
    )(hidden2)

    deep_q_network = keras.Model(inputs=inputs, outputs=[q_values])

    return deep_q_network

def mean_squared_error_loss(q_value: tf.Tensor, reward: tf.Tensor) -> tf.Tensor:
    loss = 0.5 * (q_value - reward) ** 2

    return loss

plt.ion()
fig, ax = plt.subplots()
plt.xlabel("Games")
plt.ylabel("Sizes")
plt.show()

def plot(sizes: np.array) -> None:
    width = 0.4
    x = np.arange(len(sizes))
    
    ax.bar(x, sizes, width, label="Sizes", color="b")

    fig.canvas.draw()
    fig.canvas.flush_events()

    return

# NaiveHunter stratégia implementációja távoli eléréshez.
class MyDeepQStrategy:

    

    def __init__(self):
        # Dinamikus viselkedéshez szükséges változók definíciója
        self.oldpos = None
        self.oldcounter = 0

        self.q_network = construct_q_network(81, 9)
        self.opt = tf.keras.optimizers.Adam(learning_rate=0.01)
        self.last_size = 5
        self.exploration_rate = 0.1
        self.sizeGainCtr = 0
        self.sizes = []
        self.running = True
        
    

    # Egyéb függvények...
    def getRandomAction(self):
        actdict = {0: "0", 1: "+", 2: "-"}
        r = np.random.randint(0, 3, 2)
        action = ""
        for act in r:
            action += actdict[act]

        return action

    # Az egyetlen kötelező elem: A játékmestertől jövő információt feldolgozó és választ elküldő függvény
    def processObservation(self, fulljson, sendData):
        """
        :param fulljson: A játékmestertől érkező JSON dict-be konvertálva.
        Két kötelező kulccsal: 'type' (leaderBoard, readyToStart, started, gameData, serverClose) és 'payload' (az üzenet adatrésze).
        'leaderBoard' type a játék végét jelzi, a payload tartalma {'ticks': a játék hossza tickekben, 'players':[{'name': jáétékosnév, 'active': él-e a játékos?, 'maxSize': a legnagyobb elért méret a játék során},...]}
        'readyToStart' type esetén a szerver az indító üzenetre vár esetén, a payload üres (None)
        'started' type esetén a játék elindul, tickLength-enként kiküldés és akciófogadás várható payload {'tickLength': egy tick hossza }
        'gameData' type esetén az üzenet a játékos által elérhető információkat küldi, a payload:
                                    {"pos": abszolút pozíció a térképen, "tick": az aktuális tick sorszáma, "active": a saját életünk állapota,
                                    "size": saját méret, "vision": [{"relative_coord": az adott megfigyelt mező relatív koordinátája,
                                                                    "value": az adott megfigyelt mező értéke (0-3,9),
                                                                    "player": None, ha nincs aktív játékos, vagy
                                                                            {name: a mezőn álló játékos neve, size: a mezőn álló játékos mérete}},...] }
        'serverClose' type esetén a játékmester szabályos, vagy hiba okozta bezáródásáról értesülünk, a payload üres (None)
        :param sendData: A kliens adatküldő függvénye, JSON formátumú str bemenetet vár, melyet a játékmester felé továbbít.
        Az elküldött adat struktúrája {"command": Parancs típusa, "name": A küldő azonosítója, "payload": az üzenet adatrésze}
        Elérhető parancsok:
        'SetName' A kliens felregisztrálja a saját nevét a szervernek, enélkül a nevünkhöz tartozó üzenetek nem térnek vissza.
                 Tiltott nevek: a configban megadott játékmester név és az 'all'.
        'SetAction' Ebben az esetben a payload az akció string, amely két karaktert tartalmaz az X és az Y koordináták (matematikai mátrix indexelés) menti elmozdulásra.
                a karakterek értékei '0': helybenmaradás az adott tengely mentén, '+' pozitív irányú lépés, '-' negatív irányú lépés lehet. Amennyiben egy tick ideje alatt
                nem külünk értéket az alapértelmezett '00' kerül végrehajtásra.
        'GameControl' üzeneteket csak a Config.py-ban megadott játékmester névvel lehet küldeni, ezek a játékmenetet befolyásoló üzenetek.
                A payload az üzenet típusát (type), valamint az ahhoz tartozó 'data' adatokat kell, hogy tartalmazza.
                    'start' type elindítja a játékot egy "readyToStart" üzenetet küldött játék esetén, 'data' mezője üres (None)
                    'reset' type egy játék után várakozó 'leaderBoard'-ot küldött játékot állít alaphelyzetbe. A 'data' mező
                            {'mapPath':None, vagy elérési útvonal, 'updateMapPath': None, vagy elérési útvonal} formátumú, ahol None
                            esetén az előző pálya és növekedési map kerül megtartásra, míg elérési útvonal megadása esetén új pálya kerül betöltésre annak megfelelően.
                    'interrupt' type esetén a 'data' mező üres (None), ez megszakítja a szerver futását és szabályosan leállítja azt.
        :return:
        """

        # Játék rendezéssel kapcsolatos üzenetek lekezelése
        if fulljson["type"] == "leaderBoard":
            print("Game finished after",fulljson["payload"]["ticks"],"ticks!")
            print("Leaderboard:")
            for score in fulljson["payload"]["players"]:
                print(score["name"],score["active"], score["maxSize"])
                if score["name"] == "RemotePlayer" and score["active"] == True:
                    self.sizes.append(score["maxSize"])

            
            sendData(json.dumps({"command": "GameControl", "name": "master",
                                 "payload": {"type": "reset", "data": {"mapPath": None, "updateMapPath": None}}}))

        if fulljson["type"] == "readyToStart":
            print("Game is ready, starting in 5")
            sendData(json.dumps({"command": "GameControl", "name": "master",
                                 "payload": {"type": "start", "data": None}}))

        if fulljson["type"] == "started":
            print("Startup message from server.")
            print("Ticks interval is:",fulljson["payload"]["tickLength"])


        # Akció előállítása bemenetek alapján (egyezik a NaiveHunterBot-okéval)
        elif fulljson["type"] == "gameData":
            jsonData = fulljson["payload"]
            if "pos" in jsonData.keys() and "tick" in jsonData.keys() and "active" in jsonData.keys() and "size" in jsonData.keys() and "vision" in jsonData.keys():
                if self.oldpos is not None:
                    if tuple(self.oldpos) == tuple(jsonData["pos"]):
                        self.oldcounter += 1
                    else:
                        self.oldcounter = 0
                if jsonData["active"]:
                    self.oldpos = jsonData["pos"].copy()

                vals = []
                for field in jsonData["vision"]:
                    if field["player"] is not None:
                        if tuple(field["relative_coord"]) == (0, 0):
                            if 0 < field["value"] <= 3:
                                vals.append(field["value"])
                            elif field["value"] == 9:
                                vals.append(-1)
                            else:
                                vals.append(0)
                        elif field["player"]["size"] * 1.1 < jsonData["size"]:
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

                values = tf.constant([vals])
                
                with tf.GradientTape() as tape:
                    q_values = self.q_network(values)

                    epsilon = np.random.rand()
                    if epsilon <= self.exploration_rate:
                        action = np.random.choice(9)
                    else:
                        action = np.argmax(q_values)

                    self.size = jsonData["size"]
                    sizeDiff = self.size - self.last_size
                    if sizeDiff > 0:
                        self.sizeGainCtr = 0
                    else:
                        self.sizeGainCtr = self.sizeGainCtr + 1
                    self.last_size = self.size
                    reward = [sizeDiff-(self.sizeGainCtr/10)]

                    q_value = q_values[0, action]

                    loss_value = mean_squared_error_loss(q_value, reward)
                    grads = tape.gradient(loss_value[0], self.q_network.trainable_variables)

                    self.opt.apply_gradients(zip(grads, self.q_network.trainable_variables))

                    if action == 1:
                        actstring = "++"
                    elif action == 2:
                        actstring = "+0"
                    elif action == 3:
                        actstring = "+-"
                    elif action == 4:
                        actstring = "0+"
                    elif action == 5:
                        actstring = "00"
                    elif action == 6:
                        actstring = "0-"
                    elif action == 7:
                        actstring = "-+"
                    elif action == 8:
                        actstring = "-0"
                    elif action == 0:
                        actstring = "--"
                    else:
                        actstring = "--"

                

                    # Akció JSON előállítása és elküldése
                    sendData(json.dumps({"command": "SetAction", "name": "RemotePlayer", "payload": actstring}))


                



if __name__=="__main__":
    # Példányosított stratégia objektum
    hunter = MyDeepQStrategy()

    # Socket kliens, melynek a szerver címét kell megadni (IP, port), illetve a callback függvényt, melynek szignatúrája a fenti
    # callback(fulljson, sendData)
    client = SocketClient("localhost", 42069, hunter.processObservation)

    # Kliens indítása
    client.start()
    # Kis szünet, hogy a kapcsolat felépülhessen, a start nem blockol, a kliens külső szálon fut
    time.sleep(0.1)
    # Regisztráció a megfelelő névvel
    client.sendData(json.dumps({"command": "SetName", "name": "RemotePlayer", "payload": None}))

    # Nincs blokkoló hívás, a főszál várakozó állapotba kerül, itt végrehajthatók egyéb műveletek a kliens automata működésétől függetlenül.
    while hunter.running == True:
        plot(hunter.sizes)
        time.sleep(1.5)