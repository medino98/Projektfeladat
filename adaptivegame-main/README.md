<img src="/pics/GPK_BME_MOGI.png">

# AdaptIO

Az AdaptIO az "Adaptív rendszerek modellezése" tárgyhoz készült játék felület. A tárgy sikeres teljesítéséhez 
szükséges házi feladat egy agent elkészítése, mely elboldogul az AdaptIO játékban. A cél az életben maradás.

A feladat sikeres teljesítésének feltétele az agent kódjának beadása, illetve egy rövid prezentáció a szemeszter
végén, melyen az agent logikájának kialakítását és betanításának főbb lépéseit mutatják be a csapatok. Az agenteket
ez után természetesen egymással is megversenyeztetjük!

A feladat elkészítése során a tárgyban tanult genetikus, bakteriális vagy neurális háló alapú megoldásokat preferáljuk.
Nem tiltott tovább lépni se és a tárgyban esetlegesen nem érintett technikákat alkalmazni. A feladat elkészítése során
a környezet szabadon átalakítható, viszont a bemutatás egységesen a master branchen elérhető verziót fogjuk használni.

<img src="/pics/displayy.png" width="600">

## Telepítés

Repository letöltése vagy clonozása.

    git clone https://gitlab.com/microlab1/public-projects/adaptivegame

### Függőségek

A program **python 3.7** verzióval készült. <br>
A futtatáshoz az alábbi python packagekre lesz szükség:
- numpy
- pygame

### Indítás

`Main.py` futtatása: <br>
Elindítja a játékot. Minden szükséges paraméter a Config.py fileban található, indítás előtt
a paraméterek módosíthatóak.

`Example_Client_Main.py` futtatása: <br>
Csatlakozó kliens játékos. Ez a file ad mintát az elkészítendő játékos kódjához. A játékosok
saját gépről tudnak majd futni socket kapcsolaton keresztül bejelentkezve.

## Szabályok

A játékban minden agent egy kockányi mezőt foglal el. Végrehajtható akcióként egy iterációban 9 választási lehetősége van.
Vagy helyben marad vagy a 8 szomszédos mező valamelyikére mozog.

Az agentek rendelkeznek mérettel, mely a játék kezdetén egy alap paraméter (5). Az agentek mérete a játék során növelhető táplálkozással.
A játéktéren találhatók kaja mezők különböző intenzitással (1, 2, 3). Ha az agent kaját tartalmazó mezőre lép
a bekebelezés automatikusan megtörténik és az agent mérete a kaja méretével növekszik. A pályán továbbá találhatóak falak, melyek nem elmozdíthatóak, nem termelődik rajtük kaja
és rálépni se lehet. Az agentek átlátnak a falakon. A játokosok a következő látomezővel rendelkeznek:

<img src="/pics/latomezo.png" width="200"> <br>

Ha több játokos azonos időben ugyan arra a mezőre lépne:
- először ellenőrizzük, hogy a legnagyobb játékos meg tudja-e enni a második legnagyobbat.
- ha igen, mindenkit megeszik.
- ha nem a játékosok korábbi helyükön maradnak, mintha nem léptek volna.

A bekebelezés (egyik játékos megeszi a másikat) akkor jön létre, ha a kisebb játékos méretét felszorozzuk
a Config.py fileban található MIN_RATIO paraméterrel és még így is kisebb, mint a nagyobb játékos.
Minden más esetben a játékosok közti méretkülönbség túl kicsi, így csak lepattanak egymésról és korábbi helyükön maradnak.
 
### Pálya elemek:

<img src="/pics/map.png" width="200"> <br>

| Érték |  Jelentés   |   Szín |
|-------|:-----------:|-------:|
| 0     |  üres mező  | szürke |
| 1     |  kaja LOW   |   zöld |
| 2     | kaja MEDIUM |   zöld |
| 3     |  kaja HIGH  |   zöld |
| 9     |     fal     | fekete |

Előre generált pályák a maps mappában találhatóak, de további pályák is generálhatóak a feladat minél jobb
megoldása érdekében. Pálya generáláshoz hasznos lehet a **maps.xlsx** fájl. 

Javaslat pályageneráláshoz:
- Árdemes a `01_empty_ringet` másolni és csak a keretezett bal felső sarokba rajzolni, a többi tükröződik.
- Egy meglévő pálya lapján jobb klikk
- Áthelyezés vagy másolás
- Válasszuk a (végére) opciót
- Legyen másolat opció kiválasztása
- OK
- Módosítsuk a pályát
- Jelöljük ki a pálya elemit (Ctrl+C)
- Másoljuk egy txt fileba a maps mappán belül (Ctrl+V)
- Figyeljünk, hogy ne legyen üres sor, a fájl 40 sort tartalmazzon

Ezen lépések alkalmazásával az összes kényelmi formázás megtartható.

### Kaja frissítési térkép

<img src="/pics/foodupdate.png" width="200">

Minden mezőhöz a pályán tartozik egy kaja termelődési valószínűség, mely segítségével a játék
lefutása során a kaják térképen való elhelyezkedése jelentősen megváltozhat. A Config.py fileban
rögzített paraméterek szerint bizonyos tickenként valamilyen térkép elosztás szerint random helyeken
1 értékű kaják jelennek meg, melyek a tickek során felhalmozódhatnak 2 vagy 3 szintig. 

Előre generált valószínűségi térképek a fieldupdate mappában találhatóak, de további térképek 
is generálhatóak a feladat minél jobb megoldása érdekében. 
Térkép generáláshoz hasznos lehet a **fieldupdate.xlsx** fájl. 

## Útmutató

### Paraméterek

**Config.py** <br>
Ez a file tartalmazza a játék főbb beállításait, melyeket a készülés során is lehet állítani. Illetve
itt vannak meghatározva a játékszabályok és a kijelző színpalettája.

| **Paraméter**         | **Default érték**             | **Magyarázat**                                         |
|-----------------------|-------------------------------|--------------------------------------------------------|
| **#GameMaster**       |                               |                                                        |
| GAMEMASTER_NAME       | "master"                      | Game master név                                        |
| IP                    | "localhost"                   | A játék IP címe                                        |
| PORT                  | 42069                         | A játék által nyitott port                             |
| DEFAULT_TICK_LENGTH_S | 0.3                           | Egy TICK lefutási ideje                                |
| DISPLAY_ON            | True                          | Kijelző bekapcsolása                                   |     
| WAIT_FOR_JOIN         | 20                            | Indítás utáni várakozás a játékosok bejelntkezéséhez   |    
| LOG                   | True                          | Logolás be/kikapcsolása                                |           
| LOG_PATH              | './log'                       | Logfileok mentési helye                                |  
| **#Engine**           |                               |                                                        |           
| MAPPATH               | "./maps/02_base.txt"          | Játék térkép elérési útja                              |
| FIELDUPDATE_PATH      | "./fieldupdate/01_corner.txt" | Kaja termelődés valószínűségi térképének elérési útja  |
| STARTING_SIZE         | 5                             | Kezdő játékos méret                                    |
| MIN_RATIO             | 1.1                           | Bekebelezési arány (a kisebb játékos méretét tekintve) |
| STRATEGY_DICT         | {}                            | Játékos stratégiák                                     |
| VISION_RANGE          | 5                             | Játékosok látási távolsága                             |
| UPDATE_MODE           | "statistical"                 | Kaja újratermelődés módja (static - nincs termelődés)  |
| DIFF_FROM_SIDE        | 1                             | Kezdő pozíciók távolsága a pálya szélétől (4 sarok)    |
| FOODGEN_COOLDOWN      | 10                            | Kaja termelődés ciklusideje tickekben                  |
| FOODGEN_OFFSET        | 10                            | Kaja termelődés először ebben a Tickben                |
| FOODGEN_SCALER        | 0.3                           | Kaja termelődés valószínűségi térképének módosítója.   |
| MAXTICKS              | 100                           | Játék maximális Tick száma                             |
| SOLO_ENABLED          | True                          | A játék futásának engedélyezése solo módba             |

### Játékos stratégiák

**Player.py** <br>
Ez a file tartalmaz pár előre megírt botot, melyekkel tesztelhető a rendszer és az új fejlesztésű játékos teljesítménye.

**RemotePlayerStrategy:**<br>
A távoli csatlakozású játékos. Erre a beállításra lesz szükség a saját játékosunk futtatásához.
A `Main_Client.py`-ban kódolt 'hunter' így tud csatlakozni a GameMasterhez.

**DummyStrategy:** <br>
Indítás után meghaló játékos.

**RandBotStrategy:** <br>
Random akciókat választó játékos.

**NaiveStrategy:** <br>
A látóterének legnagyobb értékű kajája felé haladó játékos (egyenlőség esetén a bal felső a célpont).

**NaiveHunterStrategy:** <br>
A látóterének legnagyobb értékű kajája felé haladó játékos (egyenlőség esetén a bal felső a célpont), de ha másik játékost lát és meg tudja enni, akkor vadászk rá.

## Motor működése
Engine.py

A játékmestert és a játékmotort alapvetően a feladat során nem kell programozni, de a működését érdemes megérteni.   
A motor az alábbi lépéseket végzi el minden ciklusban ebben a sorrendben:   
- Kilépési feltétel vizsgálata (maximális tick szám, élő játékosok száma)
- Tervezett akciók végrehajtása (két karakterből álló stringek (egy-egy tengelyhez): "0" a tengely menti helyben maradás, "+" a pozitív irányú lépés, "-" a negatív irányú lépés)
- A tervezett akciók után kialakult ütközések megoldása (bekebelezés, visszapattanás, lásd fentebb)
- Játékosok pozíciójának és új méretének véglegesítése
- Étel növesztése a térképen
- Log információk írása
- Tick sorszám inkrementálása
- Látótérben található információk kiküldése a játékosoknak

## Játékmester működése
Gamemaster.py

A játékmester a motort és a kommunikációs szervert koordinálja, fogadja a játék irányító üzeneteket.   
A játékmester állapotai a következők:
- PRERUN: Az első futás előtti állapot, ilyenkor a motor már futáskész, a szerver képes fogadni a kapcsolatokat, de csak akkor indul el, ha minden távoli játékos felregisztrált a szerverre, vagy ha letelik az előre beállított időtartam le nem telik (30 másodperc).
- RUNNING: A játék maga fut, a motort ciklikusan léptetjük. A megadott ciklusidő a feldolgozási ciklusok indításai között eltelt idő.
- WAIT_COMMAND: A játék futásának végén ebbe az állapotba kerül, itt a reset GameControl paranccsal lehetséges új játékot indítani akár új térkép betöltése mellett is. A maximális várakozási idő lejártával a program automatikusan kilép.
- WAIT_START: A reset parancs után a játék ebbe az állapotba kerül. Ekkor start GameControl paranccsal lehetséges az indítás. Ebben az állapotban a játék végtelen ideig várakozik.

Az összes fenti állapotból ki lehet lépni interrupt GameControl paranccsal.
A játékosok a szerverre való csatlakozáskor meg kell, hogy adják nevüket, a SetName paranccsal, egyéb esetben nem kapnak információt a játékmestertől.

## Kliens-szerver kommunikáció
Example_Client_Main.py

A feladat során a játékba kliensként van lehetőség becsatlakozni. A kommunikáció kétirányú, TCP/IP protokollon alapuló socket kommunikáció, melyet JSON formátumban valósítunk meg.   
https://www.w3schools.com/js/js_json_intro.asp

Python nyelven dict, list, str és float értékekkel a JSON funkcionalitást egyszerűen reprodukálhatjuk, a json.loads() és a json.dumps() parancsokkal könnyedén konvertálhatunk ilyen struktúrákat egyik formátumból a másikba.
Az alábbiakban az érvényes üzenet struktúrák kerülnek részletezésre.
Fontos kiegészítés: A kliens-szerver kommunikáció során futtatókörnyezettől függően 2-5 ms késleltetés felléphet tickenként, így érdemes az akciót előállító folyamatot a játékmotor ciklusidejénél legalább ennyivel rövidebbre tervezni. 

### Szerveroldali üzenetek
Ezen üzeneteket a szerver küldi a kliensnek.
Két kötelező kulccsal rendelkeznek, ezek: 
- 'type' (leaderBoard, readyToStart, started, gameData, serverClose)
- 'payload' (az üzenet adatrésze)

A 'payload' tartalma típusfüggő:

- 'leaderBoard' type a játék végét jelzi, a payload tartalma **{'ticks': a játék hossza tickekben, 'players':[{'name': jáétékosnév, 'active': él-e a játékos?, 'maxSize': a legnagyobb elért méret a játék során},...]}**
- 'readyToStart' type esetén a szerver az indító üzenetre vár esetén, a payload üres (**None**)
- 'started' type esetén a játék elindul, tickLength-enként kiküldés és akciófogadás várható payload **{'tickLength': egy tick hossza }**
- 'gameData' type esetén az üzenet a játékos által elérhető információkat küldi, a payload:
                                    **{"pos": abszolút pozíció a térképen, "tick": az aktuális tick sorszáma, "active": a saját életünk állapota,
                                    "size": saját méret, "vision": [{"relative_coord": az adott megfigyelt mező relatív koordinátája,
                                                                    "value": az adott megfigyelt mező értéke (0-3,9),
                                                                    "player": None, ha nincs aktív játékos, vagy
                                                                            {name: a mezőn álló játékos neve, size: a mezőn álló játékos mérete}},...] }**
- 'serverClose' type esetén a játékmester szabályos, vagy hiba okozta bezáródásáról értesülünk, a payload üres (**None**)

### Kliensoldali üzenetek
Az alábbi üzenetekkel lehetünk ráhatással a játékmester és a játékmotor viselkedésére.

Az elküldött adat struktúrája minden esetben **{"command": Parancs típusa, "name": A küldő azonosítója, "payload": az üzenet adatrésze}**   
Az elérhető parancsok a következők:
- 'SetName' A kliens felregisztrálja a saját nevét a szervernek, enélkül a nevünkhöz tartozó üzenetek nem térnek vissza.
                 Tiltott nevek: a configban megadott játékmester név és az 'all'.
- 'SetAction' Ebben az esetben a payload az **akció string**, amely két karaktert tartalmaz az X és az Y koordináták (matematikai mátrix indexelés) menti elmozdulásra. A karakterek értékei **'0'**: helybenmaradás az adott tengely mentén, **'+'** pozitív irányú lépés, **'-'** negatív irányú lépés lehetnek. Amennyiben egy tick ideje alatt nem küldünk értéket az alapértelmezett '00' kerül végrehajtásra.
- 'GameControl' üzeneteket csak a Config.py-ban megadott játékmester névvel lehet küldeni, ezek a játékmenetet befolyásoló üzenetek. A payload az üzenet típusát (type), valamint az ahhoz tartozó 'data' adatokat kell, hogy tartalmazza. 
    - 'start' type elindítja a játékot egy "readyToStart" üzenetet küldött játék esetén, 'data' mezője üres (**None**)
    - 'reset' type egy játék után várakozó 'leaderBoard'-ot küldött játékot állít alaphelyzetbe. A 'data' mező **{'mapPath':None, vagy elérési útvonal, 'updateMapPath': None, vagy elérési útvonal}** formátumú, ahol None esetén az előző pálya és növekedési map kerül megtartásra, míg elérési útvonal megadása esetén új pálya kerül betöltésre annak megfelelően.
    - 'interrupt' type esetén a 'data' mező üres (**None**), ez megszakítja a szerver futását és szabályosan leállítja azt.


## Logolás
A Config.py fileban engedélyezhető a logolás. Ebben az esetben a játékmotor minden lépés végén beleír a megfelelő útvonalon található fileba. Minden lépést egyetlen sortörés karakter választ el. A lépések adatai az alábbi JSON formátumban kerülnek mentésre:   
**{"tick:": lépés sorszáma, "actions": 4 hosszú akcióstring tömb, "player_info":[{"name": játékos neve, "pos": játékos pozíciója, "size": játékos aktuális mérete},...], "field":játéktér játékosok nélkül}**

## Credits
Gyöngyössy Natabara (gyongyossy.natabara@mogi.bme.hu) <br>
Nagy Balázs (nagybalazs@mogi.bme.hu)
