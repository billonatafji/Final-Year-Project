# :traffic_light: Welcome to my Final Year Project :red_car: :taxi: :blue_car:

## Description

This is Smart Traffic Light Using Machine Learning. In this project, we're trying to reduce traffic by optimizing 
the traffic light signals for a given intersection such that we reduce the average queue lengths and queueing times. 
This is done by taking into consideration the current state of an intersection and feeding it to a ML model which in 
return controls the signal phase of that intersection. 

You can read more about this in this paper [Smart Traffic Light Using Machine Learning](https://doi.org/10.1109/IMCET.2018.8603041).


## Stack

+ **Python**: our choice of programming language since it has great support for both, ML and TraCI (see below).

+ [**Keras**](https://keras.io/): for machine learning, which is a high-level neural networks API, backed with Theano.  

+ [**SUMO**](https://sumo.dlr.de/): for traffic simulation, which exposes APIs (TraCI) that allow us to control the environment.
 
 
## Modules

There are three main modules that made this possible:

+ **server.py**: the module that listens (using sockets) for structured input coming from the simulator as the environment changes and feeds it to the model module, and for decisions made by model module and feeds it back to the simulator.
+ **sumo.py**: the module that programmatically runs the simulator and initializes its environment, such as traffic roads, lanes, vehicles, and phases.
+ **model.py**: the module responsible for training and using the deep neural network.

## Getting Started

The project can be run on Windows, MacOS, and Linux given that both Python and SUMO are available for those platforms.

### SUMO Installation

In case your machine is running a debian-based Linux distro, you can easily install SUMO with two simple steps:

First, download the binaries that are part of the regular apt distribution:
```
sudo apt-get install sumo sumo-tools sumo-doc
```

Next, set the `SUMO_HOME` path variable in the `.bashrc` file. The default path should be `/usr/share/sumo`:
```
export SUMO_HOME="/usr/share/sumo/"
``` 

### Running the Project

To continue running the project: 

Install required Python packages using pip:

```
python3 -m pip install --user -r requirements.txt
```

Finally, you can run `server.py` which sets everything up and starts the simulator like so:
```
python3 server.py
```

## Configuration

The simulator is initialized with data, for a case-study intersection, and other configurations that can be changed 
either manually in `server.py` or graphically by running `gui.py` (which is a Flask module the represents the 
configurations/operations in a graphical user interface) and accessing it by a browser @ `localhost:5000`. 

### Data

The data used was actually collected from TMC (Traffic Management Center) for three months; 
November 2017, December 2017, and January 2018, and can be found @ the directory `sumo-data`. 

Each month's data is represented in a CSV file, where an entry represents an hour in that month along with the number
of vehicles that have passed for each lane for each road in that intersection.

You can specify which day and hour to run the simulator on using the fields `current_day` and `current_time` in 
`server.py` respectively. You can set `current_time` to `"Full Day"` in case you want to simulate that entire day. 

If configuring from the `gui.py`, you can select that through a date-picker field.

### Mode

There are three main modes that the project can be configured to run at:

+ **train**: when the model is to be trained by making decisions and getting rewards.
+ **test_model**: when the model is to be tested after it has been trained.
+ **test_static**: when the model is not involved in making decisions. In this case, traffic phases are changed according to a timer-based sequence (simulating a non-adaptive traffic light system). 

This mode can be configured through the field `mode` in `server.py`, or by selecting the mode in case you're running 
`gui.py`.


## Results

Each time you run `server.py`, whether for training or testing, an output will be appended to `model-statistics.txt`. 

This output includes the day and the time of the simulation, and more importantly the average queue lengths and queueing 
times the took presence in the simulation. 

When training the model, you should expect the average queue lengths and queueing times to be very high since the model
does not know how to make decisions yet (high loss) and hence manage the traffic. However, that should get better with each
iteration (reduced loss) up to a certain number of iterations in which after that over-fitting starts to take place and hurt the results.

You can use this file to watch how the model progresses during training, and how the model compares to a static solution post-training.

## Screenshots

### SUMO

![image](https://github.com/mbnatafgi/Final-Year-Project/raw/master/screenshots/sumo.png)

### GUI

![image](https://github.com/mbnatafgi/Final-Year-Project/raw/master/screenshots/gui.png)


>In case you inspect the code, you'll notice that there is a lot of room for improvement. We did not follow best 
>practices by any means, our target at that point was to just get things working since we were total beginners at 
>Python, machine learning, SUMO and pretty much all of this :p!

