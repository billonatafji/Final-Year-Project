import os
os.environ["KERAS_BACKEND"] = "theano"
import json
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import sgd
import sys
import datetime
from keras import optimizers



class Simulation(object):
    def __init__(self, current_time,roads, num_phases):
        self.roads = roads
        self.num_phases = num_phases
        self.current_phase = 0
        self.current_time = current_time
        self.old_state = self.get_state()

    def get_reward(self):
        queue_length_reward = 0
        time_delay_reward = 0
        overflows = 0
        for road in self.roads:
            queue_length_reward += (sum(road.get_previous_queue_lengths()) - sum(road.get_queue_lengths()))#/road.get_max_allowed_queue_length()) #+ (1 if sum(road.get_overflows()) > 0 else 0)
            time_delay_reward += (sum(road.get_previous_time_delays()) - sum(road.get_time_delays()))#/road.get_max_allowed_time_delay()
            overflows += sum(road.get_overflows())
        return queue_length_reward + time_delay_reward #- overflows #(-1 if self.is_lane_overflow() else 0)

    def get_state(self):
        model_input = [self.current_phase]#self.current_phase]
        queue_lengths = []
        time_delays = []
        overflows = []
        for road in self.roads:
            queue_lengths += road.get_queue_lengths()
            time_delays += road.get_time_delays()
            overflows += road.get_overflows()
        return np.asarray(model_input+queue_lengths+time_delays)[np.newaxis]

    def get_current_time(self):
        return self.current_time

    def get_total_num_lanes(self):
        sum = 0
        for road in self.roads:
            sum += road.num_lanes
        return sum
    
    def get_current_phase(self):
        return self.current_phase

    def set_current_phase(self, current_phase):
        self.current_phase = current_phase

    def set_old_state(self):
        self.old_state = self.get_state()

    def get_old_state(self):
        return self.old_state

    def is_lane_overflow(self):
        for road in self.roads:
            if road.get_max_allowed_queue_length() < max(road.get_queue_lengths()):
                return True
        return False

    def get_roads(self):
        return self.roads

    # def reset(self):
    #     self.set_current_phase(np.random.randint(0,self.num_phases,1)[0])
    
    def print_simulation(self):
        print("\nSimulation: \n")
        print("\tCurrent Phase\t\t", self.current_phase)
        print("\tTotal Number of Lanes\t", self.get_total_num_lanes())  
        print("\tRoads:")   
        for road in self.roads:
            road.print_road()

class ExperienceReplay(object):
    def __init__(self, max_memory, discount):
        self.max_memory = max_memory
        self.memory = list()
        self.discount = discount

    def remember(self, experiences): # experiences is [old_state, phase, reward, state]
        self.memory.append([experiences])
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def get_batch(self, model, batch_size=10):
        len_memory = len(self.memory)
        num_phases = model.output_shape[-1]
        env_dim = self.memory[0][0][0].shape[1] # This is the size of the state vector in other words it is the input size to the model
        inputs = np.zeros((min(len_memory, batch_size), env_dim))# Create empty batches of input states
        targets = np.zeros((inputs.shape[0], num_phases)) #each input state in the batch will have rewards associated to phase choices
        for i, idx in enumerate(np.random.randint(0, len_memory,size=inputs.shape[0])):# inputs.shape[0] is the number of elements in inputs / Here we are generating the batch 
            old_state, phase, reward, state = self.memory[idx][0]# We get the value from memory at index equal to idx 
            #lane_overflow = self.memory[idx][1]# We get wether or not we had game over at idx in memory

            # print("\n-------------------","\nold state:\t",old_state,"\nphase:\t",phase,"\nreward:\t",reward,"\n state:\t",state,"\n-------------------")


            inputs[i:i+1] = old_state# We insert the values from the batch to the inputs array to use in batch training 
            # There should be no target values for actions not taken.
            # Thou shalt not correct actions not taken #deep
			
			
            targets[i] = model.predict(old_state)[0]#Get from the model the rewards expected for the actions given each input state in inputs
            Q_sa = np.max(model.predict(state)[0])# Get the maximum expected reward for the future state

            targets[i, phase] = reward + self.discount * Q_sa# If the lane did not overflow set the reward as sum of immediate reward and discounted future reward
        
        return inputs, targets

class Road(object):
    def __init__(self, name, max_allowed_queue_length, max_allowed_time_delay, lanes):
        self.name = name
        self.num_lanes = len(lanes)
        self.max_allowed_queue_length = max_allowed_queue_length
        self.max_allowed_time_delay = max_allowed_time_delay
        self.lanes = {}
        for lane in lanes:
            self.lanes[lane.get_id()] = lane

    def get_num_lanes(self):
        return self.num_lanes

    def get_queue_lengths(self):
        queue_lengths = []
        for key in self.lanes:
            queue_lengths.append(self.lanes[key].get_queue_length())
        return queue_lengths

    def get_previous_queue_lengths(self):
        previous_queue_lengths = []
        for key in self.lanes:
            previous_queue_lengths.append(self.lanes[key].get_previous_queue_length())
        return previous_queue_lengths

    def get_avg_queue_lengths(self):
        avg_queue_lengths = 0.
        for key in self.lanes:
            avg_queue_lengths += self.lanes[key].get_queue_length()
        return avg_queue_lengths/len(self.lanes)

    def get_avg_time_delays(self):
        avg_time_delays = 0.
        for key in self.lanes:
            avg_time_delays += self.lanes[key].get_time_delay()
        return avg_time_delays/len(self.lanes)

    def get_time_delays(self):
        time_delays = []
        for key in self.lanes:
            time_delays.append(self.lanes[key].get_time_delay())
        return time_delays

    def get_overflows(self):
        overflows = []
        for key in self.lanes:
            overflows.append(self.lanes[key].get_queue_length()>self.get_max_allowed_queue_length())
        return overflows

    def get_previous_time_delays(self):
        previous_time_delays = []
        for key in self.lanes:
            previous_time_delays.append(self.lanes[key].get_previous_time_delay())
        return previous_time_delays

    def get_max_allowed_queue_length(self):
        return self.max_allowed_queue_length

    def get_max_allowed_time_delay(self):
        return self.max_allowed_time_delay
        
    def reset(self):
        for key in self.lanes:
            self.lanes[key].set_queue_length(np.random.randint(10,self.max_allowed_queue_length/4,1)[0])
            self.lanes[key].set_time_delay(np.random.randint(5,self.max_allowed_time_delay*5,1)[0])


    def print_road(self):
        print("\n\t\tRoad:")
        print("\t\t\tLanes:")
        for key in self.lanes:
            self.lanes[key].print_lane()
        print("\n\t\t\tMax Queue Length:\t", self.get_max_allowed_queue_length())
        print("\t\t\tNumber of Lanes:\t", self.get_num_lanes())
        print("\t\t\tQueue Lengths:\t\t", self.get_queue_lengths())
        print("\t\t\tTime Delays:\t\t",self.get_time_delays())

class Lane(object):
    def __init__(self, id):
        self.id = id
        self.current_queue_length = 0.
        self.current_time_delay = 0.
        self.previous_queue_length = 0.
        self.previous_time_delay = 0.

    def get_queue_length(self):
        return self.current_queue_length

    def set_queue_length(self,queue_length):
        self.current_queue_length = queue_length

    def get_previous_queue_length(self):
        return self.previous_queue_length

    def set_previous_queue_length(self):
        self.previous_queue_length = self.current_queue_length

    def get_time_delay(self):
        return self.current_time_delay

    def set_time_delay(self, time_delay):
        self.current_time_delay = time_delay

    def get_previous_time_delay(self):
        return self.previous_time_delay

    def set_previous_time_delay(self):
        self.previous_time_delay = self.current_time_delay

    def get_id(self):
        return self.id


    def update_lane(self, new_queue_length, new_time_delay):
        self.set_previous_queue_length()
        self.set_previous_time_delay()
        self.set_queue_length(new_queue_length)
        self.set_time_delay(new_time_delay)

    def print_lane(self):
        print("\n\t\t\t\tLane:\t\t",self.get_id())
        print("\t\t\t\tQueue Length:\t",self.get_queue_length())
        print("\t\t\t\tTime Delay:\t",self.get_time_delay())

def train(shared_variables):

    shared_variables["modelEvent"].wait()
    shared_variables["modelEvent"].clear()

    global env
    global roads


    for key in shared_variables["lanes"]:
        shared_variables["lanes"][key] = Lane(key)

    # print(shared_variables["lanes"])
    # sys.exit()

    print(shared_variables["edges_lanes"])

    roads = [
        Road("zeytouna-ain_mreysse",100,100,[shared_variables["lanes"]["zeytouna-ain_mreysse_0"],shared_variables["lanes"]["zeytouna-ain_mreysse_1"],shared_variables["lanes"]["zeytouna-ain_mreysse_2"],shared_variables["lanes"]["zeytouna-ain_mreysse_3"]]),
        Road("aub-ain_mreysse",100,100,[shared_variables["lanes"]["aub-ain_mreysse_0"],shared_variables["lanes"]["aub-ain_mreysse_1"]]),
        Road("bliss-ain_mreysse",100,200,[shared_variables["lanes"]["bliss-ain_mreysse_0"],shared_variables["lanes"]["bliss-ain_mreysse_1"]])
    ]

    # for edge, lanes in shared_variables["edges_lanes"].items():
    #     lanes_input = []
    #     for lane in lanes:
    #         lanes_input.append(shared_variables["lanes"][lane])
    #     roads.append(Road(edge,shared_variables["max_queue_lengths"][edge],shared_variables["max_time_delays"][edge],lanes_input))

	# Define environment/game
    num_phases = shared_variables["num_phases"]
    env = Simulation(shared_variables["current_time"],roads,num_phases )
    phase_duration = 0
    num_lanes = env.get_total_num_lanes()
    input_size = 1 + 2*num_lanes

	#Define Model
    model = Sequential()
    model.add(Dense(input_size, input_shape=(input_size,), activation='relu'))
    model.add(Dense(input_size, activation='relu'))
    model.add(Dense(num_phases))
    model.compile(sgd(lr=shared_variables["learning_rate"]), shared_variables["optimizer"])

    try:
        model.load_weights("model.h5")
    except:
        print("failed to load model")

    # Initialize experience replay object
    exp_replay = ExperienceReplay(shared_variables["max_memory"], shared_variables["discount"])

    shared_variables["lock"].acquire()
    while not shared_variables["doneEvent"].is_set():

        shared_variables["lock"].release()
        shared_variables["modelEvent"].wait()
        shared_variables["modelEvent"].clear()

        state = "\nold state:\t\t" + str(np.asarray(env.get_old_state()[0]).tolist())
        state += "\nnew phase:\t\t" + str(env.get_current_phase())
        state += "\nreward:\t\t\t" + str(env.get_reward())
        state += "\nnew state:\t\t" + str(np.asarray(env.get_state()[0]).tolist()) + "\n"


        if shared_variables["mode"] == "train":
            # # adapt model
            exp_replay.remember([env.get_old_state(), env.get_current_phase(), env.get_reward(), env.get_state()])
            inputs, targets = exp_replay.get_batch(model, batch_size= shared_variables["batch_size"])
            shared_variables["loss"] += model.train_on_batch(inputs, targets)

        elif shared_variables["mode"] == "test_model":
            exp_replay.remember([env.get_old_state(), env.get_current_phase(), env.get_reward(), env.get_state()])
            inputs, targets = exp_replay.get_batch(model, batch_size= shared_variables["batch_size"])
            shared_variables["loss"] += model.test_on_batch(inputs, targets)

        for edge, lanes in shared_variables["edges_lanes"].items():
            shared_variables["cumulative_queue_length_" + edge] += [road for road in env.get_roads() if road.name == edge][0].get_avg_queue_lengths()
            shared_variables["cumulative_time_delay_" + edge] += [road for road in env.get_roads() if road.name == edge][0].get_avg_time_delays()
        
        phase_duration += 1
        env.set_old_state()



        # if phase_duration == shared_variables["phase_duration"]:
        phase_duration = 0
            # get next phase
        if shared_variables["mode"] == "train" and np.random.rand() <= shared_variables["exploration"]:
            new_phase = np.random.randint(0, num_phases, size=1)[0]
        else:
            q = model.predict(env.get_state())
            new_phase = np.argmax(q[0])
        
        env.set_current_phase(new_phase)
        shared_variables["new_phase"] = env.get_current_phase() #if not shared_variables["accumulation"] or accumulation_counter > shared_variables["accumulate_duration"] else shared_variables["accumulate_phase"]

  
        # apply new phase, get rewards and new state

        
        #state += "\nnew phase:\t\t" + str(shared_variables["new_phase"]) + "\n"

        # print(state)

        shared_variables["states"] += state

        shared_variables["lock"].acquire()
        shared_variables["serverEvent"].set()

    if shared_variables["mode"] == "train":
        model.save_weights("model.h5", overwrite=True)
        # with open("model.json", "w") as outfile:
        #     json.dump(model.to_json(), outfile)

    shared_variables["lock"].release()

    print("Model exiting")
