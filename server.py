import socket
import sumo
import threading
import time
import re
import pandas as pd
import model
from xml.dom import minidom
import os


def parseXML(xml_string, shared_variables):
    try:
        xmldoc = minidom.parseString(xml_string)
        xmlLanes = xmldoc.getElementsByTagName("lanes")
        lanes_keys = []

        for node in xmlLanes[len(xmlLanes) - 1].childNodes:
            if node.attributes["id"].value in shared_variables["lanes"]:
                lanes_keys.append(node.attributes["id"].value)
                shared_variables["lanes"][node.attributes["id"].value].update_lane(float(
                    node.attributes["queueing_length"].value), float(node.attributes["queueing_time"].value))

        for key in shared_variables["lanes"]:
            if key not in lanes_keys:
                shared_variables["lanes"][key].update_lane(0., 0.)

    except Exception as e:
        print("failed to parse", e)


def serve(shared_variables):
    time.sleep(2)

    start = True

    HOST = ''  # Symbolic name meaning the local host
    PORT = 5123  # Arbitrary non-privileged port

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((HOST, PORT))
    s.listen(1)

    conn, addr = s.accept()

    # print('Connected by', addr)

    temp = ""
    shared_variables["lock"].acquire()
    while not shared_variables["doneEvent"].is_set():

        shared_variables["sumoEvent"].set()
        shared_variables["lock"].release()

        shared_variables["serverEvent"].wait()
        shared_variables["serverEvent"].clear()

        if not start:
            data = conn.recv(1024 * 15 * 1024)

            tags = ["<" + a.decode('ascii') + ">" for a in re.findall(rb"<(.*)>", data)]
            parseXML("<root>" + "".join(tags) + "</root>", shared_variables)
            if tags:
                shared_variables["shared_list"].extend(tags)

        else:
            start = False
        # lastTag = re.search(rb"^[^<]*>", data)
        # nextTag = re.search(rb"<[^>]*$", data)

        # if lastTag:
        #     sharedList.append(temp + lastTag.group().decode('ascii'))

        # if nextTag:
        #     temp = nextTag.group().decode('ascii')

        # if not start:

        shared_variables["lock"].acquire()
        shared_variables["modelEvent"].set()
        shared_variables["lock"].release()

        shared_variables["serverEvent"].wait()
        shared_variables["serverEvent"].clear()

        # print("\nfrom server after model \n")

        shared_variables["lock"].acquire()

    shared_variables["lock"].release()

    conn.close()

    print("server exiting")

    f = open("sumo-data/sharedList.xml", "w")
    f.write("\n".join(shared_variables["shared_list"]))
    f.close()
    # lock.release()


def loadMonths(month_files, reset):
    if reset:
        try:
            f = open("model-statistics.txt", "w")
            f.write("")
            f.close()
            os.remove("model.h5")
        except:
            print("model.h5 not found")
    days = {}
    for month_file in month_files:
        day = {}
        data = pd.read_csv(month_file, sep=',')

        lanes_ids = data.columns.values

        new_day = True
        for key, value in data.iterrows():
            data = value.tolist()
            if new_day:
                day = {}
            day[data[0].split(" ")[1]] = data[1:len(data)]

            days[data[0].split(" ")[0]] = day
            if data[0].split(" ")[1] == "23":
                new_day = True
            else:
                new_day = False
    return days, lanes_ids[1:len(lanes_ids)]


def epoch(gui_variables, days, lanes_ids):
    shared_variables = {
        ################################################################################
        "mode": gui_variables["mode"],
        "current_day": gui_variables["current_day"],
        "current_time": gui_variables["current_time"],
        "phase_duration": gui_variables["phase_duration"],
        "accumulation": gui_variables["accumulation"],
        "accumulate_duration": gui_variables["accumulate_duration"],
        "accumulate_phase": gui_variables["accumulate_phase"],
        "accumulate_start": gui_variables["accumulate_start"],
        "sumo": gui_variables["sumo"],
        "max_queue_lengths": gui_variables["max_queue_lengths"],
        "max_time_delays": gui_variables["max_time_delays"],
        ################################################################################
        "optimizer": "msle",
        "learning_rate": .003,
        "discount": 0.5,
        "max_memory": 500,
        "exploration": .1,
        "batch_size": 50,
        ################################################################################
        "current_phase": 0,
        "current_step": 1,
        "new_phase": 0,
        "num_phases": 3,
        "num_vehicles": 0,
        "loss": 0.,
        "avg_queue_length": 0.,
        "avg_time_delay": 0.,
        "days": days,
        "doneEvent": threading.Event(),
        "serverEvent": threading.Event(),
        "sumoEvent": threading.Event(),
        "modelEvent": threading.Event(),
        "lock": threading.Lock(),
        "states": "",
        "lanes": {
        },
        "lanes_ids": lanes_ids,
        "edges_lanes": {},
        "start": True,
        "shared_list": []
    }

    serverThread = threading.Thread(target=serve, args=(shared_variables,))
    serverThread.start()

    sumoThread = threading.Thread(target=sumo.runSim, args=(shared_variables,))
    sumoThread.start()

    modelThread = threading.Thread(
        target=model.train, args=(shared_variables,))
    modelThread.start()

    sumoThread.join()
    serverThread.join()

    f = open("model-statistics.txt", "a")

    avg_queue_length = 0
    avg_time_delay = 0

    html_summary = ''
    if shared_variables["mode"] != "test_static":
        html_summary += '<div class="row"><div class="col-md-6 font-weight-bold"><label>Phase Duration</label></div><div class="col"><label>' + str(
            shared_variables["phase_duration"]) + ' seconds</label></div></div>'
    html_summary += '\
    <div class="row"><div class="col-md-6 font-weight-bold"><label># Vehicles</label></div><div class="col"><p>' + str(
        shared_variables["num_vehicles"]) + '</p></div></div>\
    <div class="row"><div class="col-md-6 font-weight-bold"><label>Date</label></div><div class="col"><p>' + str(
        shared_variables["current_day"]) + '</p></div></div>\
    <div class="row"><div class="col-md-6 font-weight-bold"><label>Time</label></div><div class="col"><label>' + str(
        shared_variables["current_time"]) + '</label></div></div>\
    <div class="row"><div class="col-md-6 font-weight-bold"><label>Loss</label></div><div class="col"><label>' + str(
        shared_variables["loss"]) + '</label></div></div>\
    <div class="row"><div class="col-md-6 font-weight-bold"><label># Steps</label></div><div class="col"><label>' + str(
        shared_variables["current_step"]) + '</label></div></div>'

    summary = "\n\n--------------------------------------------------------------"
    summary += "\nMode:\t\t\t\t\t" + str(shared_variables["mode"])
    summary += "\n# Vehicles:\t\t\t\t" + str(shared_variables["num_vehicles"])
    if shared_variables["mode"] != "test_static":
        summary += "\nPhase Duration:\t\t\t\t" + str(shared_variables["phase_duration"]) + " seconds"
    summary += "\nDay:\t\t\t\t\t" + str(shared_variables["current_day"])
    summary += "\nTime:\t\t\t\t\t" + str(shared_variables["current_time"])
    summary += "\nLoss:\t\t\t\t\t" + str(shared_variables["loss"])

    summary += "\nNumber of steps:\t\t\t" + str(shared_variables["current_step"])

    for key in shared_variables:
        if "cumulative_queue_length_" in key:
            avg_queue_length += shared_variables[key] / shared_variables["current_step"]
            summary += "\nAverage queue length in " + key[24:] + ":\t\t" + str(
                shared_variables[key] / shared_variables["current_step"]) + " meters"
            html_summary += '<div class="row"><div class="col-md-6 font-weight-bold"><label>Avg. Queue Length in ' + key[
                                                                                                                     24:] + '</label></div><div class="col"><label>' + str(
                shared_variables[key] / shared_variables["current_step"]) + ' meters</label></div></div>'

    for key in shared_variables:
        if "cumulative_time_delay_" in key:
            avg_time_delay += shared_variables[key] / \
                              shared_variables["current_step"]
            summary += "\nAverage time_delay in " + key[22:] + ":\t\t" + str(
                shared_variables[key] / shared_variables["current_step"]) + " seconds"
            html_summary += '<div class="row"><div class="col-md-6 font-weight-bold"><label>Avg. Time Delay in ' + key[
                                                                                                                   22:] + '</label></div><div class="col"><label>' + str(
                shared_variables[key] / shared_variables["current_step"]) + ' seconds</label></div></div>'

    shared_variables["avg_queue_length"] = avg_queue_length / 3
    shared_variables["avg_time_delay"] = avg_time_delay / 3

    summary += "\nAverage queue_length in roads:\t\t" + str(shared_variables["avg_queue_length"]) + " meters"
    summary += "\nAverage time_delay in roads:\t\t" + str(shared_variables["avg_time_delay"]) + " seconds"
    summary += "\n--------------------------------------------------------------\n\n"

    html_summary += '<div class="row"><div class="col-md-6 font-weight-bold"><label>Avg. Queue Length in all Roads</label></div><div class="col"><label>' + str(
        shared_variables["avg_queue_length"]) + ' meters</label></div></div>'
    html_summary += '<div class="row"><div class="col-md-6 font-weight-bold"><label>Avg. Time Delay in all Roads</label></div><div class="col"><label>' + str(
        shared_variables["avg_time_delay"]) + ' seconds</label></div></div>'
    html_summary += '<hr/>'
    print(summary)

    f.write(summary)
    f.close()

    f = open("model-states.txt", "w")

    states = "\n\n\n\n\n-----------------------------------------------------------------------------------------------"
    states += shared_variables["states"]
    states += "\n----------------------------------------------------------------------------------------------\n\n\n\n\n"

    f.write(states)
    f.close()

    gui_variables["summary"] = html_summary


def start(gui_variables):
    days, lanes_ids = loadMonths(gui_variables["file_names"], False)

    if gui_variables["mode"] == "train":
        start = True
        for day_key, day_value in days.items():
            if day_key != gui_variables["current_day"] and start == True:
                continue
            else:
                start = False
                for hour_key, hour_value in day_value.items():  # days[gui_variables["current_day"]].items():#

                    gui_variables["current_day"] = day_key  # gui_variables["current_day"] #
                    gui_variables["current_time"] = hour_key

                    epoch(gui_variables, days, lanes_ids)


    else:
        epoch(gui_variables, days, lanes_ids)


if __name__ == "__main__":
    gui_variables = {

        "mode": "test_model",
        "current_day": "05/12/17",
        "current_time": "17",
        "phase_duration": 10,
        "accumulation": False,
        "accumulate_duration": 12,
        "accumulate_phase": 1,
        "accumulate_start": 10,
        "sumo": "sumo-gui",
        "file_names": ["sumo-data/December.csv", "sumo-data/January.csv", "sumo-data/November.csv"],
        "max_queue_lengths": {
            "bliss-ain_mreysse": 200,
            "zeytouna-ain_mreysse": 100,
            "aub-ain_mreysse": 100
        },
        "max_time_delays": {
            "bliss-ain_mreysse": 100,
            "zeytouna-ain_mreysse": 50,
            "aub-ain_mreysse": 50
        }
    }
    start(gui_variables)
