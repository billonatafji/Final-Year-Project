from __future__ import absolute_import
from __future__ import print_function
import os
import sys
import optparse
import random
from xml.dom import minidom

# we need to import python modules from the $SUMO_HOME/tools directory


if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

import traci


def get_options():
    optParser = optparse.OptionParser()

    optParser.add_option("--nogui", action="store_true",

                         default=False, help="run the commandline version of sumo")

    options, args = optParser.parse_args()

    return options


def load_lanes_edges(shared_variables):
    file = open("sumo-data/map.rou.xml", "w")
    file.write("<routes></routes>")
    file.close()

    traci.start(["sumo", "-c", "sumo-data/map.sumocfg",
                 "--start", "--quit-on-end", "--no-step-log"])
    links = {}

    inflow_lanes = set(traci.lane.getIDList()).intersection(
        shared_variables["lanes_ids"])
    inflow_edges = set()
    for inflow_lane in inflow_lanes:
        inflow_edges.add(traci.lane.getEdgeID(inflow_lane))
        links[inflow_lane] = [x[0] for x in traci.lane.getLinks(inflow_lane)]
    edges_lanes = {}
    inflow_edges = list(inflow_edges)
    inflow_edges.sort()
    for inflow_edge in inflow_edges:
        shared_variables["cumulative_queue_length_" + inflow_edge] = 0.
        shared_variables["cumulative_time_delay_" + inflow_edge] = 0.
        edges_lanes[inflow_edge] = []
    for inflow_lane in inflow_lanes:
        edges_lanes[traci.lane.getEdgeID(inflow_lane)].append(inflow_lane)
        shared_variables["lanes"][inflow_lane] = ""

    shared_variables["edges_lanes"] = edges_lanes

    edges = set()

    links_edges = {}

    for link_key, link_value in links.items():
        links_edges[link_key] = []
        for link in link_value:
            links_edges[link_key].append(traci.lane.getEdgeID(
                link_key) + " " + traci.lane.getEdgeID(link))
            edges.add(traci.lane.getEdgeID(link_key) +
                      " " + traci.lane.getEdgeID(link))

    shared_variables["links_edges"] = links_edges
    shared_variables["inflow_edges"] = inflow_edges
    shared_variables["edges"] = edges

    traci.close()


# this is the main entry point of this script

def generate_route(shared_variables):
    load_lanes_edges(shared_variables)

    route_file = """<routes>
        <vType id="type1" accel="2.6" decel="4.5" sigma="0.5" length="5" maxSpeed="70"/>\n"""

    for edge in shared_variables["edges"]:
        route_file += "<route id=\"" + edge + "\" edges=\"" + edge + "\" />\n"

    with open("sumo-data/map.rou.xml", "w") as routes:

        random.seed(0)
        vehicles = "<root>"

        hour_offset = 0

        if shared_variables["current_time"] == "Full Day":

            for hour_key, hour_value in shared_variables["days"][shared_variables["current_day"]].items():

                detectors = shared_variables["days"][shared_variables["current_day"]][hour_key]

                for j in range(len(shared_variables["lanes_ids"])):
                    for i in range(detectors[j]):
                        rand_route_index = random.randint(
                            0, len(shared_variables["links_edges"][shared_variables["lanes_ids"][j]]) - 1)

                        vehicles += '<vehicle id="%s" type="type1" route="%s" depart="%f" departLane= "%s" />' % (
                            shared_variables["lanes_ids"][j] +
                            "_" + str(3600 * hour_offset + i),
                            shared_variables["links_edges"][shared_variables["lanes_ids"]
                            [j]][rand_route_index],
                            3600 * hour_offset + i * 3600 / detectors[j],
                            shared_variables["lanes_ids"][j][len(shared_variables["lanes_ids"][j]) - 1])

                        shared_variables["num_vehicles"] += 1

                hour_offset += 1

        else:

            detectors = shared_variables["days"][shared_variables["current_day"]
            ][shared_variables["current_time"]]
            for j in range(len(shared_variables["lanes_ids"])):
                for i in range(detectors[j]):
                    rand_route_index = random.randint(
                        0, len(shared_variables["links_edges"][shared_variables["lanes_ids"][j]]) - 1)

                    vehicles += '<vehicle id="%s" type="type1" route="%s" depart="%f" departLane= "%s" />' % (
                        shared_variables["lanes_ids"][j] + "_" + str(i),
                        shared_variables["links_edges"][shared_variables["lanes_ids"]
                        [j]][rand_route_index],
                        i * 3600 / detectors[j],
                        shared_variables["lanes_ids"][j][len(shared_variables["lanes_ids"][j]) - 1])

                    shared_variables["num_vehicles"] += 1

        vehicles += "</root>"

        dom = minidom.parseString(vehicles)
        nodes = dom.getElementsByTagName('vehicle')

        nodes.sort(key=lambda x: float(x.attributes['depart'].value))
        vehicles = ""
        for node in nodes:
            vehicles += node.toxml() + "\n"

        print(route_file, vehicles, "</routes>", file=routes)


def runSim(shared_variables):
    generate_route(shared_variables)

    traci.start([shared_variables["sumo"], "-c", "sumo-data/map.sumocfg", "--start",
                 "--quit-on-end", "--queue-output", "localhost:5123", "--no-step-log"])

    shared_variables["sumoEvent"].wait()
    shared_variables["sumoEvent"].clear()

    shared_variables["lock"].acquire()
    shared_variables["modelEvent"].set()
    shared_variables["serverEvent"].set()

    while traci.simulation.getMinExpectedNumber() > 0 and not shared_variables["doneEvent"].is_set():

        shared_variables["lock"].release()
        shared_variables["sumoEvent"].wait()
        shared_variables["sumoEvent"].clear()

        for i in range(shared_variables["phase_duration"]):

            if shared_variables["mode"] != "train" and shared_variables["accumulation"] and shared_variables[
                "accumulate_start"] <= shared_variables["current_step"] and shared_variables["current_step"] <= \
                    shared_variables["accumulate_start"] + shared_variables["accumulate_duration"]:
                traci.trafficlight.setPhase(traci.trafficlight.getIDList()[
                                                0], shared_variables["accumulate_phase"])

            elif shared_variables["mode"] != "test_static":
                traci.trafficlight.setPhase(traci.trafficlight.getIDList()[
                                                0], shared_variables["new_phase"])

            traci.simulationStep()

        shared_variables["current_step"] += 1

        shared_variables["lock"].acquire()
        shared_variables["serverEvent"].set()

    shared_variables["doneEvent"].set()
    shared_variables["lock"].release()

    traci.close()
    sys.stdout.flush()

    print("sumo exiting")
