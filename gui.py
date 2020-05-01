from flask import Flask, render_template, request
import os
import server
from flask import jsonify
from flask import session

gui_variables = {}

app = Flask(__name__)
app.secret_key = 'A0Zr98j/3yX R~XHH!jmN]LWX/,?RT'


@app.route("/")
def hello():
    return render_template('index.html')


@app.route("/load_lanes", methods=['POST'])
def load_lanes():
    files = []
    try:

        for file in request.files.getlist("data"):
            files.append(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))

    except:
        return False

    days, lanes_ids = server.loadMonths(files, False)

    gui_variables = {
        "lanes_ids": lanes_ids,
        "lanes": {}
    }

    server.sumo.load_lanes_edges(gui_variables)

    session['inflow_edges'] = list(gui_variables["inflow_edges"])

    return jsonify(list(gui_variables["inflow_edges"]))


@app.route("/run_sim", methods=['POST'])
def run_sim():
    files = []
    if request.form["mode"] == "test_static":

        try:

            for file in request.files.getlist("data"):
                files.append(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))

        except:
            return False

        days, lanes_ids = server.loadMonths(files, False)

        gui_variables = {
            "lanes_ids": lanes_ids,
            "lanes": {}
        }

        server.sumo.load_lanes_edges(gui_variables)
        session['inflow_edges'] = list(gui_variables["inflow_edges"])

    inflow_edges = list(session['inflow_edges'])
    print(inflow_edges)

    gui_variables = {
        "mode": str(request.form['mode']),
        "phase_duration": int(request.form['phase_duration'] if request.form['phase_duration'] != "" else 0),
        "current_day": str(request.form['current_day']),
        "current_time": str(request.form['current_time']),
        "accumulate_phase": int(request.form['accumulate_phase'] if request.form['accumulate_phase'] != "" else 0),
        "accumulate_start": int(request.form['accumulate_start'] if request.form['accumulate_start'] != "" else 0),
        "accumulate_duration": int(
            request.form['accumulate_duration'] if request.form["accumulate_duration"] != "" else 0),
        "max_queue_lengths": {},
        "max_time_delays": {}
    }

    for inflow_edge in inflow_edges:
        gui_variables["max_queue_lengths"][inflow_edge] = int(request.form[inflow_edge + "0"]) if request.form[
                                                                                                      "mode"] != "test_static" else 1
        gui_variables["max_time_delays"][inflow_edge] = int(request.form[inflow_edge + "1"]) if request.form[
                                                                                                    "mode"] != "test_static" else 1

    try:
        gui_variables["accumulation"] = bool(request.form['accumulation'])
    except:
        gui_variables["accumulation"] = False

    try:
        request.form['sumo']
        gui_variables["sumo"] = "sumo-gui"
    except:
        gui_variables["sumo"] = "sumo"

    if request.form["mode"] != "test_static":
        try:
            for file in request.files.getlist("data"):
                files.append(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
            gui_variables["file_names"] = files

        except:
            gui_variables["file_names"] = False

    else:
        gui_variables["file_names"] = files

    for key, value in gui_variables.items():
        print(key, " : ", value)

    server.start(gui_variables)

    return gui_variables["summary"]


if __name__ == "__main__":
    app.config["TEMPLATES_AUTO_RELOAD"] = True
    app.config['UPLOAD_FOLDER'] = "gui-data"
    app.config['SESSION_TYPE'] = 'filesystem'
    app.run()
