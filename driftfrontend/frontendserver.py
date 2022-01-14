

from flask import Flask, render_template
import pickle
app = Flask(__name__)



@app.route("/<uuid4>")
def project(uuid4):
    with open('/driftfrontend/templates/'+str(uuid4)+'_historical.pkl', 'rb') as f:
        data = pickle.load(f)
    dates = data.keys()
    detection = []
    for date in dates:
        lock = True
        for feature in data[date]:
            if data[date][feature]['Drift detected'] == "True":
                detection.append('True')
                lock = False
                break
        if lock:
            detection.append('False')
    
    print(detection)
    return render_template('project_template.html', dates=dates, projectid = str(uuid4), detection = detection)



@app.route("/<uuid4>/report/<date>")
def driftpage(uuid4, date):
    return render_template(f'{uuid4}_report_{date}.html')





if __name__ == "__main__":
    # Only for debugging while developing
    app.run(host='0.0.0.0', debug=True, port=5000)