

from flask import Flask, render_template
import pickle
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
app = Flask(__name__)


#/driftfrontend/templates/
@app.route("/<uuid4>")
def project(uuid4):
    with open('/driftfrontend/templates/'+str(uuid4)+'_historical.pkl', 'rb') as f:
        data = pickle.load(f)
    dates = list(data.keys())
    detection = []
    detection_numbers = []
    for date in dates:
        detection_num = 0
        detection_result = 'False'
        for feature in data[date]:
            if 'Drift detected' in data[date][feature]:
                print(data[date])
                if data[date][feature]['Drift detected'] == 'True':
                    detection_result = 'True'
                    detection_num += 1
        detection.append(detection_result)
        detection_numbers.append(detection_num)
    
    last_ten_detection_dates = dates[-10:]
    all_features = {}
    for date in last_ten_detection_dates:
        pandas_frame = data[date]['data']
        for feature in pandas_frame.columns:
            if feature not in all_features:
                all_features[feature] = list(pandas_frame[feature].values)
            else:
                all_features[feature] += list(pandas_frame[feature].values)
    
    with open('/driftfrontend/templates/'+str(uuid4)+'.pkl', 'rb') as f:
        data = pickle.load(f)
    
    refdata_df = pd.DataFrame(data['reference_data'])
    
    all_features_df = pd.DataFrame(all_features)
    fig, ax = plt.subplots(1, len(all_features_df.columns))
    for i, feature in enumerate(all_features_df.columns):
        if i == 0:
            sns.kdeplot(all_features_df[feature], ax=ax[i], shade = True, label = 'new data', color = 'red')
            sns.kdeplot(refdata_df[feature], ax=ax[i], shade = True, label = 'reference data', color = 'green', linewidth = 0)
        sns.kdeplot(all_features_df[feature], ax=ax[i], shade = True, color = 'red')
        sns.kdeplot(refdata_df[feature], ax=ax[i], shade = True, color = 'green', linewidth = 0)
    ax[0].legend()
    plt.tight_layout()
    plt.savefig('/driftfrontend/static/'+str(uuid4)+'_historical.png')
    plt.close()
    return render_template('project_template.html', dates=dates, projectid = str(uuid4), detection = detection, detection_numbers = detection_numbers, numberoffeatures = len(all_features_df.columns))




@app.route("/<uuid4>/report/<date>")
def driftpage(uuid4, date):
    return render_template(f'{uuid4}_report_{date}.html')




if __name__ == "__main__":
    # Only for debugging while developing
    app.run(host='0.0.0.0', debug=True, port=5000)