import sys
import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn2pmml import PMMLPipeline, sklearn2pmml

def is_non_zero_file(fpath):
    return os.path.isfile(fpath) and os.path.getsize(fpath) > 0

if __name__ == "__main__":
    # Find the location where data are saved
    end = "grcuda/projects/com.nvidia.grcuda/src/com/nvidia/grcuda/runtime/stream/data/"
    abspath = os.path.abspath("")
    path = ""
    for dir in abspath.split("/"):
        if dir == "grcuda":
            path += end
            break
        path += (dir + "/")

    for name in sys.argv[1:]:
        if is_non_zero_file(path + name + ".csv") == False: #if the file does not exist or is empty
            print(path + name + ".csv empty or not found")
        else:
            #grouping data and saving median time for each group
            df = pd.read_csv(path + name + ".csv")
            df = df.groupby([x for x in df.columns if x != "Time"], as_index=False).agg({'Time':'median'})
            y = df.pop("Time")
            #creating model
            pipeline = PMMLPipeline([ ('regressor', RandomForestRegressor(random_state=42, n_estimators=300)) ]) #random forest regressor
            pipeline.fit(df, y) #fitting the model
            #saving model
            if not os.path.exists("trainedmodels/"):
                os.makedirs("trainedmodels/")
            sklearn2pmml(pipeline, "trainedmodels/" + name + ".pmml", with_repr = True)