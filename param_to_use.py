import pandas as pd
import ast

class ParametersToUse:
    """
    class used to define the parameters used in the training of the EMG signals
    """
    def __init__(self):
        self.file = None
        self.cv = None
        self.random_state = None
        self.test_size = None
        self.type_matrix = None
        self.levels_to_use = None
        self.filter_to_use = None
        self.layers_to_catch = None
        self.sos_bandstop_ = None
        self.sos_high_pass_ = None
        self.n_of_channels_and_category = None
        self.frequency = None
        self.window_time = None
        self.time_between_captures_of_samples = None

        self.read_csv()

    def read_csv(self):
        df = pd.read_csv("Parameters/parameters.csv", index_col="Parameter")
        self.frequency = float(df.loc["frequency_of_capture", "Value"])
        self.time_between_captures_of_samples = (1 / self.frequency) * 1000.0
        self.window_time = float(df.loc["window_time", "Value"])
        self.n_of_channels_and_category = int((df.loc["n_of_channels_and_category", "Value"]))
        sos_high_pass_ = df.loc["sos_high_pass_", "Value"]
        self.sos_high_pass_ = ast.literal_eval(sos_high_pass_)
        sos_bandstop_ = df.loc["sos_bandstop_", "Value"]
        self.sos_bandstop_ = ast.literal_eval(sos_bandstop_)
        layers_to_catch = df.loc["layers_to_catch", "Value"]
        self.layers_to_catch = ast.literal_eval(layers_to_catch)
        self.filter_to_use = df.loc["filter_to_use", "Value"]
        self.levels_to_use = int(df.loc["levels_to_use", "Value"])
        self.type_matrix = df.loc["type_matrix", "Value"]
        self.test_size = float(df.loc["test_size", "Value"])
        self.random_state = int(df.loc["random_state", "Value"])
        self.cv = int(df.loc["cv", "Value"])
        self.file = df.loc["file", "Value"]
