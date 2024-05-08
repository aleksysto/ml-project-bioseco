from keras.utils import plot_model
from keras.models import load_model
model = load_model("./new_data_model/73_new_data_second_go.keras")
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
print(model.summary())
