msg = ""
try:
    msg += f"Python Version \t: {sys.version}\n"
except:
    pass
msg += f"\n"
try:
    msg += f"Tensorflow\n"
    msg += f" Version  \t: {tf.__version__}\n"
    msg += f" GPU ready \t: {tf.test.gpu_device_name() if tf.test.is_gpu_available() else ''}\n"
    msg += f"\n"
except:
    pass
try:
    msg += f"Keras\n"
    msg += f" Version \t: {keras.__version__}\n"
    from tensorflow.python import keras

    msg += f" GPU ready \t: {keras.backend._get_available_gpus()}\n"
    msg += f"\n"
except:
    pass
try:
    msg += f"scikit Version \t: {sklearn.__version__}\n"
except:
    pass

msg += f"\n"
try:
    msg += f"Statsmodel Vers\t: {sm.__version__}\n"
except:
    pass
try:
    msg += f"SciPy Version \t: {scipy.__version__}\n"
except:
    pass
try:
    msg += f"Numpy Version \t: {np.__version__}\n"
except:
    pass
try:
    msg += f"Pandas Version \t: {pd.__version__}\n"
except:
    pass
try:
    msg += f"Skyfield Version: {librosa.__version__}\n"
except:
    pass
try:
    msg += f"Librosa Version: {skyfield.VERSION}\n"
except:
    pass
try:
    msg += f"PyTables Version: {tables.__version__}\n"
except:
    pass
try:
    msg += f"HDF5 Version \t: {h5.__version__}\n"
except:
    pass
try:
    msg += f"TOML Version \t: {toml.__version__}\n"
except:
    pass
try:
    msg += f"Notebook Version: {nb.__version__}\n"
except:
    pass
try:
    msg += f"Matplotlib Vers\t: {matplotlib.__version__}\n"
except:
    pass
try:
    msg += f"Seaborn Version\t: {sns.__version__}\n"
except:
    pass
try:
    msg += f"Plotly Version\t: {py.__version__}\n"
except:
    pass
try:
    msg += f"Bokeh Version\t: {bokeh.__version__}\n"
except:
    pass
try:
    msg += f"Holoview Version: {hv.__version__}\n"
except:
    pass
try:
    msg += f"Pillow Version \t: {PIL.__version__}\n"
except:
    pass
try:
    msg += f"xlwings Version\t: {xl.__version__}\n"
except:
    pass
try:
    msg += f"black Version\t: {black.__version__}\n"
except:
    pass

msg += f"\n"

try:
    msg += f"IBAPI Version \t: {ibapi.__version__}\n"
except:
    pass
try:
    msg += f"InSync Version \t: {ib_insync.__version__}\n"
except:
    pass

print(msg)
