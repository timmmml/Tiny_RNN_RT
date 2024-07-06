# cognitive_dynamics

Detect cognitive dynamics underlying behavior with RNNs

1. User-specific path issues:

data_path.json, containing *absolute* paths to the data files, should be created in the root directory of the project.

For example, the file should look like this:

{"BartoloMonkey": "D:\\p7ft2bvphx-1"}

All code in the project will use (mostly) the *relative* pathstored in path_settings.py; 
thus any scripts called by the console or the main file should from utils import goto_root_dir.

2. Environment requirements:

pytorch, scikit-learn, numpy, scipy, pandas, matplotlib.

3. Multiprocessing issues (multiprocessing is disabled for now, so ignore this):
- Advanced system settings
- Advanced tab
- Performance - Settings button
- Advanced tab - Change button
- Uncheck the "Automatically... BLA BLA" checkbox
- Select the System managed size option box.

4. If any errors related to joblib loading:
- joblib =1.2.0

5. How to use?
- check main.py for one entry
- In each experiment, the models are first trained (exp\*.py), analyzed (ana\*.py), then plotted (plotting\*.py).
- In the training experiment, agents are trained on some datasets.
- 