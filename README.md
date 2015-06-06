### CS-275-Cooperative-Hunting-Simulation

------------------------------------------------------------
#### How To Run:
* Quick Run: 	               python simulation.py
* Use a save file for animats: python simulation.py filename.dat

#### Quick Model Training Example
* import animats
* e = animats.Environment(10, 1000, 700, "example.dat")
* e.update() # repeat as desired
* e.save()

------------------------------------------------------------
#### Python Package:
1. Pygame: for GUI
2. pybrain: for machine learning
3. [enum34](https://pypi.python.org/pypi/enum34#downloads): only defult in python 3.4 or latter
4. [sklearn](http://scikit-learn.org/dev/install.html): for machine learning