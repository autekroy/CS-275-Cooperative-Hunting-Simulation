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
