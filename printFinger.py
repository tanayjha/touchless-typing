import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plot
import pylab
import _pickle as cPickle
from matplotlib import pyplot

def PlotMe():
	number = 1
	cPickle.dump(number, open('save.p','wb'))
	# number = cPickle.load(open('save.p','rb'))
	with open('abc') as f:
	  array = []
	  start_pts = 500
	  end_pts = 200
	  for line in f: 
	    array.append([int(x) for x in line.split()])
	# print("HERE", len(array))	    
	for i in range(start_pts, len(array)-end_pts):
		if((array[i+1][0]-array[i][0])*(array[i+1][0]-array[i][0]) + (array[i+1][1]-array[i][1])*(array[i+1][1]-array[i][1]) <= 10000):
			plot.plot([700-array[i][0], 700-array[i+1][0]], [400-array[i][1], 400-array[i+1][1]])
	axes = plot.gca()
	axes.set_xlim([0,800])
	axes.set_ylim([0,600])
	name = 'foo'+str(number)+'.jpeg'
	plot.savefig(name)
	number+=1
	cPickle.dump(number, open('save.p','wb'))
	f.close()
	plot.close()