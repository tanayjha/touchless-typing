import matplotlib
import matplotlib.pyplot as plot
import pylab
from matplotlib import pyplot
matplotlib.use("WXAgg")
with open('abc') as f:
  array = []
  start_pts = 500	
  end_pts = 200
  for line in f: 
    array.append([int(x) for x in line.split()])
for i in range(start_pts, len(array)-end_pts):
	if((array[i+1][0]-array[i][0])*(array[i+1][0]-array[i][0]) + (array[i+1][1]-array[i][1])*(array[i+1][1]-array[i][1]) <= 1000):
		plot.plot([700-array[i][0], 700-array[i+1][0]], [400-array[i][1], 400-array[i+1][1]])
axes = plot.gca()
axes.set_xlim([0,711])
axes.set_ylim([0,400])
# F = pylab.gcf()
# F.set_size_inches( (0.08, 0.08) )
# DefaultSize = F.get_size_inches()
# print ("Default size in Inches", DefaultSize)
pylab.savefig("test3.jpeg")
plot.show()
# pyplot.show()