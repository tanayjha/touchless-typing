# import cPickle
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import testcrop
import predictor
# matplotlib.use('GTKAgg')

def plotting(itemlist, model, session, saver, save_path, hyperparams):
	x = [row[0] for row in itemlist]
	y = [row[1] for row in itemlist]
	plt.axis((0,800,600,0))
	plt.plot(x,y)
	plt.savefig('foo1.jpeg')
	try:
	    testcrop.CropImage()
	    predictor.predict(model, session, saver, save_path, hyperparams)
	except IndexError:
	    print("Draw Again!!!")
	    pass
	plt.close()