import glob, os

f = open("myground.gt", "w")
c = 1

os.chdir('.')
for file in glob.glob('*.JPG'):
	#print(file)
	tempStr = str(c) 
	f.write(file + " 45 45 90 90 " + tempStr + " 1" +"\n")
	c+=1

f.close()