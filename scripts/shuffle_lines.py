import sys, random

fin = open(sys.argv[1], 'r')
data = fin.readlines()
fin.close()

random.shuffle(data)

sys.stdout.writelines(data)
