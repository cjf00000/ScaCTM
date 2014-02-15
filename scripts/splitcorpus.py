import sys, os

if (len(sys.argv) != 3):
	print 'Usage: python %s <original corpus name> <number>' % sys.argv[0]
	sys.exit(0)

m = int(sys.argv[2])
l = open(sys.argv[1]).readlines()
n = len(l)

p = (n-1)/m + 1

for i in xrange(0, m):
	begin = i*p
	end = min( (i+1)*p, n )

	f = open('%s.%d' % (sys.argv[1], i), 'w')
	f.writelines(l[begin:end])
	f.close()
