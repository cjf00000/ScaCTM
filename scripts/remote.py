from subprocess import *
import os

tansuo100='tansuo100'

def dumpArgs(func):
        '''Decorator to print function call details - parameters names and effective values'''
        def wrapper(*func_args, **func_kwargs):
                arg_names = func.func_code.co_varnames[:func.func_code.co_argcount]
                args = func_args[:len(arg_names)]
                defaults = func.func_defaults or ()
                args = args + defaults[len(defaults) - (func.func_code.co_argcount - len(args)):]
                params = zip(arg_names, args)
                args = func_args[len(arg_names):]
                if args: params.append(('args', args))
                if func_kwargs: params.append(('kwargs', func_kwargs))
                print func.func_name + ' (' + ', '.join('%s = %r' % p for p in params) + ' )'
                return func(*func_args, **func_kwargs)
        return wrapper  

def bash(command):
	#print 'Debug: executing ' + command
	f = open('TMP.sh', 'w')
	f.write(command)
	f.close()

	proc = Popen(['sh', 'TMP.sh'], stdout=PIPE, stderr=PIPE)
	retval = proc.communicate()

	os.remove('TMP.sh')
	return retval

def nohupBash(command):
	f = open('TMP.sh', 'w')
	f.write(command)
	f.close()

	proc = Popen(['nohup', 'sh', 'TMP.sh', '>output 2>&1 &'], stdout=PIPE, stderr=PIPE)

	return "Success"

def real_time_bash(command):
	#print 'Debug: executing ' + command
	f = open('TMP.sh', 'w')
	f.write(command)
	f.close()

	proc = Popen(['sh', 'TMP.sh', '1>&2'], stderr=PIPE)
	for line in proc.stderr:
		yield line.rstrip()

def pwd():
	return bash('pwd')[0].splitlines()[0]

def ssh(host, command, port=22):
	return bash('ssh -p %d %s -t "%s"' % (port, host, command))

def sshShell(host, command, port=22):
	return bash('echo "%s" | ssh -p %d %s' % (command, port, host))

def scp(src, dest, port=22):
	return bash('scp -P %d %s %s' % (port, src, dest))

def broadcast(src, dest, hosts):
	for host in hosts:
		scp(src, host + ':' + dest)

def gather(src_prefix, dest, hosts):
	for host in hosts:
		scp(host + ':' + src_prefix + '*', dest)

def ssh_all(hosts, command):
	for host in hosts:
		ssh(host, command)

def uploadData(dest):
	pass

def cleanData():
	pass

def cleanResults():
	pass

def getJobList():
	tmp = sshShell(tansuo100, 'bjobs -w')
	return tmp[0] + tmp[1]

def getQueues():
	tmp = sshShell(tansuo100, 'bqueues')
	return tmp[0] + tmp[1]

def launchJob(train_file, ntopics='100', run='0', beta='0.01', prior='1000', burnin='350', testiter='100', subiter='8', pgsamples='1', samplemode='pg1'):
	tmp = sshShell(tansuo100, '''cd /home/lijm/WORK/jianfei/ctm_general;
	pwd;
        sed -e "s/@train_file/%s/" \
            -e "s/@ntopics/%s/" \
            -e "s/@run/%s/" \
            -e "s/@beta/%s/" \
            -e "s/@prior/%s/" \
            -e "s/@iter/%s/" \
            -e "s/@testiter/%s/" \
            -e "s/@subiter/%s/" \
            -e "s/@pgsamples/%s/" \
            -e "s/@samplemode/%s/" \
multi-machine.sh | bsub;''' % (train_file, ntopics, run, beta, prior, burnin, testiter, subiter, pgsamples, samplemode))

	return tmp[0] + tmp[1]

def killJobs():
	return sshShell(tansuo100, 'cd ~/WORK/jianfei/ctm_general; ./clean.sh')

def killJob(jobid):
	tmp = sshShell(tansuo100, 'cd ~/WORK/jianfei/ctm_general; ./erase.sh %s' % jobid)

	return tmp[0] + tmp[1]

def jobStatus():
	tmp = sshShell(tansuo100, 'cd ~/WORK/jianfei/ctm_general; ./statusall.sh')

	return tmp[0] + tmp[1]

def listData():
	tmp = ssh(tansuo100, 'ls -l ~/WORK/jianfei/data/msra/*.txt')

	return tmp[0]

def formatData(name, path):
	return nohupBash('scp %s %s; ' % (path, tansuo100 + ":WORK/jianfei/tmp") + \
		'ssh %s -t "%s"' % (tansuo100, \
		'python /home/lijm/jianfei/research_scripts/python/shuffle_lines.py ~/WORK/jianfei/tmp/%s > ~/WORK/jianfei/data/msra/%s' % (name, name)))

if __name__ == '__main__':
	#print launchJob('nips_train.txt', '100', '0', '0.01', '1000', '350', '100', '8', '1', 'pg1')
	#print killJobs()
	#print killJob('result_nips_train.txt_t100b0.01p1000i350si8pg1pg1_run0')
	#print formatData('test.txt', '/home/temp/test.txt')
	#print pwd()
	#broadcast('/home/jianfei/test', ['juncluster2', 'juncluster4'])
	pass
