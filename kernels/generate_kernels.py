import re

#thanks to txt2re.com for this regular expression
re1='.*?'	# Non-greedy match on filler
re2='(?:[a-z][a-z0-9_]*)'	# Uninteresting: var
re3='.*?'	# Non-greedy match on filler
re4='((?:[a-z][a-z0-9_]*))'	# Variable Name 1

rg = re.compile(re1+re2+re3+re4,re.IGNORECASE|re.DOTALL)

template = ''
with open("matrix_kernels_template.tcl", 'r') as file:
	for line in file:
		template = template + line

t_string = rg.search(template)
t_string = t_string.group(1)
print(t_string)

template = template.replace("#define " + t_string, "")
#all types supported by opencl (sorta... for now)
types = ["char", "short", "int", "long", "float", "double"]

for t in types:
	kernel = template.replace(t_string, t)

	with open("matrix_kernels_" + t + ".cl", 'w') as file:
		file.write(kernel)
