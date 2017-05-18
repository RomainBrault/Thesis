# -*- coding: UTF-8 -*-



import os
import sys
import codecs

if '--interactive' not in sys.argv[1:]:
    if sys.version_info[0] == 2:
        sys.stdout = codecs.getwriter('UTF-8')(sys.stdout, 'strict')
        sys.stderr = codecs.getwriter('UTF-8')(sys.stderr, 'strict')
    else:
        sys.stdout = codecs.getwriter('UTF-8')(sys.stdout.buffer, 'strict')
        sys.stderr = codecs.getwriter('UTF-8')(sys.stderr.buffer, 'strict')

if '/usr/local/texlive/2016/texmf-dist/scripts/pythontex' and '/usr/local/texlive/2016/texmf-dist/scripts/pythontex' not in sys.path:
    sys.path.append('/usr/local/texlive/2016/texmf-dist/scripts/pythontex')    
from pythontex_utils import PythonTeXUtils
pytex = PythonTeXUtils()

pytex.docdir = os.getcwd()
if os.path.isdir('.'):
    os.chdir('.')
    if os.getcwd() not in sys.path:
        sys.path.append(os.getcwd())
else:
    if len(sys.argv) < 2 or sys.argv[1] != '--manual':
        sys.exit('Cannot find directory .')
if pytex.docdir not in sys.path:
    sys.path.append(pytex.docdir)



pytex.id = 'py_default_default'
pytex.family = 'py'
pytex.session = 'default'
pytex.restart = 'default'

pytex.command = 'c'
pytex.set_context('')
pytex.args = ''
pytex.instance = '0'
pytex.line = '514'

print('=>PYTHONTEX:STDOUT#0#c#')
sys.stderr.write('=>PYTHONTEX:STDERR#0#c#\n')
pytex.before()

print(r'\centering\resizebox{\textwidth}{!}{\input{./not_Mercer.pgf}}')


pytex.after()
pytex.command = 'c'
pytex.set_context('')
pytex.args = ''
pytex.instance = '1'
pytex.line = '416'

print('=>PYTHONTEX:STDOUT#1#c#')
sys.stderr.write('=>PYTHONTEX:STDERR#1#c#\n')
pytex.before()

print(r'\centering \resizebox {\textheight }{!}{\input {./representer.pgf}}')


pytex.after()
pytex.command = 'c'
pytex.set_context('')
pytex.args = ''
pytex.instance = '2'
pytex.line = '561'

print('=>PYTHONTEX:STDOUT#2#c#')
sys.stderr.write('=>PYTHONTEX:STDERR#2#c#\n')
pytex.before()

print(r'\centering\resizebox{\textwidth}{!}{\input{./efficient_decomposable_gaussian.pgf}}')


pytex.after()
pytex.command = 'c'
pytex.set_context('')
pytex.args = ''
pytex.instance = '3'
pytex.line = '576'

print('=>PYTHONTEX:STDOUT#3#c#')
sys.stderr.write('=>PYTHONTEX:STDERR#3#c#\n')
pytex.before()

print(r'\centering\resizebox{\textwidth}{!}{\input{./efficient_curlfree_gaussian.pgf}}')


pytex.after()
pytex.command = 'c'
pytex.set_context('')
pytex.args = ''
pytex.instance = '4'
pytex.line = '593'

print('=>PYTHONTEX:STDOUT#4#c#')
sys.stderr.write('=>PYTHONTEX:STDERR#4#c#\n')
pytex.before()

print(r'\centering\resizebox{\textwidth}{!}{\input{./efficient_divfree_gaussian.pgf}}')


pytex.after()
pytex.command = 'c'
pytex.set_context('')
pytex.args = ''
pytex.instance = '5'
pytex.line = '32'

print('=>PYTHONTEX:STDOUT#5#c#')
sys.stderr.write('=>PYTHONTEX:STDERR#5#c#\n')
pytex.before()

print(r'\centering\resizebox{\textwidth}{!}{\input{./approximation.pgf}}')


pytex.after()


pytex.cleanup()
