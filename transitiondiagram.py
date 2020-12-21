import numpy as np
from graphviz import Digraph

dot = Digraph(comment='The Round Table')
dot.format = 'svg'
dot.attr(rankdir='LR')
dot.attr('node', shape='circle', fixedsize='true', width = '1')
dot.attr('edge', minlen='2')

dot.node('S', 'Susceptible')
dot.node('I', 'Infected')
dot.node('D', 'Diagnosed')
dot.node('H', 'Healthy')
dot.node('E', 'Expired')

dot.edge('S','I', label='S(βI + αD)')
dot.edge('I','D', label='γI')
dot.edge('I','H', label='δI')
dot.edge('D','H', label='ζD')
dot.edge('D','E', label='ωD')

a = Digraph('child1')
a.attr(rank='same')
a.node('I')
a.node('D')
a.edge('I','D', style='invis')
dot.subgraph(a)

b = Digraph('child2')
b.attr(rank='same')
b.node('H')
b.node('E')
b.edge('H','E', style='invis')
dot.subgraph(b)

print(dot.source)

dot.render('test-output/round-table.gv', view=True,  renderer='cairo')  