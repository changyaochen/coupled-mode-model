#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script is to process the data produced by Mathematica simulation.
The Mathematica script is "!=nonlinear MEMS calculations transient 1to1.nb"

for v2: I will use p.line, instead of: p.multi_line

@author: changyaochen
"""
import os, re
import bokeh.plotting as bkp
from bokeh.palettes import Category10
from bokeh.models import ColumnDataSource
from bokeh.layouts import gridplot, column

dname = os.path.dirname(os.path.abspath(__file__))
os.chdir(dname)

print('Processing files...') 
# now let me collect all the files
A_files, B_files, PhaseA_files, PhaseDiff_files = [], [], [], []
rous = []
all_files = os.listdir(dname)
for file in all_files:
  if os.path.isfile(file) and file.endswith('.dat'):
    if re.findall('(AmpA)', file):
      A_files.append(file)
      match = re.search('([-0-9\.]+).dat', file)
      rous.append(match.group(1))
    if re.findall('(AmpB)', file):
      B_files.append(file)
    if re.findall('(PhaseA)', file):
      PhaseA_files.append(file)
    if re.findall('(PhaseDiff)', file):
      PhaseDiff_files.append(file)
     
# now let me process all the files
to_remove = ['{', '}', '\n']
datas = [[] for _ in range(4)]
for i, rou in enumerate(rous):  # different rous
  # to plot the 4 different quantities that matter
  for j, q in enumerate([A_files, B_files, PhaseA_files, PhaseDiff_files]):
    file  = q[i]
    # process the file
    data = []
    with open(file, 'r') as infile:
      for line in infile:
        if len(line) > 0:
          for item in to_remove:
            line = line.replace(item, '')
          line = line.replace('*^', 'e')
          data.append(list(map(float, line.split(','))))
    datas[j].append(data)

print('Preparing plots...')
# let me plot  
N = len(rous) 
for j, label in enumerate(['AmpA', 'AmpB', 'PhaseA', 'PhaseDiff']):
  p = bkp.figure(plot_height = 300, plot_width = 800, toolbar_location = 'right',
                 tools = 'pan,box_zoom,reset,resize,save',
                 x_axis_label='Time (a.u.)', y_axis_label = label)

  output_html_file = label + '_test.html'
  if os.path.exists(output_html_file):
    os.remove(output_html_file)
  bkp.output_file(output_html_file)

  print('\nProcessing {}'.format(label))
  ts = [[x[0] for x in datas[j][i]] for i in range(len(rous))]
  ys = [[x[1] for x in datas[j][i]] for i in range(len(rous))]
  colors_list = Category10[N]
  legend_list = rous
  
  for (color, leg, x, y) in zip(colors_list, legend_list, ts, ys):
    temp_plot = p.line(x, y, color = color, legend = leg,
             alpha = 0.7, line_width = 2)
    
  bkp.save(p)  
  
  
