# coronavirusmodelling_python

A system to model COVID-19 using a compartmental model of epidemiology.
Contains 2 implementations, a network based model and a mathematical function based model.

v1.0 Changelog
SIDHE.py
  Spun SIDHE class into separate file
  fullsimulation() working

learning.py
  Learns w/ fullsimulation() model
  Outputs final parameters to parameters.txt in format: a,b,g,d,z,o
