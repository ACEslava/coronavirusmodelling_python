# coronavirusmodelling_python

A system to model COVID-19 using a compartmental model of epidemiology.
Contains 2 implementations, a network based model and a mathematical function based model.

###Instructions for use:

Run learning.py to generate parameters based on COVID19-US data

To update datasets: CSV format, chronologically ordered. Needs Diagnoses, Recoveries, and Deaths data



###v1.0 Changelog

* SIDHE.py
  * Spun SIDHE class into separate file
  * fullsimulation() working

* learning.py 
  * Learns w/ fullsimulation() model
  * Outputs final parameters to parameters.txt in format: a,b,g,d,z,o
  
* COVID Datasets
  * Updated to 2020-11-03 from JHU database
