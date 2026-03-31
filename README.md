# U.S. Drug Overdose Mortality Analysis (1999–2018)

## Overview
Analyzed U.S. drug overdose death rates across demographic groups and drug 
types using a CDC dataset of 6,229 observations spanning 1999–2018. 
Conducted hypothesis testing to identify statistically significant trends 
in racial disparities and the fentanyl crisis.

## Tools & Methods
- **Language:** Python 
- **Libraries:** pandas, numpy, matplotlib, seaborn, scipy
- **Statistical Tests:** One-sample t-test, Welch's two-sample t-test
- **Techniques:** Data cleaning, demographic extraction, descriptive 
  statistics, hypothesis testing, time series visualization

## Dataset
- Source: CDC National Center for Health Statistics
- 6,229 observations of age-adjusted drug overdose death rates per 100,000
- Variables: drug type, year (1999–2018), sex, race/ethnicity, age group

## Key Findings
- Q1: REJECT H0 — 2017 sample mean (8.57) differs significantly from 
  CDC benchmark, reflecting disaggregated demographic structure
- Q2: REJECT H0 — Non-Hispanic Black death rates increased ~180% from 
  early to late period (p < 0.001)
- Q3: REJECT H0 — Synthetic opioid death rates rose dramatically after 
  2014, coinciding with illicitly manufactured fentanyl entering the 
  drug supply


