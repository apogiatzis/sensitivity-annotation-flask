# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 04:43:00 2019

@author: Andreas
"""

def generate_reverse_table():
    table = {}
    for st in SENSITIVITY_TYPES:
        table[st["class"]] = st
    return table

def generate_color_options():
    options = {}
    for st in SENSITIVITY_TYPES:
        if "color" in st: options[st["label"]] = st["color"]
    return options

def generate_start_tag_table():
    tag_table = {}
    for st in SENSITIVITY_TYPES:
        tag_table[st['start']]  = st
    return tag_table

## Future Color ref
# =============================================================================
#  colors = {'ORG': '#7aecec', 'PRODUCT': '#bfeeb7', 'GPE': '#feca74',
#                   'LOC': '#ff9561', 'PERSON': '#aa9cfc', 'NORP': '#c887fb',
#                   'FACILITY': '#9cc9cc', 'EVENT': '#ffeb80', 'LAW': '#ff8197',
#                   'LANGUAGE': '#ff8197', 'WORK_OF_ART': '#f0d0ff',
#                   'DATE': '#bfe1d9', 'TIME': '#bfe1d9', 'MONEY': '#e4e7d2',
#                   'QUANTITY': '#e4e7d2', 'ORDINAL': '#e4e7d2',
#                   'CARDINAL': '#e4e7d2', 'PERCENT': '#e4e7d2'}
# =============================================================================


GENERAL_SENSITIVITY = {"start":  "B-SENSITIVITY", "end": "E-SENSITIVITY",
                       "class": 1, "label":"GENERAL"}
PHYSICAL_CONTACT_INFO_SENSITIVITY = {"start":  "B-PHSCL_CNCT_INFO", "end": "E-PHSCL_CNCT_INFO",
                                     "class": 2, "label":"PHSCL_CNCT_INFO", "color": "#7aecec"}
ONLINE_CONTACT_INFO_SENSITIVITY = {"start":  "B-ONLINE_CTCT_INFO", "end": "E-B-ONLINE_CTCT_INFO",
                                   "class": 3, "label":"ONLINE_CNCT_INFO", "color":"#ff8197"}
UNIQUE_IDENTIFIER_SENSITIVITY = {"start":  "B-UNIQUE_IDENTIFIER", "end": "E-UNIQUE_IDENTIFIER",
                                   "class": 4, "label":"UNIQUE_IDENTIFIER", "color":"#9cc9cc"}
PURCHASE_INFORMATION_SENSITIVITY = {"start":  "B-PURCHASE_INFO", "end": "E-PURCHASE_INFO",
                                   "class": 5, "label":"PURCHASE_INFO", "color":"#aa9cfc"}
FINANCIAL_INFORMATION_SENSITIVITY = {"start":  "B-FINANCIAL_INFO", "end": "E-FINANCIAL_INFO",
                                   "class": 6, "label":"FINANCIAL_INFO", "color":"#bfe1d9"}
COMPUTER_INFORMATION_SENSITIVITY = {"start":  "B-COMPUTER_INFO", "end": "E-COMPUTER_INFO",
                                   "class": 7, "label":"COMPUTER_INFO", "color":"#5c85ab"}
STATE_MANAGEMENT_SENSITIVITY = {"start": "B-STATE_MGMT", "end": "E-STATE_MGMT",
                                   "class": 8, "label":"STATE_MANAGEMENT", "color":"#b094bd"}
DEMOGRAPHIC_SOCIOECONOMIC_SENSITIVITY = {"start": "B-DEMOGRAPHIC_SOCIOECONOMIC_INFO", "end": "E-DEMOGRAPHIC_SOCIOECONOMIC_INFO",
                                   "class": 9, "label":"DEMOGRAPHIC_SOCIOECONOMIC_INFO", "color":"#c2c972"}
POLITICAL_INFORMATION_SENSITIVITY = {"start": "B-POLITICAL_INFO", "end": "E-POLITICAL_INFO",
                                   "class": 10, "label":"POLITICAL_INFO", "color":"#61df95"}
HEALTH_INFORMATION_SENSITIVITY = {"start": "B-HEALTH_INFO", "end": "E-HEALTH_INFO",
                                   "class": 11, "label":"HEALTH_INFO", "color":"#b861df"}
PREFERENCE_INFORMATION_SENSITIVITY = {"start": "B-PREFERENCE_INFO", "end": "E-PREFERENCE_INFO",
                                   "class": 12, "label":"PREFERENCE_INFO", "color":"#df6f61"}
LOCATION_INFORMATION_SENSITIVITY = {"start": "B-LOCATION_INFO", "end": "E-LOCATION_INFO",
                                   "class": 13, "label":"LOCATION_INFORMATION", "color":"#a0df61"}
GOVERNMENT_IDENTIFIER_SENSITIVITY = {"start": "B-GOVERNMENT_IDENTIFIER", "end": "E-GOVERNMENT_IDENTIFIER",
                                   "class": 14, "label":"GOVERNMENT_IDENTIFIER", "color":"#15a2ff"}
SENSITIVITY_TYPES = [
            PHYSICAL_CONTACT_INFO_SENSITIVITY,
            ONLINE_CONTACT_INFO_SENSITIVITY,
            UNIQUE_IDENTIFIER_SENSITIVITY,
            PURCHASE_INFORMATION_SENSITIVITY,
            FINANCIAL_INFORMATION_SENSITIVITY,
            COMPUTER_INFORMATION_SENSITIVITY,
            STATE_MANAGEMENT_SENSITIVITY,
            DEMOGRAPHIC_SOCIOECONOMIC_SENSITIVITY,
            POLITICAL_INFORMATION_SENSITIVITY,
            HEALTH_INFORMATION_SENSITIVITY,
            PREFERENCE_INFORMATION_SENSITIVITY,
            LOCATION_INFORMATION_SENSITIVITY,
            GOVERNMENT_IDENTIFIER_SENSITIVITY
        ]

VIZ_COLOR_OPTIONS = generate_color_options()

REVERSE_SENSITIVITY_TABLE = generate_reverse_table()

LABELS_LIST = list(map(str,REVERSE_SENSITIVITY_TABLE.keys())) + ['0','-1']

START_TAG_TABLE = generate_start_tag_table()
