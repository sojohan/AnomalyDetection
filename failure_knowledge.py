"""Curated historical failure and sensor reference from Data Description_Metro.pdf.

Failure records are retrospective ground truth, NOT features for the live model.
Historical records are disclosed to the agent only after their end time.
"""
from datetime import datetime

FAILURES = [
    {'id':'F1','start':'2020-04-18 00:00','end':'2020-04-18 23:59','type':'Air leak','severity':'High stress','report':None},
    {'id':'F2','start':'2020-05-29 23:30','end':'2020-05-30 06:00','type':'Air leak','severity':'High stress','report':'PDF says Maintenance on 30Apr at 12:00; inconsistent with failure date, not verified'},
    {'id':'F3','start':'2020-06-05 10:00','end':'2020-06-07 14:30','type':'Air leak','severity':'High stress','report':'Maintenance on 8Jun at 16:00'},
    {'id':'F4','start':'2020-07-15 14:30','end':'2020-07-15 19:00','type':'Air leak','severity':'High stress','report':'Maintenance on 16Jul at 00:00'},
]
SENSORS = {
    'TP2':'Compressor pressure (bar)',
    'TP3':'Pneumatic panel pressure (bar)',
    'H1':'Pressure generated during cyclonic separator filter discharge (bar)',
    'DV_pressure':'Air dryer tower discharge pressure drop; zero indicates compressor under load (bar)',
    'Reservoirs':'Downstream reservoir pressure; expected near TP3 (bar)',
    'Motor_current':'One motor phase current (A); approximately 0 off, 4 offloaded, 7 loaded, 9 starting',
    'Oil_temperature':'Compressor oil temperature (C)',
    'COMP':'Air intake valve signal; active when off or offloaded',
    'DV_electric':'Outlet valve control; active under load',
    'TOWERS':'Which drying tower is operating',
    'MPG':'Compressor start-under-load control activated when APU pressure below 8.2 bar',
    'LPS':'Low-pressure signal active below 7 bar',
    'Pressure_switch':'Air-drying tower discharge detector',
    'Oil_level':'Active when compressor oil level below expected values',
    'Caudal_impulse':'Pulse count corresponding to air flow to reservoirs',
}

def parse_time(value):
    try:
        return datetime.fromisoformat(str(value).replace('Z','+00:00')).replace(tzinfo=None)
    except (ValueError, TypeError):
        return None

def reference_for_event(timestamp):
    """Return only historical failures known by event timestamp (avoid hindsight leakage)."""
    t = parse_time(timestamp)
    prior = [f for f in FAILURES if t is not None and parse_time(f['end']) < t]
    return {'source':'Data Description_Metro.pdf, pages 2-3',
            'sensor_definitions':SENSORS,
            'prior_documented_failures':prior,
            'caution':'The PDF provides intervals, not causal sensor signatures or confirmed diagnoses for new events. Future and ongoing failure records are withheld.'}

def retrospective_failure_match(timestamp):
    """For offline evaluation only; NEVER supply this to agents during replay."""
    t = parse_time(timestamp)
    if t is None: return None
    return next((f['id'] for f in FAILURES if parse_time(f['start']) <= t <= parse_time(f['end'])), None)
