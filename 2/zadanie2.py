#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By  : Ilya Ryzhkov / Katarzyna Węsierska
# Created Date: nov 2022
# version = '0.1'
# Algorytm FL oblicza ryzyko nadania dostępu do pomieszczenia na
# podstawie wartości odczytanych z czujników temperatury,
# wilgotności, i wysokości obiektu
# Required: numpy, scikit-fuzzy
# ---------------------------------------------------------------------------
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


class AccessControlRisk:
    """
    Klasa wywołuje framework FL i deklaruje reguły:
    ryzyko jest wysokie w przypadku podwyższonej temperatury i wilgotności
    lub niskiego wzrostu
    """
    def __init__(self):
        self.humidity = ctrl.Antecedent(np.arange(0, 1, 0.01), 'humidity')
        self.tall = ctrl.Antecedent(np.arange(1, 250, 1), 'tall')
        self.temp = ctrl.Antecedent(np.arange(35, 42, 0.1), 'temp')
        self.risk = ctrl.Consequent(np.arange(0, 1, 0.01), 'risk')

        # Mapujemy zakresy liczb
        self.humidity.automf(5, variable_type='quant')
        self.tall.automf(7, variable_type='quant')
        self.temp.automf(3, variable_type='quant')
        self.risk.automf(3, variable_type='quant')

        self.risk['low'] = fuzz.trimf(self.risk.universe, [0, 0, 0.33])
        self.risk['medium'] = fuzz.trimf(self.risk.universe, [0, 0.33, 0.5])
        self.risk['high'] = fuzz.trimf(self.risk.universe, [0.5, 1, 1])

        self.rules = [
            ctrl.Rule(self.tall['high'] & self.temp['low'] &
                      self.humidity['low'], self.risk['low']),
            ctrl.Rule(self.tall['higher'] & self.temp['low'] &
                      self.humidity['low'], self.risk['low']),
            ctrl.Rule(self.tall['highest'] & self.temp['low']
                      & self.humidity['low'], self.risk['low']),
            ctrl.Rule(self.tall['high'] & self.humidity['average']
                      & self.temp['low'], self.risk['average']),
            ctrl.Rule(self.tall['higher'] & self.humidity['average']
                      & self.temp['low'],
                      self.risk['average']),
            ctrl.Rule(self.tall['highest'] & self.humidity['average']
                      & self.temp['low'],
                      self.risk['average']),
            ctrl.Rule(self.tall['low'] | self.tall['average']
                      | self.tall['lower'] | self.tall['lowest'],
                      self.risk['high']),
            ctrl.Rule(self.humidity['high'] | self.temp['high'],
                      self.risk['high'])
        ]

        self.access_ctrl = ctrl.ControlSystem(self.rules)
        self.access = ctrl.ControlSystemSimulation(self.access_ctrl)

    def compute(self, humidity, tall, temp):
        """Zwraca obliczone ryzyko"""
        self.access.input['humidity'] = humidity
        self.access.input['tall'] = tall
        self.access.input['temp'] = temp
        self.access.compute()
        return str(self.access.output['risk'])


access_control_risk = AccessControlRisk()
print('5% 170cm 36.6 C = ' + access_control_risk.compute(0.05, 170, 36.6))
# Wynik: 0.14895755305867667

print('30% 170cm 36.6 C = ' + access_control_risk.compute(0.5, 170, 36.6))
# Wynik: 0.3776572454105089

print('70% 170cm 36.6 C = ' + access_control_risk.compute(0.7, 170, 36.6))
# Wynik: 0.6809200791795371

print('40% 170cm 40 C = ' + access_control_risk.compute(0.4, 170, 40.0))
# Wynik: 0.7955923250321608

print('10% 100cm 36.6 C = ' + access_control_risk.compute(0.1, 100, 36.6))
# Wynik: 0.8093567967688188

print('70% 100cm 36.6 C = ' + access_control_risk.compute(0.7, 100, 36.6))
# Wynik: 0.8232768179233729

print('50% 170cm 36.6 C = ' + access_control_risk.compute(0.5, 170, 36.6))
# Wynik: 0.49506390267089445

print('10% 250cm 36.6 C = ' + access_control_risk.compute(0.1, 1250, 36.6))
# Wynik: 0.13447960618846697
