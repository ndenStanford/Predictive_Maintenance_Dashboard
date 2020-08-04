#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 18:03:19 2020

@author: nutchapoldendumrongsup
"""

import dash
import dash_table
import pandas as pd
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import numpy as np
from function import *


df = pd.read_csv('../data/nab/realKnownCause/realKnownCause/ambient_temperature_system_failure.csv')
data=preprocess_data(df)
a_isf=isolation_forest(data,df)
a_svm=one_class_svm(data,df)

app = dash.Dash(__name__)

app.layout =dash_table.DataTable(
        id='table',
        columns=[{"name": i, "id": i} for i in df.columns],
        data=df.iloc[:1000].to_dict('records'),
        style_table={
            'maxHeight': '300px',
            'overflowY': 'scroll'
        },
        fixed_rows={ 'headers': True, 'data': 0 }
    )

app.layout = html.Div(children=[
        html.H1(children='Predictive Maintenance'),
        
        dash_table.DataTable(
            id='table',
            columns=[{"name": i, "id": i} for i in df.columns],
            data=df.to_dict('records'),
            style_table={
                'maxHeight': '300px',
                'overflowY': 'scroll'
            },
            fixed_rows={ 'headers': True, 'data': 0 }
        ),
  
        dcc.Dropdown(
            id='dropdown-1',
            options=[
                {'label': 'Isolation Forest', 'value': 'ISF'},
                {'label': 'One-Class SVM', 'value': 'SVM'},
            ],
            value='ISF'
        ), 
                
       dcc.Graph(id='graph-with-drowdown-1'),  
       
       dcc.Dropdown(
            id='dropdown-2',
            options=[
                {'label': 'Isolation Forest', 'value': 'ISF'},
                {'label': 'One-Class SVM', 'value': 'SVM'},
            ],
            value='ISF'
        ), 
                
       dcc.Graph(id='graph-with-drowdown-2')  
         
  
])
        
@app.callback(
    Output('graph-with-drowdown-1', 'figure'),
    [Input('dropdown-1', 'value')])
def update_figure_1(model):
    
    if model=='ISF':
        a=a_isf
    elif model=='SVM':
        a=a_svm
                    
    return {
        'data': [
                dict(
                    x=df['timestamp'],
                    y=df['value'],
                    name='measurement'
                ),
                               
                
                dict(
                    x=a['timestamp'],
                    y=a['value'],
                    mode='markers',
                    opacity=0.5,
                    marker={
                        'size': 8,
                        'line': {'width': 0.5, 'color': 'red'}
                    },
                    name='anomaly'
                ) 
 
            ],
            
            'layout': dict(
                xaxis={'title': 'time'},
                yaxis={'title': 'value'},
                margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
                legend={'x': 0, 'y': 1},
                hovermode='closest'
            )
    }
        
@app.callback(
    Output('graph-with-drowdown-2', 'figure'),
    [Input('dropdown-2', 'value')])
def update_figure_2(model):
    
    if model=='ISF':
        a=a_isf
    elif model=='SVM':
        a=a_svm
                    
    return {
        'data': [
                dict(
                    x=df['timestamp'],
                    y=df['value'],
                    name='measurement'
                ),
                               
                
                dict(
                    x=a['timestamp'],
                    y=a['value'],
                    mode='markers',
                    opacity=0.5,
                    marker={
                        'size': 8,
                        'line': {'width': 0.5, 'color': 'red'}
                    },
                    name='anomaly'
                ) 
 
            ],
            
            'layout': dict(
                xaxis={'title': 'time'},
                yaxis={'title': 'value'},
                margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
                legend={'x': 0, 'y': 1},
                hovermode='closest'
            )
    }

if __name__ == '__main__':
    app.run_server(debug=True)
    
