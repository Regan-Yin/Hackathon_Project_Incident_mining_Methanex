import os
from dotenv import load_dotenv

load_dotenv()

os.environ['DASH_HOT_RELOAD'] = 'False'

# ADD YOUR OWN VALUE: Path to your GCP service account key file.
# Set GOOGLE_APPLICATION_CREDENTIALS in your .env file (see .env.example).
# Default: ./gcp-key.json (relative to project root)
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.getenv(
    'GOOGLE_APPLICATION_CREDENTIALS', './gcp-key.json'
)

import dash
from dash import dcc, html, Input, Output, State, dash_table, no_update
import pandas as pd
import numpy as np # Add numpy for logarithmic calculations
import plotly.express as px
import plotly.graph_objects as go
from rag_engine import analyze_new_event

# --- METHANEX CORPORATE COLOR PALETTE ---
METHANEX_PALETTE = ["#002C77", "#8CC63F", "#44A0C8", "#5D7B9D", "#C4D6B0"]

# --- DATA LOADING ---
def load_data():
    events = pd.read_csv("data/events_clean.csv")
    actions = pd.read_csv("data/actions_clean.csv")
    clusters = pd.read_csv("data/case_cluster_map.csv")
    events = events.merge(clusters[['event_id', 'cluster_name']], on='event_id', how='left')
    return events, actions

events_df, actions_df = load_data()
cluster_options = [{'label': c, 'value': c} for c in events_df['cluster_name'].dropna().unique()]

# --- HELPER FUNCTIONS ---
def apply_corporate_theme(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#333333", family="'Open Sans', Arial"),
        margin=dict(t=50, b=30, l=30, r=20),
        colorway=METHANEX_PALETTE
    )
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#EAEAEA', zeroline=False)
    fig.update_xaxes(showgrid=False, zeroline=False)
    return fig

def calc_priority(r, s):
    """Calculate Priority Score based on Risk and Severity"""
    r_str = str(r).lower()
    s_str = str(s).lower()
    r_score = 2 if 'high' in r_str else (1 if 'medium' in r_str else 0)
    s_score = 3 if ('major' in s_str or 'serious' in s_str) else (2 if 'significant' in s_str else 1)
    return r_score + s_score

# --- DASH APP INITIALIZATION ---
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "Methanex EPSSC"
server = app.server

# --- JAVASCRIPT ANIMATION ENGINE: KPI COUNTERS ---
app.clientside_callback(
    """
    function(total, incPct, convPct, highRisk, severe) {
        window.kpi_raf_ids = window.kpi_raf_ids || {};
        
        function animateValue(id, start, endRaw, duration, isPercent) {
            let obj = document.getElementById(id);
            if (!obj) return;
            let end = parseFloat(endRaw) || 0;
            
            if (window.kpi_raf_ids[id]) {
                window.cancelAnimationFrame(window.kpi_raf_ids[id]);
            }
            
            let startTime = null;
            const step = (timestamp) => {
                if (!startTime) startTime = timestamp;
                const progress = Math.min((timestamp - startTime) / duration, 1);
                const easeProgress = 1 - Math.pow(1 - progress, 3); 
                let current = easeProgress * (end - start) + start;
                
                if (isPercent) obj.innerHTML = current.toFixed(1) + "%";
                else obj.innerHTML = Math.floor(current);
                
                if (progress < 1) {
                    window.kpi_raf_ids[id] = window.requestAnimationFrame(step);
                } else {
                    if (isPercent) obj.innerHTML = end.toFixed(1) + "%";
                    else obj.innerHTML = Math.floor(end);
                    delete window.kpi_raf_ids[id];
                }
            };
            window.kpi_raf_ids[id] = window.requestAnimationFrame(step);
        }

        window.kpi_prev = window.kpi_prev || [0, 0, 0, 0, 0];
        let v = [parseFloat(total)||0, parseFloat(incPct)||0, parseFloat(convPct)||0, parseFloat(highRisk)||0, parseFloat(severe)||0];

        animateValue("kpi-total-val", window.kpi_prev[0], v[0], 1200, false);
        animateValue("kpi-inc-pct-val", window.kpi_prev[1], v[1], 1200, true);
        animateValue("kpi-conversion-val", window.kpi_prev[2], v[2], 1200, true);
        animateValue("kpi-high-risk-val", window.kpi_prev[3], v[3], 1200, true);
        animateValue("kpi-severe-val", window.kpi_prev[4], v[4], 1200, true);

        window.kpi_prev = v;
        return window.dash_clientside.no_update;
    }
    """,
    Output('dummy-kpi-output', 'children'),
    [Input('kpi-total-store', 'data'), Input('kpi-inc-pct-store', 'data'), Input('kpi-conversion-store', 'data'), Input('kpi-high-risk-store', 'data'), Input('kpi-severe-store', 'data')]
)

# --- JAVASCRIPT ANIMATION ENGINE: CLUSTER PROFILE COUNTERS ---
app.clientside_callback(
    """
    function(cases, rate, actions, active_tab) {
        if (active_tab !== 'perf-clusters') return window.dash_clientside.no_update;
        
        window.prof_raf_ids = window.prof_raf_ids || {};

        function animateValue(id, start, endRaw, duration, isPercent) {
            let obj = document.getElementById(id);
            if (!obj) return;
            let end = parseFloat(endRaw);
            if (isNaN(end) || endRaw === "N/A") {
                obj.innerHTML = endRaw;
                return;
            }
            
            if (window.prof_raf_ids[id]) {
                window.cancelAnimationFrame(window.prof_raf_ids[id]);
            }

            let startTime = null;
            const step = (timestamp) => {
                if (!startTime) startTime = timestamp;
                const progress = Math.min((timestamp - startTime) / duration, 1);
                const easeProgress = 1 - Math.pow(1 - progress, 3); 
                let current = easeProgress * (end - start) + start;
                
                if (isPercent) obj.innerHTML = current.toFixed(1) + "%";
                else obj.innerHTML = Math.floor(current);
                
                if (progress < 1) {
                    window.prof_raf_ids[id] = window.requestAnimationFrame(step);
                } else {
                    if (isPercent) obj.innerHTML = end.toFixed(1) + "%";
                    else obj.innerHTML = Math.floor(end);
                    delete window.prof_raf_ids[id];
                }
            };
            window.prof_raf_ids[id] = window.requestAnimationFrame(step);
        }

        window.prof_prev = window.prof_prev || [0, 0, 0];
        
        let c_val = parseFloat(cases); let c_na = isNaN(c_val);
        let r_val = parseFloat(rate);  let r_na = isNaN(r_val);
        let a_val = parseFloat(actions); let a_na = isNaN(a_val);

        animateValue("prof-cases-val", window.prof_prev[0], cases, 1000, false);
        animateValue("prof-rate-val", window.prof_prev[1], rate, 1000, true);
        animateValue("prof-actions-val", window.prof_prev[2], actions, 1000, false);

        window.prof_prev = [c_na?0:c_val, r_na?0:r_val, a_na?0:a_val];
        return window.dash_clientside.no_update;
    }
    """,
    Output('dummy-prof-output', 'children'),
    [Input('prof-cases-store', 'data'), Input('prof-rate-store', 'data'), Input('prof-actions-store', 'data'), Input('perf-subtabs', 'value')]
)

# --- JAVASCRIPT ANIMATION ENGINE: AI TYPEWRITER EFFECT ---
app.clientside_callback(
    """
    function(trigger_val, n_intervals, full_text) {
        const ctx = window.dash_clientside.callback_context;
        if (!full_text || !ctx.triggered) return ["", true];
        
        const trigger_id = ctx.triggered[0].prop_id;
        
        if (trigger_id === "ai-typewriter-trigger.data") {
            window.tw_idx = 0;
            let container = document.getElementById("ai-output-container");
            let expander = document.getElementById("top10-expander-container");
            if (container) container.style.display = "block";
            if (expander) expander.style.opacity = "0"; 
            return ["", false]; 
        }
        
        window.tw_idx = (window.tw_idx || 0) + 40; 
        
        if (window.tw_idx >= full_text.length) {
            let expander = document.getElementById("top10-expander-container");
            if (expander) {
                expander.style.transition = "opacity 0.8s ease-in";
                expander.style.opacity = "1"; 
            }
            return [full_text, true]; 
        }
        
        return [full_text.slice(0, window.tw_idx), false];
    }
    """,
    [Output('ai-output-markdown', 'children'), Output('ai-interval', 'disabled')],
    [Input('ai-typewriter-trigger', 'data'), Input('ai-interval', 'n_intervals')],
    [State('ai-full-text-store', 'data')]
)

# --- LAYOUT ---
app.layout = html.Div([
    
    html.Div(id='dummy-kpi-output', style={'display': 'none'}),
    html.Div(id='dummy-prof-output', style={'display': 'none'}),
    dcc.Store(id='top10-events-store'),
    dcc.Store(id='kpi-total-store'), dcc.Store(id='kpi-inc-pct-store'), dcc.Store(id='kpi-conversion-store'), dcc.Store(id='kpi-high-risk-store'), dcc.Store(id='kpi-severe-store'),
    dcc.Store(id='prof-cases-store'), dcc.Store(id='prof-rate-store'), dcc.Store(id='prof-actions-store'),
    dcc.Store(id='ai-full-text-store'), dcc.Store(id='ai-typewriter-trigger', data=0),
    dcc.Interval(id='ai-interval', interval=30, max_intervals=-1, disabled=True), 

    # 1. Sticky Header 
    html.Div([
        html.Img(src="/assets/logo.svg", alt="Methanex logo", style={"height": "44px", "marginRight": "14px", "verticalAlign": "middle"}),
        html.H1("Executive Process Safety Steering Committee (EPSSC)", className="header-title"),
    ], className="header-container"),

    # Main Content Container
    html.Div([
        # 2. Filters 
        html.Div([
            html.H3("Dashboard Filters", style={'marginTop': '0', 'color': '#002C77'}),
            html.Div([
                html.Div([
                    html.Label("Year Range (2019-2024)", style={'fontWeight': 'bold'}),
                    dcc.RangeSlider(id='year-slider', min=2019, max=2024, step=1, marks={i: str(i) for i in range(2019, 2025)}, value=[2019, 2024], tooltip={"placement": "bottom", "always_visible": True})
                ], className="filter-item"),
                html.Div([
                    html.Label("Filter by Cluster", style={'fontWeight': 'bold'}),
                    dcc.Dropdown(id='cluster-dropdown', options=cluster_options, multi=True, placeholder="Select specific clusters...")
                ], className="filter-item"),
                html.Div([
                    dcc.Checklist(id='high-risk-toggle', options=[{'label': ' Show High Risk Only', 'value': 'High'}], value=[], style={'fontWeight': 'bold', 'marginTop': '25px', 'fontSize':'16px'})
                ], className="filter-item", style={'flex': '0.5'})
            ], className="filter-row"),
        ], className="graph-card", style={'position': 'relative', 'zIndex': 9999, 'overflow': 'visible'}), 

        # 3. KPI Tiles 
        html.Div([
            html.Div([html.Div("Total Cases", className="kpi-title"), html.Div("0", id="kpi-total-val", className="kpi-value")], className="kpi-card"),
            html.Div([html.Div("Incident Volume", className="kpi-title"), html.Div("0.0%", id="kpi-inc-pct-val", className="kpi-value")], className="kpi-card"),
            html.Div([html.Div("Near Miss Conversion", className="kpi-title"), html.Div("0.0%", id="kpi-conversion-val", className="kpi-value")], className="kpi-card"),
            html.Div([html.Div("High Risk Volume", className="kpi-title"), html.Div("0.0%", id="kpi-high-risk-val", className="kpi-value")], className="kpi-card"),
            html.Div([html.Div("Severe Cases", className="kpi-title"), html.Div("0.0%", id="kpi-severe-val", className="kpi-value")], className="kpi-card"),
        ], className="kpi-row", style={'position': 'relative', 'zIndex': 1}),

        # 4. Tabs
        dcc.Tabs(id='tabs-container', className='custom-tabs', value='tab-1', children=[

            # --- TAB 1: Performance Dashboard ---
            dcc.Tab(label='Performance Dashboard', value='tab-1', className='custom-tab', selected_className='custom-tab--selected', children=[
                html.Div([
                    dcc.Tabs(
                        id="perf-subtabs", value="perf-overview", className="custom-tabs",
                        children=[
                            dcc.Tab(label="Overview", value="perf-overview", className="custom-tab", selected_className="custom-tab--selected",
                                children=[
                                    html.Div([
                                        html.Div([
                                            html.Div([dcc.Graph(id='perf-fig-cat')], className="graph-card", style={'flex': '1'}),
                                            html.Div([dcc.Graph(id='perf-fig-risk')], className="graph-card", style={'flex': '1'}),
                                        ], style={'display': 'flex', 'gap': '20px'}),

                                        html.Div([
                                            html.Div([dcc.Graph(id='perf-fig-year')], className="graph-card", style={'flex': '1'}),
                                            html.Div([dcc.Graph(id='perf-fig-year-stack')], className="graph-card", style={'flex': '1'}),
                                        ], style={'display': 'flex', 'gap': '20px'}),

                                        html.Div([
                                            html.Div([dcc.Graph(id='perf-fig-pc-stack')], className="graph-card", style={'flex': '1'}),
                                        ], style={'display': 'flex', 'gap': '20px'}),

                                        html.Div([
                                            html.Div([
                                                html.Div([
                                                    html.H4("Cases table (filtered)", style={'color': '#002C77', 'display': 'inline-block', 'marginRight': '20px'}),
                                                    html.Button('Export Filtered Events to CSV', id='perf-export-events-btn', n_clicks=0, className='methanex-btn-cyan', style={'float': 'right'})
                                                ], style={'overflow': 'hidden', 'marginBottom': '15px'}),
                                                dcc.Download(id="perf-download-events-csv"),
                                                dash_table.DataTable(
                                                    id='perf-tbl-events', page_size=15, sort_action="native", filter_action="native",
                                                    style_table={'overflowX': 'auto', 'minWidth': '100%'}, 
                                                    style_header={'backgroundColor': '#002C77', 'color': 'white', 'fontWeight': 'bold'},
                                                    style_cell={'textAlign': 'left', 'padding': '10px', 'fontFamily': 'Open Sans', 'whiteSpace': 'normal', 'height': 'auto'},
                                                    style_data_conditional=[{'if': {'row_index': 'odd'}, 'backgroundColor': '#F4F6F9'}],
                                                )
                                            ], className="table-card", style={'width': '100%'})
                                        ]),
                                    ], style={'paddingTop': '10px'})
                                ],
                            ),
                            dcc.Tab(label="Clusters", value="perf-clusters", className="custom-tab", selected_className="custom-tab--selected",
                                children=[
                                    html.Div([
                                        html.Div([html.Div(id="perf-clusters-alert", className="graph-card", style={'flex': '1'})], style={'display': 'flex', 'gap': '20px'}),
                                        html.Div([html.Div([dcc.Graph(id='perf-fig-cluster-sizes')], className="graph-card", style={'flex': '1'})], style={'display': 'flex', 'gap': '20px'}),
                                        html.Div([
                                            html.Div([
                                                html.Label("Select a cluster", style={'fontWeight': 'bold'}),
                                                dcc.Dropdown(id="perf-cluster-pick", placeholder="Select a cluster..."),
                                                html.Div(id="perf-cluster-key-counts", style={'marginTop': '10px'}),
                                                dcc.Graph(id="perf-fig-cluster-donut"),
                                            ], className="graph-card", style={'flex': '1'}),

                                            html.Div([
                                                html.H5("Cluster AI Insights", style={'color': '#002C77', 'marginBottom': '20px'}),
                                                html.H6("Key Terms Triggering this Cluster", style={'color': '#666666'}),
                                                html.Div(id="perf-cluster-top-terms"),
                                                html.Hr(style={'borderColor': '#EAEAEA', 'margin': '25px 0'}),
                                                html.H6("ESG Performance Metrics", style={'color': '#666666', 'marginBottom': '15px'}),
                                                
                                                html.Div([
                                                    html.Div([
                                                        html.Div("0", id="prof-cases-val", className="prof-val"),
                                                        html.Div("Total Cases", className="prof-label")
                                                    ], className="prof-card"),
                                                    html.Div([
                                                        html.Div("0.0%", id="prof-rate-val", className="prof-val"),
                                                        html.Div("Incident Rate", className="prof-label")
                                                    ], className="prof-card"),
                                                    html.Div([
                                                        html.Div("0", id="prof-actions-val", className="prof-val"),
                                                        html.Div("Registered Actions", className="prof-label")
                                                    ], className="prof-card"),
                                                ], className="prof-grid"),
                                                
                                                html.Div(id="perf-cluster-prof-text"), 
                                            ], className="graph-card", style={'flex': '1'}),
                                        ], style={'display': 'flex', 'gap': '20px'}),
                                        
                                        # --- NEW: EARLY WARNING DASHBOARD SECTION ---
                                        html.Hr(style={'borderColor': '#EAEAEA', 'margin': '40px 0 30px 0'}),
                                        html.H4("Systemic Risk & Early Warning Dashboard", style={'color': '#002C77', 'marginBottom': '10px'}),
                                        dcc.Markdown(
                                            "**Early Warning Index** $= (\\text{Near miss rate}) \\times (\\text{High-priority share within near misses}) \\times \\log(1 + \\text{No.Cases})$", 
                                            mathjax=True, 
                                            style={'color': '#666666', 'fontSize': '14px', 'margin': '0'}
                                        ),
                                        html.P("This dynamic metric prioritizes clusters where near misses are frequent, serious, and occur at a meaningful scale.", style={'color': '#666666', 'fontSize': '13px', 'marginBottom': '25px'}),
                                        
                                        html.Div([
                                            html.Div([dcc.Graph(id='perf-fig-ewi')], className="graph-card", style={'flex': '1'}),
                                        ], style={'display': 'flex', 'gap': '20px'}),
                                        
                                        html.Div([
                                            html.Div([dcc.Graph(id='perf-fig-counts-heat')], className="graph-card", style={'flex': '1', 'minWidth': '300px'}),
                                            html.Div([dcc.Graph(id='perf-fig-rates-heat')], className="graph-card", style={'flex': '1', 'minWidth': '300px'}),
                                        ], style={'display': 'flex', 'gap': '20px', 'marginTop': '10px'})

                                    ], style={'paddingTop': '10px'})
                                ],
                            ),
                        ],
                    ),
                ], style={'paddingTop': '10px'})
            ]),

            # --- TAB 2: Event Intelligence ---
            dcc.Tab(label='Event Intelligence', value='tab-2', className='custom-tab', selected_className='custom-tab--selected', children=[
                html.Div([
                    html.Div([
                        html.H4("Executive Data Portal", style={'color': '#002C77', 'display': 'inline-block', 'marginRight': '20px'}),
                        html.Button('Export to CSV', id='export-csv-btn', n_clicks=0, className='methanex-btn-cyan', style={'float': 'right'})
                    ], style={'overflow': 'hidden', 'marginBottom': '15px'}),
                    dcc.Download(id="download-csv"),
                    dash_table.DataTable(
                        id='drilldown-table',
                        columns=[{"name": i.capitalize().replace("_", " "), "id": i} for i in ['title', 'year', 'risk_level', 'severity', 'category_type']],
                        page_size=15,
                        style_table={'overflowX': 'auto', 'minWidth': '100%'}, 
                        style_header={'backgroundColor': '#002C77', 'color': 'white', 'fontWeight': 'bold'},
                        style_cell={'textAlign': 'left', 'padding': '10px', 'fontFamily': 'Open Sans', 'whiteSpace': 'normal', 'height': 'auto'},
                        style_data_conditional=[{'if': {'row_index': 'odd'}, 'backgroundColor': '#F4F6F9'}],
                        filter_action="native",
                        sort_action="native",
                    )
                ], className="table-card")
            ]),

            # --- TAB 3: AI Safety Analyst ---
            dcc.Tab(label='AI Safety Analyst', value='tab-3', className='custom-tab', selected_className='custom-tab--selected', children=[
                html.Div([
                    html.H4("Generative AI Safety Analyst", style={'color': '#002C77'}),
                    html.P("Query the historical 2019-2024 Methanex knowledge base to predict risk and formulate actions for new hazard reports."),
                    
                    dcc.Textarea(
                        id='ai-input',
                        placeholder='Describe the incident you want to analyze. It can be keywords, what happened, or the complete report. More specific details lead to better retrieval and analysis of similar historical events.',
                        className='text-area'
                    ),
                    html.Button('Generate Analysis', id='ai-btn', n_clicks=0, className='methanex-btn'),

                    dcc.Download(id="download-top10-csv"),
                    
                    dcc.Loading(
                        id="loading-ai", type="dot", color="#8CC63F",
                        children=[
                            html.Div(id='ai-output-container', children=[
                                dcc.Markdown(id='ai-output-markdown')
                            ], style={'padding': '25px', 'backgroundColor': '#FFFFFF', 'borderLeft': '4px solid #8CC63F', 'marginBottom': '20px', 'borderRadius': '8px', 'boxShadow': '0 2px 10px rgba(0,0,0,0.03)', 'display': 'none'}),
                            
                            html.Div(id='top10-expander-container', style={'opacity': '0'})
                        ]
                    )
                ], className="graph-card")
            ]),
        ])
    ], className="main-container"),

    # 5. LEGAL FOOTER BANNER
    html.Div([
        html.Span("© Methanex Corporation 2026. All rights Reserved. |"),
        html.Span("Data belongs to UBC MBAn 2026 Hackathon Team 9. No guarantee of authenticity. For educational and competition purposes only. For commercial use, "),
        html.A("Contact Team 9", href="mailto:reganchy@student.ubc.ca", className="footer-link")
    ], className="footer-banner")
])

# --- HELPER LOGIC ---
def _first_col(df, candidates):
    for c in candidates:
        if c in df.columns: return c
    return None

def _safe_series(df, col, default="Unknown"):
    if col and col in df.columns: return df[col].fillna(default)
    return pd.Series([default] * len(df))

def _filtered_events(years, clusters, high_risk):
    filtered = events_df.copy()
    year_col = _first_col(filtered, ["year", "Year", "incident_year", "event_year"])
    if year_col:
        filtered[year_col] = pd.to_numeric(filtered[year_col], errors="coerce")
        filtered = filtered[(filtered[year_col] >= years[0]) & (filtered[year_col] <= years[1])]
    if clusters and "cluster_name" in filtered.columns:
        filtered = filtered[filtered["cluster_name"].isin(clusters)]
    if high_risk and ("High" in high_risk) and "risk_level" in filtered.columns:
        filtered = filtered[filtered["risk_level"] == "High"]
    return filtered

# --- CALLBACKS ---
@app.callback(
    Output("download-csv", "data"),
    Input('export-csv-btn', 'n_clicks'),
    [State('year-slider', 'value'), State('cluster-dropdown', 'value'), State('high-risk-toggle', 'value')],
    prevent_initial_call=True
)
def export_csv(n_clicks, years, clusters, high_risk):
    filtered = _filtered_events(years, clusters, high_risk)
    if len(filtered) == 0: return None
    return dcc.send_data_frame(filtered.to_csv, "filtered_events.csv", index=False)

@app.callback(
    [
        Output('kpi-total-store', 'data'), Output('kpi-inc-pct-store', 'data'),
        Output('kpi-conversion-store', 'data'), Output('kpi-high-risk-store', 'data'), Output('kpi-severe-store', 'data'),
        Output('perf-fig-cat', 'figure'), Output('perf-fig-risk', 'figure'), Output('perf-fig-year', 'figure'),
        Output('perf-fig-year-stack', 'figure'), Output('perf-fig-pc-stack', 'figure'),
        
        # New Outputs: Early Warning & Heatmaps
        Output('perf-fig-ewi', 'figure'), Output('perf-fig-counts-heat', 'figure'), Output('perf-fig-rates-heat', 'figure'),
        
        Output('perf-tbl-events', 'data'), Output('perf-tbl-events', 'columns'),
        Output('drilldown-table', 'data'),
    ],
    [Input('year-slider', 'value'), Input('cluster-dropdown', 'value'), Input('high-risk-toggle', 'value')]
)
def update_dashboard(years, clusters, high_risk):
    filtered = _filtered_events(years, clusters, high_risk)

    total = len(filtered)
    cat_col = _first_col(filtered, ["category_type", "category", "type"])
    risk_col = _first_col(filtered, ["risk_level", "risk", "risklevel"])
    sev_col = _first_col(filtered, ["severity", "sev"])

    incidents = len(filtered[_safe_series(filtered, cat_col).astype(str).str.lower().eq("incident")]) if cat_col else 0
    near_misses = len(filtered[_safe_series(filtered, cat_col).astype(str).str.lower().isin(["near miss", "nearmiss", "near_miss"])]) if cat_col else 0

    kpi1 = total
    kpi2 = (incidents / total * 100) if total else 0
    kpi3 = (incidents / (incidents + near_misses) * 100) if (incidents + near_misses) else 0
    kpi4 = (len(filtered[_safe_series(filtered, risk_col).astype(str).str.lower().eq('high')]) / total * 100) if total else 0
    kpi5 = (len(filtered[_safe_series(filtered, sev_col).astype(str).str.lower().isin(['major','serious'])]) / total * 100) if total else 0

    if total == 0:
        empty_fig = apply_corporate_theme(go.Figure().update_layout(title="No Data Available"))
        return (kpi1, kpi2, kpi3, kpi4, kpi5, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, [], [], [])

    # Overview Charts (Restore stable no-animation mode)
    cat_series = _safe_series(filtered, cat_col, default="Unknown").astype(str) if cat_col else pd.Series(["Unknown"] * len(filtered))
    cat_df = cat_series.value_counts(dropna=False).reset_index()
    cat_df.columns = ["category_type", "count"]
    fig_cat = apply_corporate_theme(px.bar(cat_df, x="category_type", y="count", title="Cases by Category Type", color_discrete_sequence=[METHANEX_PALETTE[0]]))

    risk_series = _safe_series(filtered, risk_col, default="Unknown").astype(str) if risk_col else pd.Series(["Unknown"] * len(filtered))
    risk_df = risk_series.value_counts(dropna=False).reset_index()
    risk_df.columns = ["risk_level", "count"]
    fig_risk = apply_corporate_theme(px.bar(risk_df, x="risk_level", y="count", title="Cases by Risk Level", color_discrete_sequence=[METHANEX_PALETTE[1]]))

    year_col = _first_col(filtered, ["year", "Year", "incident_year", "event_year"])
    if year_col:
        yr_df = filtered.groupby(year_col).size().reset_index(name="count").sort_values(year_col)
        fig_year = px.line(yr_df, x=year_col, y="count", markers=True, title="Cases Over Time (Year)", color_discrete_sequence=[METHANEX_PALETTE[0]])
        fig_year.update_layout(xaxis=dict(dtick=1), xaxis_title="", yaxis_title="Cases")
        fig_year = apply_corporate_theme(fig_year)
    else:
        fig_year = apply_corporate_theme(px.scatter(title="Year column not found."))

    if year_col:
        stack_col = cat_col if cat_col else risk_col
        if stack_col:
            tmp = filtered.copy()
            tmp[stack_col] = _safe_series(tmp, stack_col, default="Unknown").astype(str)
            st_df = tmp.groupby([year_col, stack_col]).size().reset_index(name="count")
            fig_year_stack = px.bar(st_df, x=year_col, y="count", color=stack_col, barmode="stack", title=f"Cases by Year (stacked by {stack_col})", color_discrete_sequence=METHANEX_PALETTE)
            fig_year_stack.update_layout(xaxis=dict(dtick=1), xaxis_title="", yaxis_title="Cases")
            fig_year_stack = apply_corporate_theme(fig_year_stack)
        else:
            fig_year_stack = apply_corporate_theme(px.scatter(title="No category/risk column found for stacking."))
    else:
        fig_year_stack = apply_corporate_theme(px.scatter(title="Year column not found."))

    pc_col = _first_col(filtered, ["primary_classification", "primary_classification_clean", "primary_classification_name", "primary_classification_text"])
    if pc_col and cat_col:
        pc_tmp = filtered.copy()
        pc_tmp[pc_col] = _safe_series(pc_tmp, pc_col, default="Unknown").astype(str)
        pc_tmp[cat_col] = _safe_series(pc_tmp, cat_col, default="Unknown").astype(str)
        top_pc = pc_tmp[pc_col].value_counts().head(15).index.tolist()
        pc_tmp = pc_tmp[pc_tmp[pc_col].isin(top_pc)]
        pc_df = pc_tmp.groupby([pc_col, cat_col]).size().reset_index(name="count")
        fig_pc = px.bar(pc_df, x="count", y=pc_col, color=cat_col, orientation="h", title="Top Primary Classifications", color_discrete_sequence=METHANEX_PALETTE)
        fig_pc.update_layout(yaxis={'categoryorder': 'total ascending'}, xaxis_title="Cases", yaxis_title="")
        fig_pc = apply_corporate_theme(fig_pc)
    else:
        fig_pc = apply_corporate_theme(px.scatter(title="Primary classification column not found."))


    # --- NEW: EARLY WARNING INDEX & HEATMAPS LOGIC ---
    if "cluster_name" in filtered.columns and not filtered.empty:
        ewi_df = filtered.copy()
        
        # Calculate Priority Score dynamically
        ewi_df['priority_score'] = ewi_df.apply(lambda x: calc_priority(x.get(risk_col, ''), x.get(sev_col, '')), axis=1)
        ewi_df['is_hp'] = ewi_df['priority_score'] >= 4
        ewi_df['is_nm'] = ewi_df[cat_col].astype(str).str.lower().isin(['near miss', 'nearmiss', 'near_miss']) if cat_col else False
        ewi_df['is_inc'] = ewi_df[cat_col].astype(str).str.lower() == 'incident' if cat_col else False
        
        # Aggregate each Cluster
        g = ewi_df.groupby('cluster_name')
        agg = pd.DataFrame({
            'n_cases': g.size(),
            'n_incident': g['is_inc'].sum(),
            'n_near_miss': g['is_nm'].sum(),
            'n_high_priority': g['is_hp'].sum(),
            'hp_near_miss': g.apply(lambda x: (x['is_hp'] & x['is_nm']).sum()),
            'hp_incident': g.apply(lambda x: (x['is_hp'] & x['is_inc']).sum())
        }).reset_index()
        
        # Calculate Rates
        agg['near_miss_rate'] = np.where((agg['n_near_miss'] + agg['n_incident']) > 0, 
                                         agg['n_near_miss'] / (agg['n_near_miss'] + agg['n_incident']) * 100, 0)
        agg['hp_within_nm'] = np.where(agg['n_near_miss'] > 0, agg['hp_near_miss'] / agg['n_near_miss'] * 100, 0)
        agg['hp_within_inc'] = np.where(agg['n_incident'] > 0, agg['hp_incident'] / agg['n_incident'] * 100, 0)
        agg['hp_rate'] = np.where(agg['n_cases'] > 0, agg['n_high_priority'] / agg['n_cases'] * 100, 0)
        
        # Calculate core EWI formula
        agg['ewi'] = (agg['near_miss_rate']/100.0) * (agg['hp_within_nm']/100.0) * np.log1p(agg['n_cases'])
        
        # --- EWI Ranking Bar Chart ---
        agg = agg.sort_values('ewi', ascending=True) # Reverse order so highest bar is on top
        fig_ewi = px.bar(agg, x='ewi', y='cluster_name', orientation='h', 
                         title='Ranked Early Warning Index (Higher = More Urgent Focus)',
                         color_discrete_sequence=[METHANEX_PALETTE[2]]) # Use Cyan to match screenshot color
        fig_ewi = apply_corporate_theme(fig_ewi)
        fig_ewi.update_layout(xaxis_title="Early Warning Index", yaxis_title="", height=400)
        
        # --- Counts Heatmap ---
        c_cols = ['n_incident', 'n_near_miss', 'n_high_priority', 'hp_near_miss', 'hp_incident']
        c_labels = ['Incidents', 'Near Misses', 'High Priority (All)', 'High Priority (NM)', 'High Priority (Inc)']
        c_mat = agg.set_index('cluster_name')[c_cols]
        c_mat.columns = c_labels
        
        fig_counts = px.imshow(c_mat, text_auto=True, aspect="auto", color_continuous_scale="Blues", title="Counts by Cluster")
        fig_counts = apply_corporate_theme(fig_counts)
        fig_counts.update_layout(height=400, coloraxis_showscale=False) # Hide color bar on the right to keep it clean
        
        # --- Rates Heatmap ---
        r_cols = ['near_miss_rate', 'hp_rate', 'hp_within_nm', 'hp_within_inc']
        r_labels = ['Near Miss Rate', 'High Priority Rate', 'HP within NM', 'HP within Inc']
        r_mat = agg.set_index('cluster_name')[r_cols]
        r_mat.columns = r_labels
        
        fig_rates = px.imshow(r_mat, text_auto=".0f", aspect="auto", color_continuous_scale="YlOrRd", zmin=0, zmax=100, title="Rates by Cluster (%)")
        fig_rates = apply_corporate_theme(fig_rates)
        fig_rates.update_layout(height=400, coloraxis_showscale=False)
        
    else:
        empty = apply_corporate_theme(go.Figure().update_layout(title="No Cluster Data Available"))
        fig_ewi, fig_counts, fig_rates = empty, empty, empty


    preferred_cols = [c for c in ["event_id", "title", "year", "risk_level", "severity", "category_type", "cluster_name"] if c in filtered.columns]
    if not preferred_cols: preferred_cols = filtered.columns.tolist()[:10]
    tbl_df = filtered[preferred_cols].copy()
    tbl_data = tbl_df.to_dict("records")
    tbl_cols = [{"name": c.replace("_", " ").title(), "id": c} for c in preferred_cols]

    drill_cols = [c for c in ['title', 'year', 'risk_level', 'severity', 'category_type'] if c in filtered.columns]
    drill_data = filtered[drill_cols].to_dict('records') if drill_cols else []

    return (kpi1, kpi2, kpi3, kpi4, kpi5, fig_cat, fig_risk, fig_year, fig_year_stack, fig_pc, 
            fig_ewi, fig_counts, fig_rates, # Return the three new charts
            tbl_data, tbl_cols, drill_data)

def _load_optional_csv(path):
    try:
        if os.path.exists(path): return pd.read_csv(path)
    except Exception: pass
    return None

CLUSTER_SUMMARY_PATH = os.path.join("data", "cluster_summary_with_terms_examples.csv")
CLUSTER_PROFILE_PATH = os.path.join("data", "cluster_profile_sorted.csv")
cluster_summary_df = _load_optional_csv(CLUSTER_SUMMARY_PATH)
cluster_profile_df = _load_optional_csv(CLUSTER_PROFILE_PATH)

def _clusters_missing_files():
    missing = []
    for p in [CLUSTER_SUMMARY_PATH, CLUSTER_PROFILE_PATH]:
        if not os.path.exists(p): missing.append(p)
    return missing

@app.callback(
    [Output("perf-clusters-alert", "children"), Output("perf-fig-cluster-sizes", "figure"), Output("perf-cluster-pick", "options"), Output("perf-cluster-pick", "value")],
    [Input('year-slider', 'value'), Input('cluster-dropdown', 'value'), Input('high-risk-toggle', 'value')],
)
def init_clusters(years, clusters, high_risk):
    missing = _clusters_missing_files()
    if missing:
        alert = html.Div([
            html.H4("Missing file(s) for Clusters tab", style={'color': '#8B0000'}),
            html.Ul([html.Li(x) for x in missing]),
        ])
    else:
        alert = html.Div([html.H4("Clusters — Visual Summary", style={'color': '#002C77', 'margin': '0'})])

    filtered = _filtered_events(years, clusters, high_risk)
    if "cluster_name" in filtered.columns and len(filtered) > 0:
        size_df = filtered["cluster_name"].fillna("Unknown").value_counts().reset_index()
        size_df.columns = ["cluster_name", "n_cases"]
        size_df = size_df.head(30)
        fig_sizes = px.bar(size_df, x="n_cases", y="cluster_name", orientation="h", title="Top Clusters by Case Count", color_discrete_sequence=[METHANEX_PALETTE[0]])
        fig_sizes.update_layout(yaxis={'categoryorder': 'total ascending'})
        fig_sizes = apply_corporate_theme(fig_sizes)
        opts = [{"label": c, "value": c} for c in sorted(filtered["cluster_name"].dropna().unique())]
        default_val = opts[0]["value"] if opts else None
    else:
        fig_sizes = apply_corporate_theme(px.scatter(title="No cluster_name column found."))
        opts, default_val = [], None

    return alert, fig_sizes, opts, default_val

@app.callback(
    [Output("perf-cluster-key-counts", "children"), 
     Output("perf-fig-cluster-donut", "figure"), 
     Output("perf-cluster-top-terms", "children"),
     Output("prof-cases-store", "data"), Output("prof-rate-store", "data"), Output("prof-actions-store", "data"),
     Output("perf-cluster-prof-text", "children")], 
    [Input("perf-cluster-pick", "value"), Input('year-slider', 'value'), Input('cluster-dropdown', 'value'), Input('high-risk-toggle', 'value')],
)
def cluster_drilldown(cluster_pick, years, clusters, high_risk):
    if not cluster_pick:
        return "", apply_corporate_theme(go.Figure()), "", "N/A", "N/A", "N/A", html.Div()

    filtered = _filtered_events(years, clusters, high_risk)
    dfc = filtered[filtered["cluster_name"] == cluster_pick].copy()
    total = len(dfc)
    cat_col = _first_col(dfc, ["category_type", "category", "type"])
    risk_col = _first_col(dfc, ["risk_level", "risk", "risklevel"])
    sev_col = _first_col(dfc, ["severity", "sev"])

    incidents = len(dfc[_safe_series(dfc, cat_col).astype(str).str.lower().eq("incident")]) if cat_col else 0
    near_misses = len(dfc[_safe_series(dfc, cat_col).astype(str).str.lower().isin(["near miss", "nearmiss", "near_miss"])]) if cat_col else 0
    high_r = len(dfc[_safe_series(dfc, risk_col).astype(str).str.lower().eq("high")]) if risk_col else 0
    severe = len(dfc[_safe_series(dfc, sev_col).astype(str).str.lower().isin(["major","serious"])]) if sev_col else 0

    key_counts = html.Div([
        html.Div(f"Cases: {total}"), html.Div(f"Incidents: {incidents}"),
        html.Div(f"Near Misses: {near_misses}"), html.Div(f"High Risk: {high_r}"), html.Div(f"Major/Serious: {severe}"),
    ], style={'fontWeight': 'bold', 'color': '#666666'})

    cat_order = ["Incident", "Near Miss", "Other"]
    donut_counts = []
    if cat_col:
        for c in cat_order:
            if c == "Other":
                c_val = len(dfc[~_safe_series(dfc, cat_col).astype(str).str.lower().isin(["incident", "near miss", "nearmiss", "near_miss"])])
            else:
                c_val = len(dfc[_safe_series(dfc, cat_col).astype(str).str.lower().isin([c.lower(), c.lower().replace(" ", ""), c.lower().replace(" ", "_")])])
            donut_counts.append(c_val)
    donut_df = pd.DataFrame({"category_type": cat_order, "count": donut_counts})
    fig_donut = px.pie(donut_df, names="category_type", values="count", hole=0.45, title="Cluster composition", color_discrete_sequence=METHANEX_PALETTE)
    fig_donut = apply_corporate_theme(fig_donut)
    fig_donut.update_layout(transition=dict(duration=500, easing="cubic-in-out"), uirevision='constant')  

    terms_box = html.Div("(File not found)")
    if cluster_summary_df is not None:
        name_col = _first_col(cluster_summary_df, ["cluster_name", "cluster_label", "cluster"])
        terms_col = _first_col(cluster_summary_df, ["top_terms", "terms", "top_keywords"])
        ex_col = _first_col(cluster_summary_df, ["examples", "example_titles", "sample_titles"])
        if name_col:
            row = cluster_summary_df[cluster_summary_df[name_col].astype(str) == str(cluster_pick)]
            if len(row) > 0:
                row0 = row.iloc[0]
                terms = str(row0[terms_col]) if terms_col and terms_col in row.columns else ""
                examples = str(row0[ex_col]) if ex_col and ex_col in row.columns else ""
                
                terms_list = [t.strip() for t in terms.split(",") if t.strip()][:30]
                badges = html.Div(
                    [html.Span(t, className="term-badge", style={'animationDelay': f"{i*0.06}s"}) for i, t in enumerate(terms_list)] if terms_list else html.Div("(No terms found)"),
                    style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '8px'}
                )
                
                if examples: 
                    ex_list = [e.strip() for e in examples.split("|") if e.strip()]
                    ex_cards = html.Div([
                        html.Div([
                            html.Span("📄 Historical Incident Log", className="ex-tag"),
                            html.Span(ex, className="ex-text")
                        ], className="ex-item") for ex in ex_list
                    ])
                    terms_box = html.Div([badges, html.Hr(style={'borderColor': '#EAEAEA', 'margin': '20px 0'}), ex_cards])
                else:
                    terms_box = badges

    n_cases_val = "N/A"
    inc_rate_val = "N/A"
    n_actions_val = "N/A"
    prof_text_html = html.Div()

    if cluster_profile_df is not None:
        prof_name_col = _first_col(cluster_profile_df, ["cluster_name", "cluster_label", "cluster"])
        if prof_name_col:
            prow = cluster_profile_df[cluster_profile_df[prof_name_col].astype(str) == str(cluster_pick)]
            if len(prow) > 0:
                row_data = prow.iloc[0].to_dict()
                n_cases_val = str(row_data.get('n_cases', row_data.get('N Cases', total)))
                
                ir_raw = row_data.get('incident_rate', row_data.get('Incident Rate', 'N/A'))
                if ir_raw != 'N/A':
                    try: inc_rate_val = str(float(ir_raw) * 100)
                    except: inc_rate_val = str(ir_raw)

                n_actions_val = str(row_data.get('n_actions', row_data.get('N Actions', 'N/A')))

                cause_val = 'N/A'
                action_val = 'N/A'
                for k, v in row_data.items():
                    k_str = str(k).lower()
                    if 'cause' in k_str and 'theme' in k_str: cause_val = v
                    elif 'action' in k_str and 'theme' in k_str: action_val = v
                
                prof_text_html = html.Div([
                    html.Div([
                        html.H6("Top Root Cause Themes", style={'color': '#002C77', 'marginBottom': '5px'}),
                        html.Div(str(cause_val), style={'fontSize': '13px', 'color': '#666'})
                    ], className="prof-text-card"),
                    
                    html.Div([
                        html.H6("Top Corrective Action Themes", style={'color': '#002C77', 'marginBottom': '5px'}),
                        html.Div(str(action_val), style={'fontSize': '13px', 'color': '#666'})
                    ], className="prof-text-card")
                ])

    return key_counts, fig_donut, terms_box, n_cases_val, inc_rate_val, n_actions_val, prof_text_html

@app.callback(
    [Output('ai-full-text-store', 'data'), 
     Output('ai-typewriter-trigger', 'data'),
     Output('top10-expander-container', 'children'),
     Output('top10-events-store', 'data')],
    [Input('ai-btn', 'n_clicks')],
    [State('ai-input', 'value'), State('ai-typewriter-trigger', 'data')],
    prevent_initial_call=True
)
def run_ai(n_clicks, text, trigger_val):
    if not text:
        return "", dash.no_update, html.Div("Please enter a hypothetical event text.", style={'color': 'red', 'padding': '20px'}), None

    import vertexai
    from google.cloud import aiplatform

    # ADD YOUR OWN VALUES: Set GCP_PROJECT_ID and GCP_LOCATION in your .env file
    PROJECT_ID = os.getenv("GCP_PROJECT_ID")
    LOCATION = os.getenv("GCP_LOCATION", "us-west1")
    
    try:
        vertexai.init(project=PROJECT_ID, location=LOCATION)
        aiplatform.init(project=PROJECT_ID, location=LOCATION)

        response_text, top_10 = analyze_new_event(text, events_df)
        
        top_10_table = dash_table.DataTable(
            data=top_10[['title', 'cluster_name', 'risk_level', 'severity']].to_dict('records'),
            columns=[{"name": i.capitalize().replace("_", " "), "id": i} for i in ['title', 'cluster_name', 'risk_level', 'severity']],
            style_table={'overflowX': 'auto', 'minWidth': '100%'}, 
            style_header={'backgroundColor': '#002C77', 'color': 'white', 'fontWeight': 'bold'},
            style_cell={'textAlign': 'left', 'padding': '10px', 'fontFamily': 'Open Sans', 'whiteSpace': 'normal', 'height': 'auto'},
            style_data_conditional=[{'if': {'row_index': 'odd'}, 'backgroundColor': '#F4F6F9'}]
        )

        expander = html.Details([
            html.Summary("View Top 10 Similar Historical Events", className="methanex-btn-cyan", style={'marginTop': '30px', 'marginBottom': '10px', 'display': 'inline-block', 'cursor': 'pointer'}),
            html.Div([
                html.Div([
                    html.Button(
                        'Export Top 10 Context Events to CSV',
                        id='export-top10-csv-btn',
                        n_clicks=0,
                        className='methanex-btn-cyan',
                        style={'float': 'right', 'marginBottom': '15px', 'fontSize': '13px', 'padding': '8px 18px'}
                    )
                ], style={'overflow': 'hidden'}), 
                top_10_table
            ], style={'border': '1px solid #EAEAEA', 'borderRadius': '8px', 'padding': '20px', 'backgroundColor': '#FFFFFF', 'boxShadow': '0 2px 10px rgba(0,0,0,0.02)', 'marginTop': '10px'})
        ])

        top_10_json = top_10.to_json(date_format='iso', orient='split')
        new_trigger = (trigger_val or 0) + 1 

        return response_text, new_trigger, expander, top_10_json

    except Exception as e:
        return "", dash.no_update, html.Div(f"Error connecting to Vertex AI: {str(e)}", style={'color': 'red', 'padding': '20px'}), None

@app.callback(
    Output("download-top10-csv", "data"),
    Input('export-top10-csv-btn', 'n_clicks'),
    State('top10-events-store', 'data'),
    prevent_initial_call=True
)
def export_top10(n_clicks, stored_data):
    if not n_clicks or n_clicks == 0 or not stored_data: 
        return no_update
    df = pd.read_json(stored_data, orient='split')
    return dcc.send_data_frame(df.to_csv, "top_10_similar_events.csv", index=False)

if __name__ == '__main__':
    app.run(debug=True, port=8050)