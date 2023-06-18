import plotly.graph_objects as go

from app.src.get_data.connector_sql import PostgresConnector


class Chart:

    def __init__(self, connector):
        self.con = connector


class TeamChoiceChart(Chart):
    
    def __init__(self, connector, line_width=3):
        super().__init__(connector)
        self.line_width = line_width


    def _create_pitch(self): 
        line_width = self.line_width

        fig = go.Figure()

        fig.add_vrect(
            x0=-350, x1=350, # width=3
            line=dict(
                color='#4B5563',
                width=4
            )
        )

        fig.add_trace(go.Scatter(
            x=[-350, 350],
            y=[0, 0],
            marker=dict(size=25, color='#4B5563'),
            mode='lines',
            line=dict(
                color='#4B5563',
                width=line_width
            )
        ))

        fig.add_shape(type="circle",
            xref="x", yref="y",
            x0=-100, y0=-100, x1=100, y1=100,
            line=dict(
                color='#4B5563',
                width=line_width-1
            )
        )

        fig.add_trace(
            go.Scatter(
                x=[-180, -180, 180, 180],
                y=[-550, -400,  -400, -550, ],
                mode='lines',
                line_color="#4B5563",
                showlegend=False,
                line=dict(
                    color='#4B5563',
                    width=line_width
                )
            )
        )

        fig.add_trace(
            go.Scatter(
                x=[-180, -180, 180, 180],
                y=[550, 400,  400, 550, ],
                mode='lines',
                line_color="#4B5563",
                showlegend=False,
                line=dict(
                    color='#4B5563',
                    width=line_width
                )
            )
        )
        
        fig.update_layout(
            font_family="sans-serif",
            autosize=False,
            width=600,
            height=800,
            yaxis_range=[-550,550],
            xaxis_range=[-400,400],
            showlegend=False,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )

        fig.update_xaxes(visible=False)  
        fig.update_yaxes(visible=False)

        return fig


    def _plot_players(self, first_xi, position, fig):
        df = first_xi.loc[first_xi.position == position]

        Y = {
            'FWD': 350,
            'MID': 75,
            'DEF': -250,
            'GK': -475
        }

        dims = [-250, 250]
        width = dims[1] - dims[0]
        
        if len(df) > 1:
            divisor = len(df) - 1
            jump = int(width/divisor)
            X = list(range(dims[0], dims[1]+1, jump) )
        else:
            X = [0]
        
        fig.add_trace(
            go.Scatter(
                x=X,
                y=[Y[position]]*len(df),
                mode='markers+text',
                marker=dict(
                    size=25,
                    color='#4B5563'
                ),  
                
                text=df['surname'],
                textposition='bottom center',
                textfont=dict(color='#4B5563'),
                textfont_size=10,
                
                customdata=df[['agg_win_odds', 'team_name', 'total_points', 'next_opp', 'name', 'value']],
            
                hovertemplate=
                    '<b>%{customdata[4]}</b>' + 
                    '<br><br><b>Total points:</b> %{customdata[2]}' + 
                    '<br><b>Price:</b> %{customdata[5]}' +
                    '<br><b>Team:</b> %{customdata[1]}' + 
                    '<br><b>Next opponent:</b> %{customdata[3]}' + 
                    '<br><b>Odds of a win in next GW:</b> %{customdata[0]:.0%}'  +  '<extra></extra>'
            )
        )
        
        return fig


    def _plot_subs(self, team, fig):
        
        df = team.loc[team.first_xi == 0]
        
        X = [375]*4
        Y = [-90, -30, 30, 90 ]
            
        fig.add_trace(
            go.Scatter(
                x=X,
                y=Y,
                mode='markers',
                marker=dict(
                    size=25,
                    color='#4B5563'
                ),  
                
                text=df['name'],
                textposition='bottom left',
                textfont=dict(color='#4B5563'),
                textfont_size=10,
                
                customdata=df[['agg_win_odds', 'team_name', 'total_points', 'next_opp', 'name', 'value']],
                
                hovertemplate=
                    '<b>%{text}</b>' + 
                    '<br><br><b>Total points:</b> %{customdata[2]}' + 
                    '<br><b>Price:</b> %{customdata[5]}' +
                    '<br><b>Team:</b> %{customdata[1]}' + 
                    '<br><b>Next opponent:</b> %{customdata[3]}' + 
                    '<br><b>Odds of a win in next GW:</b> %{customdata[0]:.0%}'  +  '<extra></extra>'
            )
        )
        return fig


    def create_plot(self, gameweek, season=23):
        
        con = self.con
        try:
            players = con.get_team(gameweek, season)
        except Exception:
            players = con.get_team(gameweek-1, season)

        team = players[players['picked'] == 1]
        first_xi = team.loc[team['first_xi'] == 1]

        fig = self._create_pitch()
        for pos in ['FWD', 'MID', 'DEF', 'GK']:
            self._plot_players(first_xi, pos, fig)

        fig = self._plot_subs(team, fig)

        return fig
    

class PlayerValueChart(Chart):

    def __init__(self, connector: PostgresConnector, min_games=2, season=23):
        super().__init__(connector)
        self.min_games = min_games
        self.season = season


    def plot_value(self, gameweek, season=23):

        con = self.con
        try:
            collapsed_stats = con.get_team(gameweek, season)
        except Exception:
            collapsed_stats = con.get_team(gameweek-1, season)
        
        collapsed_stats['Season Value'] = collapsed_stats.total_points / collapsed_stats.value
        collapsed_stats = collapsed_stats[collapsed_stats['minutes']/90 > self.min_games] 

        team_players = collapsed_stats[collapsed_stats['picked'] == 1]
        non_team_players = collapsed_stats[collapsed_stats['picked'] != 1]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=non_team_players.value,
            y=non_team_players['Season Value'],
            marker=dict(
                color='#9fbbe3', 
            ),
            mode='markers',
            
            customdata=non_team_players[['agg_win_odds', 'team_name', 'total_points', 'next_opp', 'name', 'value']],

            hovertemplate=
                '<b>%{customdata[4]}</b>' + 
                '<br><br><b>Total points:</b> %{customdata[2]}' + 
                '<br><b>Price:</b> %{customdata[5]}' +
                '<br><b>Team:</b> %{customdata[1]}'  +  '<extra></extra>'
            
        ))

        fig.add_trace(go.Scatter(
            x=team_players.value,
            y=team_players['Season Value'],
            marker=dict(
                color='#4B5563', 
            ),
            mode='markers',
            customdata=team_players[['agg_win_odds', 'team_name', 'total_points', 'next_opp', 'name', 'value']],

            hovertemplate=
                '<b>%{customdata[4]}</b>' + 
                '<br><br><b>Total points:</b> %{customdata[2]}' + 
                '<br><b>Price:</b> %{customdata[5]}' +
                '<br><b>Team:</b> %{customdata[1]}' +  '<extra></extra>'
        ))

        fig.update_layout(
            autosize=False,
            width=700,
            height=800,
            showlegend=False,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis_title="Player cost",
            yaxis_title="Cost/Points",
            yaxis_visible=True, 
            yaxis_showticklabels=False,
            xaxis_visible=True, 
            xaxis_showticklabels=False,
            font=dict(
                family="sans-serif",
                color="#4B5563"
            )
        )

        fig.add_annotation(text="Minimum 2 games. Dark points show players picked for the upcoming gameweek.",
                        xref="paper", yref="paper",
                        x=0, y=-0.1, showarrow=False)

        return fig
