import requests
import urllib.request
import numpy as np
from scipy.stats import binned_statistic_2d
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Arc
import seaborn as sns
from bokeh.plotting import figure, ColumnDataSource
from math import pi


sns.set_style('white')
sns.set_color_codes()


class NoPlayerError(Exception):
    """Custom Exception for invalid player search in get_player_id()"""
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)


class Players:
    """
    Players containts a pandas DataFrame with all players that have shot chart
    data.

    When a Players object is instantiated, the DataFrame is automatically
    loaded into memory.
    """
    def __init__(self):
        self.players_df = pd.read_csv("players2001.csv")

    def get_player_id(self, name):
        """
        Returns the given player's player id used in the NBA stats API as a
        numpy array.

        To extract the player id you must index it as you would a numpy array.
        Note that there are some players that have the same name, so this
        results in a an array with multiple elements being returned.

        Parameters
        ----------

        name : string
            Name in 'Last Name, First Name' format of the player whose ID we
            want.
        """
        player_id = self.players_df[self.players_df.name == name].player_id
        # May be able to use ValueError instead
        if len(player_id) == 0:
            raise NoPlayerError('There is no player with that name.')
        return player_id.values


class Shots:
    """
    Shots is a wrapper around the NBA stats API that can access the shot chart
    data and player image.
    """
    def __init__(self, player_id, league_id="00", season="2014-15",
                 season_type="Regular Season", team_id=0, game_id="",
                 outcome="", location="", month=0, season_segment="",
                 date_from="", date_to="", opp_team_id=0, vs_conference="",
                 vs_division="", position="", rookie_year="", game_segment="",
                 period=0, last_n_games=0, clutch_time="", ahead_behind="",
                 point_diff="", range_type="", start_period="", end_period="",
                 start_range="", end_range="", context_filter="",
                 context_measure="FGA"):

        self.player_id = player_id

        self.base_url = "http://stats.nba.com/stats/shotchartdetail?"

        # TODO: Figure out what all these parameters mean for NBA stats api
        self.url_paramaters = {
                                "LeagueID": league_id,
                                "Season": season,
                                "SeasonType": season_type,
                                "TeamID": team_id,
                                "PlayerID": player_id,
                                "GameID": game_id,
                                "Outcome": outcome,
                                "Location": location,
                                "Month": month,
                                "SeasonSegment": season_segment,
                                "DateFrom": date_from,
                                "DateTo": date_to,
                                "OpponentTeamID": opp_team_id,
                                "VsConference": vs_conference,
                                "VsDivision": vs_division,
                                "Position": position,
                                "RookieYear": rookie_year,
                                "GameSegment": game_segment,
                                "Period": period,
                                "LastNGames": last_n_games,
                                "ClutchTime": clutch_time,
                                "AheadBehind": ahead_behind,
                                "PointDiff": point_diff,
                                "RangeType": range_type,
                                "StartPeriod": start_period,
                                "EndPeriod": end_period,
                                "StartRange": start_range,
                                "EndRange": end_range,
                                "ContextFilter": context_filter, # unsure of what this does
                                "ContextMeasure": context_measure
                            }

        self.response = requests.get(self.base_url, params=self.url_paramaters)

    def change_params(self, parameters):
        """Pass in a disctionary of url parameters to change"""
        self.url_paramaters.update(parameters)
        self.response = requests.get(self.base_url, params=self.url_paramaters)

    def get_shots(self):
        """Returns the shot chart data as a pandas DataFrame."""
        shots = self.response.json()['resultSets'][0]['rowSet']
        headers = self.response.json()['resultSets'][0]['headers']
        return pd.DataFrame(shots, columns=headers)

    def get_img(self):
        """Returns the image of the player from stats.nba.com"""
        url = "http://stats.nba.com/media/players/230x185/" + \
            str(self.player_id) + ".png"
        img_file = str(self.player_id) + ".png"
        return urllib.request.urlretrieve(url, img_file)[0]

    def get_league_avg(self):
        """Returns the league average shooting stats for all FGA in each zone"""
        shots = self.response.json()['resultSets'][1]['rowSet']
        headers = self.response.json()['resultSets'][1]['headers']
        return pd.DataFrame(shots, columns=headers)


def draw_court(ax=None, color='gray', lw=1, outer_lines=False):
    """
    Returns an axes with a basketball court drawn onto to it.

    This function draws a court based on the x and y-axis values that the NBA
    stats API provides for the shot chart data.  For example, the NBA stat API
    represents the center of the hoop at the (0,0) coordinate.  Twenty-two feet
    from the left of the center of the hoop in is represented by the (-220,0)
    coordinates.  So one foot equals +/-10 units on the x and y-axis.

    TODO: explain the parameters
    """
    if ax is None:
        ax = plt.gca()

    # Create the various parts of an NBA basketball court

    # Create the basketball hoop
    hoop = Circle((0, 0), radius=7.5, linewidth=lw, color=color, fill=False)

    # Create backboard
    backboard = Rectangle((-30, -7.5), 60, -1, linewidth=lw, color=color)

    # The paint
    # Create the outer box 0f the paint, width=16ft, height=19ft
    outer_box = Rectangle((-80, -47.5), 160, 190, linewidth=lw, color=color,
                          fill=False)
    # Create the inner box of the paint, widt=12ft, height=19ft
    inner_box = Rectangle((-60, -47.5), 120, 190, linewidth=lw, color=color,
                          fill=False)

    # Create free throw top arc
    top_free_throw = Arc((0, 142.5), 120, 120, theta1=0, theta2=180,
                         linewidth=lw, color=color, fill=False)
    # Create free throw bottom arc
    bottom_free_throw = Arc((0, 142.5), 120, 120, theta1=180, theta2=0,
                            linewidth=lw, color=color, linestyle='dashed')
    # Restricted Zone, it is an arc with 4ft radius from center of the hoop
    restricted = Arc((0, 0), 80, 80, theta1=0, theta2=180, linewidth=lw,
                     color=color)

    # Three point line
    # Create the right side 3pt lines, it's 14ft long before it arcs
    corner_three_a = Rectangle((-220, -47.5), 0, 140, linewidth=lw,
                               color=color)
    # Create the right side 3pt lines, it's 14ft long before it arcs
    corner_three_b = Rectangle((220, -47.5), 0, 140, linewidth=lw, color=color)
    # 3pt arc - center of arc will be the hoop, arc is 23'9" away from hoop
    three_arc = Arc((0, 0), 475, 475, theta1=22, theta2=158, linewidth=lw,
                    color=color)

    # Center Court
    center_outer_arc = Arc((0, 422.5), 120, 120, theta1=180, theta2=0,
                           linewidth=lw, color=color)
    center_inner_arc = Arc((0, 422.5), 40, 40, theta1=180, theta2=0,
                           linewidth=lw, color=color)

    # List of the court elements to be plotted onto the axes
    court_elements = [hoop, backboard, outer_box, inner_box, top_free_throw,
                      bottom_free_throw, restricted, corner_three_a,
                      corner_three_b, three_arc, center_outer_arc,
                      center_inner_arc]

    if outer_lines:
        # Draw the half court line, baseline and side out bound lines
        outer_lines = Rectangle((-250, -47.5), 500, 470, linewidth=lw,
                                color=color, fill=False)
        court_elements.append(outer_lines)

    # Add the court elements onto the axes
    for element in court_elements:
        ax.add_patch(element)

    return ax


def shot_chart(x, y, title="", kind="scatter", color="b", cmap=None,
               xlim=(-250, 250), ylim=(422.5, -47.5),
               court_color="gray", outer_lines=False, court_lw=1,
               flip_court=False, kde_shade=True, hex_gridsize=None,
               ax=None, **kwargs):
    """
    Returns an Axes object with player shots plotted.

    TODO: explain the parameters
    """

    if ax is None:
        ax = plt.gca()

    if cmap is None:
        cmap = sns.light_palette(color, as_cmap=True)

    if not flip_court:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    else:
        ax.set_xlim(xlim[::-1])
        ax.set_ylim(ylim[::-1])

    ax.tick_params(labelbottom="off", labelleft="off")
    ax.set_title(title, fontsize=18)

    draw_court(ax, color=court_color, lw=court_lw, outer_lines=outer_lines)

    if kind == "scatter":
        ax.scatter(x, y, c=color, **kwargs)

    elif kind == "kde":
        sns.kdeplot(x, y, shade=kde_shade, cmap=cmap,
                    ax=ax, **kwargs)
        ax.set_xlabel('')
        ax.set_ylabel('')

    elif kind == "hex":
        if hex_gridsize is None:
            # Get the number of bins for hexbin using Freedman-Diaconis rule
            # This is idea was taken from seaborn, which got the calculation
            # from http://stats.stackexchange.com/questions/798/
            from seaborn.distributions import _freedman_diaconis_bins
            x_bin = _freedman_diaconis_bins(x)
            y_bin = _freedman_diaconis_bins(y)
            hex_gridsize = int(np.mean([x_bin, y_bin]))

        ax.hexbin(x, y, gridsize=hex_gridsize, cmap=cmap, **kwargs)

    else:
        raise ValueError("kind must be 'scatter', 'kde', or 'hex'.")

    return ax


def shot_chart_jointgrid(x, y, data=None, title="", joint_type="scatter",
                     marginals_type="both", cmap=None, joint_color="b",
                     marginals_color="b", xlim=(-250, 250),
                     ylim=(422.5, -47.5), joint_kde_shade=True,
                     marginals_kde_shade=True, hex_gridsize=None, space=0,
                     size=(12, 11), court_color="gray", outer_lines=False,
                     court_lw=1, flip_court=False, joint_kws=None,
                     marginal_kws=None, **kwargs):
    """
    Returns a JointGrid object containing the shot chart.

    TODO: explain the parameters
    """

    # The joint_kws and marginal_kws idea was taken from seaborn
    # Create the default empty kwargs for joint and marginal plots
    if joint_kws is None:
        joint_kws = {}
    joint_kws.update(kwargs)
    if marginal_kws is None:
        marginal_kws = {}

    # If a colormap is not provided, then it is based off of the joint_color
    if cmap is None:
        cmap = sns.light_palette(joint_color, as_cmap=True)

    # Flip the court so that the hoop is by the bottom of the plot
    if flip_court:
        xlim = xlim[::-1]
        ylim = ylim[::-1]

    # Create the JointGrid to draw the shot chart plots onto
    grid = sns.JointGrid(x=x, y=y, data=data, xlim=xlim, ylim=ylim,
                         space=space)

    # Joint Plot
    # Create the main plot of the joint shot chart
    if joint_type == "scatter":
        grid = grid.plot_joint(plt.scatter, color=joint_color, **joint_kws)

    elif joint_type == "kde":
        grid = grid.plot_joint(sns.kdeplot, cmap=cmap,
                               shade=joint_kde_shade, **joint_kws)

    elif joint_type == "hex":
        if hex_gridsize is None:
            # Get the number of bins for hexbin using Freedman-Diaconis rule
            # This is idea was taken from seaborn, which got the calculation
            # from http://stats.stackexchange.com/questions/798/
            from seaborn.distributions import _freedman_diaconis_bins
            x_bin = _freedman_diaconis_bins(x)
            y_bin = _freedman_diaconis_bins(y)
            hex_gridsize = int(np.mean([x_bin, y_bin]))

        grid = grid.plot_joint(plt.hexbin, gridsize=hex_gridsize, cmap=cmap,
                               **joint_kws)

    else:
        raise ValueError("joint_type must be 'scatter', 'kde', or 'hex'.")

    # Marginal plots
    # Create the plots on the axis of the main plot of the joint shot chart.
    if marginals_type == "both":
        grid = grid.plot_marginals(sns.distplot, color=marginals_color,
                                   **marginal_kws)

    elif marginals_type == "hist":
        grid = grid.plot_marginals(sns.distplot, color=marginals_color,
                                   kde=False, **marginal_kws)

    elif marginals_type == "kde":
        grid = grid.plot_marginals(sns.kdeplot, color=marginals_color,
                                   shade=marginals_kde_shade, **marginal_kws)

    else:
        raise ValueError("marginals_type must be 'both', 'hist', or 'kde'.")

    # Set the size of the joint shot chart
    grid.fig.set_size_inches(size)

    # Extract the the first axes, which is the main plot of the
    # joint shot chart, and draw the court onto it
    ax = grid.fig.get_axes()[0]
    draw_court(ax, color=court_color, lw=court_lw, outer_lines=outer_lines)

    # Get rid of the axis labels
    grid.set_axis_labels(xlabel="", ylabel="")
    # Get rid of all tick labels
    ax.tick_params(labelbottom="off", labelleft="off")
    # Set the title above the top marginal plot
    ax.set_title(title, y=1.2, fontsize=18)

    return grid

def shot_chart_jointplot(x, y, data=None, title="", kind="scatter", color="b", 
                         cmap=None, xlim=(-250, 250), ylim=(422.5, -47.5),
                         space=0, court_color="gray", outer_lines=False,
                         court_lw=1, flip_court=False, 
                         set_size_inches=(12, 11), **kwargs):
    """
    Returns a seaborn JointGrid using sns.jointplot

    TODO: Better documentation
    """

    # If a colormap is not provided, then it is based off of the color
    if cmap is None:
        cmap = sns.light_palette(color, as_cmap=True)

    plot = sns.jointplot(x, y, data=None, stat_func=None, kind=kind, space=0, 
                         color=color, cmap=cmap, **kwargs)

    plot.fig.set_size_inches(set_size_inches)


    # A joint plot has 3 Axes, the first one called ax_joint 
    # is the one we want to draw our court onto and adjust some other settings
    ax = plot.ax_joint

    if not flip_court:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    else:
        ax.set_xlim(xlim[::-1])
        ax.set_ylim(ylim[::-1])

    draw_court(ax, color=court_color, lw=court_lw, outer_lines=outer_lines)

    # Get rid of axis labels and tick marks
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.tick_params(labelbottom='off', labelleft='off')

    # Add a title
    ax.set_title(title, y=1.2, fontsize=18)

    return plot



def heatmap_fgp(x, y, z, bins=20, title="", cmap=plt.cm.YlOrRd,
                xlim=(-250, 250), ylim=(422.5, -47.5),
                facecolor='lightgray', facecolor_alpha=0.4,
                court_color="black", outer_lines=False, court_lw=0.5,
                flip_court=False, ax=None, **kwargs):

    """
    Returns an AxesImage object that contains a heatmap of the FG%

    TODO: Explain parameters
    """

    # Bin the FGA (x, y) and Calculcate the mean number of times shot was
    # made (z) within each bin
    # mean is the calculated FG percentage for each bin
    mean, xedges, yedges, binnumber = binned_statistic_2d(x=x, y=y,
                                                          values=z,
                                                          statistic='mean',
                                                          bins=bins)

    if ax is None:
        ax = plt.gca()

    if not flip_court:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    else:
        ax.set_xlim(xlim[::-1])
        ax.set_ylim(ylim[::-1])

    ax.tick_params(labelbottom="off", labelleft="off")
    ax.set_title(title, fontsize=18)

    ax.patch.set_facecolor(facecolor)
    ax.patch.set_alpha(facecolor_alpha)

    draw_court(ax, color=court_color, lw=court_lw, outer_lines=outer_lines)

    heatmap = ax.imshow(mean.T, origin='lower', extent=[xedges[0], xedges[-1],
                        yedges[0], yedges[-1]], interpolation='nearest',
                        cmap=plt.cm.YlOrRd)

    return heatmap


# Bokeh Shot Chart
def bokeh_draw_court(figure, line_width=1, line_color='gray'):
    """Returns a figure with the basketball court lines drawn onto it"""

    # hoop
    figure.circle(x=0, y=0, radius=7.5, fill_alpha=0,
             line_color=line_color, line_width=line_width)

    # backboard
    figure.line(x=range(-30,31), y=-7.5, line_color=line_color)

    # The paint
    # outerbox
    figure.rect(x=0, y=47.5, width=160, height=190,fill_alpha=0, 
              line_color=line_color, line_width=line_width)
    # innerbox
    # left inner box line
    figure.line(x=-60, y=np.arange(-47.5, 143.5), line_color=line_color,
              line_width=line_width)
    # right inner box line
    figure.line(x=60, y=np.arange(-47.5, 143.5), line_color=line_color,
              line_width=line_width)

    # plot.rect(x=0, y=47.5, width=120, height=190, fill_alpha=0,
    #           line_color=line_color, line_width=line_width)

    # Restricted Zone
    figure.arc(x=0, y=0, radius=40, start_angle=pi, end_angle=0,
             line_color=line_color, line_width=line_width)

    # top free throw arc
    figure.arc(x=0, y=142.5, radius=60, start_angle=pi, end_angle=0,
             line_color=line_color)

    # bottome free throw arc
    figure.arc(x=0, y=142.5, radius=60, start_angle=0, end_angle=pi,
             line_color=line_color, line_dash="dashed")

    # Three point line
    # corner three point lines
    figure.line(x=-220, y=np.arange(-47.5, 92.5), line_color=line_color,
              line_width=line_width)
    figure.line(x=220, y=np.arange(-47.5, 92.5), line_color=line_color,
              line_width=line_width)
    # # three point arc
    figure.arc(x=0, y=0, radius=237.5, start_angle=3.528, end_angle=-0.3863,
             line_color=line_color, line_width=line_width)

    # add center court
    # outer center arc
    figure.arc(x=0, y=422.5, radius=60, start_angle=0, end_angle=pi,
             line_color=line_color, line_width=line_width)
    # inner center arct
    figure.arc(x=0, y=422.5, radius=20, start_angle=0, end_angle=pi,
             line_color=line_color, line_width=line_width)


    # outer lines, consistting of half court lines and out of bounds
    # lines
    figure.rect(x=0, y=187.5, width=500, height=470, fill_alpha=0, 
              line_color=line_color, line_width=line_width)
    
    return figure


def bokeh_shot_chart(source, x="LOC_X", y="LOC_Y", fill_color="#1f77b4",
                     fill_alpha=0.3, line_alpha=0.3, court_lw=1,
                     court_line_color='gray'):
    """
    Returns a figure with both FGA and basketball court lines drawn onto it.

    This function expects data to be a ColumnDataSource with the x and y values
    named "LOC_X" and "LOC_Y".  Otherwise specify x and y.  
    """

    # source = ColumnDataSource(data)

    fig = figure(width=700, height=658, x_range=[-250,250],
                  y_range=[422.5, -47.5], min_border=0,
                  x_axis_type=None, y_axis_type=None,
                  outline_line_color="black")

    fig.scatter(x, y, source=source, size=10, fill_alpha=0.3,
                 line_alpha=0.3)

    bokeh_draw_court(fig, line_color='gray')

    return fig
