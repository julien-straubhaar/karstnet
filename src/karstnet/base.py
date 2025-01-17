#    Copyright (C) 2018-2024 by
#    Philippe Renard <philippe.renard@unine.ch>
#    Pauline Collon <pauline.collon@univ-lorraine.fr>
#    All rights reserved.
#    MIT license.
#
"""
Karstnet Base
=============

Karstnet is a Python package for the analysis of karstic networks.

The base module contains all the classes and tools to manipulate
graphs and compute statistics.

"""

# ----- External libraries importations
import pandas as pd
import numpy as np
import networkx as nx
import scipy.stats as st
import matplotlib.pyplot as plt
# import sqlite3
# noinspection PyUnresolvedReferences
# import mplstereonet


# *****************************************************************************
# ----- Test function -----
# *****************************************************************************
def test_kn():
    print("test ok")
    print("relance ok")


# *****************************************************************************
# ----- KGraph class -----
# *****************************************************************************

class KGraph:
    """
    Class dedicated to the construction and manipulation of graphs
    representing Karstic network.

    Attributes
    ----------
    - graph : networkx.Graph
        the complete graph of the karstic network;
        each station is a node, each line-of-sight is an edge;
        `graph` is a networkx.Graph object, with :

        - node attributes :
            - 'pos' : sequence of 2 or 3 floats
                position in 2D or 3D

            - + other optional properties (sequence) given as
                one attribute (see `properties_name` and
                `properties_name_list` below); this allows to
                store additional information

        - edge attributes :
            - 'length' : float
                real length of the edge

            - 'length2d': float
                length of the projction of the edge in the xy plane

            - 'azimuth' : float
                azimuth in degree in [0, 360), from North towards East
                (counter-clockwise); it provide "bearings" for stereonet;
                for vertical edge, the azimuth is set to Nan

            - 'dip' : float
                dip in degree in [0, 90], "plunging" direction, towards
                under the xy plane; it provide "plunges" for stereonet

    - graph_simpl : networkx.Graph
        the simplified graph of the karstic network,
        i.e. all nodes of degree 2 are removed (from the complete graph)
        except in loops (2 types of loops, cf Collon et al., 2017,
        Geomorphology), so that the topology is preserved;
        `graph_simple` is a networkx.Graph object, with :

        - node attributes :
            - 'pos' : sequence of 2 or 3 floats
                position in 2D or 3D (of retained nodes)

        - edge attributes :
            - 'length' : float
                real length of the corresponding branch in the
                complete graph

    - properties_name : str, optional
        name of the "sequence of properties" given as one additional
        attribute for nodes in the complete graph

    - properties_name_list : list (or sequence) of strs, optional
        names of each entry in the "sequence of properties" given as
        one additional attribute for nodes in the complete graph

    - branches : list
        list of branches, each entry is a list of nodes forming a
        path in the complete graph corresponding to a branch

    - br_lengths : list of floats
        list of branch lengths, each entry is the real length of
        the path forming the corresponding branch in `branches`

    - br_tort : list of floats
        list of branch tortuosities, i.e. real length of the branches
        divided by the distance between the two extremities

    - list_simpl_edges : list of tuple
        list of simple edges, i.e. the edges in the simplified graph,
        necessary to export graph to plines
    """

    def __init__(self, edges, coordinates,
                 remove_isolated_node=True,
                 properties=None,
                 properties_name=None,
                 properties_name_list=None,
                 # edge_properties=None,
                 # edge_properties_name=None,
                 # edge_properties_name_list=None,
                 verbose=True):
        """
        Creates a Kgraph from nodes and edges.

        Parameters
        ----------
        edges : list
            a list of edges

        coordinates : dict
            coordinates of the nodes, dictionary where keys are node names, and
            values are sequences of length 2 or 3 (dimension)

        remove_isolated_node: bool, default: True
            indicates if isolated nodes, i.e. nodes of degree 0, are discarded
            (not included in the graph)

        properties : dict, optional
            properties associated to the nodes, keys are node names,
            values are sequences (tuple, list, or array), i.e. a
            "sequence of properties" is associated to the nodes

        properties_name : str, optional
            name of the "sequence of properties" given as values in dictionary
            `properties`; by default, 'prop' is used

        properties_name_list : list (or sequence) of strs, optional
            names of each entry in the "sequence of properties" given as values
            in dictionary `properties`; by default: ['prop0', 'prop1', ...] is used

        verbose : bool, default: True
            indicates if some information is printed (`True`) during the creation
            of the object

        Examples
        --------
            >>> myKGraph = KGraph([],{}) # create an empty karstic network
        """

        self.verbose = verbose

        # Initialization of the graph (networkx)
        self.graph = nx.Graph()

        # Set edges and nodes
        self.graph.add_edges_from(edges)
        if not remove_isolated_node:
            self.graph.add_nodes_from(coordinates) # add nodes not in any edge...

        # Set node positions
        nx.set_node_attributes(self.graph, coordinates, 'pos')
        # ----- Notes -----
        # Position in 2d and in 3d, can be retrieved by
        # using the methodes pos2d and pos3d respectively.
        # # self.pos2d, self.pos3d = _pos_initialization(coordinates)
        # -----

        if self.verbose:
            print(
                "\nThis network contains",
                nx.number_connected_components(self.graph),
                "connected components")

        # Set node properties (if any)
        if properties is not None:
            if properties_name is None:
                properties_name = 'prop'
            if properties_name_list is None:
                properties_name_list = [f'prop{i}' for i in range(len(list(properties.values())[0]))]
            nx.set_node_attributes(self.graph, properties, properties_name)
        else:
            properties_name = None
            properties_name_list = None
        self.properties_name = properties_name
        self.properties_name_list = properties_name_list

        # ----- Could be added: Properties on edges -----
        # # Set edge properties (if any)
        # if edge_properties is not None:
        #     if edge_properties_name is None:
        #         edge_properties_name = 'edge_prop'
        #     if properties_name_list is None:
        #         edge_properties_name_list = [f'edge_prop{i}' for i in range(len(list(edge_properties.values())[0]))]
        #     nx.set_node_attributes(self.graph, properties, properties_name)
        # else:
        #     edge_properties_name = None
        #     edge_properties_name_list = None
        # self.edge_properties_name = edge_properties_name
        # self.edge_properties_name_list = edge_properties_name_list
        # -----

        # Compute and set edge attributes
        self._set_graph_lengths()
        self._set_graph_orientations()

        # Compute the branches of the graph
        # self.branches is necessary to export graph to plines
        self.branches, self.br_lengths, self.br_tort = self._getallbranches()

        # Construct the simplified graph
        # self.list_simpl_edges is necessary to export graph to plines
        self.list_simpl_edges, self.graph_simpl = self._simplify_graph()

    # *************************************************************************
    # Methods for getting node position
    # *************************************************************************
    def pos2d(self):
        """
        Gets node position in 2D (ignoring z-coordinate if it exists).

        Returns
        -------
        pos2d: dict
            keys are node names, values are the 2D position of the nodes
        """
        pos = nx.get_node_attributes(self.graph, 'pos')
        # try:
        #     if len(list(pos.values())[0]) == 2: # pos is 2D
        #         pos2d = {k:tuple(v) for k, v in pos.items()}
        #         # pos2d = pos
        #     else: # pos is 3D
        #         pos2d = {k:tuple(v[:2]) for k, v in pos.items()}
        # except:
        #     # e.g. when pos={}
        #     pos2d = {}

        pos2d = {k:tuple(v[:2]) for k, v in pos.items()}
        return pos2d

    def pos3d(self):
        """
        Gets node position in 3D (setting z-coordinate to zero if it does not exist).

        Returns
        -------
        pos2d: dict
            keys are node names, values are the 3D position of the nodes
        """
        pos = nx.get_node_attributes(self.graph, 'pos')
        try:
            if len(list(pos.values())[0]) == 2: # pos is 2D
                pos3d = {k:tuple(v[:2])+(0.0,) for k, v in pos.items()}
            else: # pos is 3D
                pos3d = {k:tuple(v) for k, v in pos.items()}
                # pos3d = pos
        except:
            # e.g. when pos={}
            pos3d = {}

        return pos3d

    # *************************************************************************
    # Methods for plots
    # *************************************************************************
    def plot2(self,
              graph_type=0,
              with_labels=True,
              node_size=300,
              node_color='lightblue',
              show_ticks=False,
              axis_equal=False,
              figsize=(6, 3)):
        """
        Plots a 2D view of the karstic network.

        Parameters
        ----------
        graph_type : int
            if 0 displays the complete graph,
            if 1 displays the simplified graph

        with_labels : bool, default: True
            indicates if labels of nodes are displayed (True) or not (False)

        node_size : float (or int), default: 300
            size of the nodes

        node_color : color, default: 'lightblue'
            color of the nodes

        show_ticks : bool, default: False
            if True, ticks are displays along x and y axes

        axis_equal : bool, default: False
            if True, same scale is used for x and y axes

        figsize : tuple, default: (6, 3)
            contains the (width, height) dimension of the figure

        Examples
        --------
            >>> myKGraph = KGraph([],{})
            >>> myKGraph.plot2()
            >>> myKGraph.plot2(1)
        """
        # 2D  plot

        if graph_type == 0:
            self._plot2(self.graph, with_labels, node_size, node_color, show_ticks, axis_equal, figsize)
            plt.title('original')
            plt.show()
        else:
            self._plot2(self.graph_simpl, with_labels, node_size, node_color, show_ticks, axis_equal, figsize)
            plt.title('simplified')
            plt.show()

        return

    def plot3(self, graph_type=0, zrotation=30, xyrotation=0, show_nodes=False, figsize=(6, 3)):
        """
        Plots a 3D view of the karstic network.

        Parameters
        ----------
        graph_type : int
            if 0 displays the complete graph,
            if 1 displays the simplified graph

        zrotation : float
            angle in degrees between horizontal plane and viewpoint

        xyrotation : float
            angle in degree for the horizontal rotation of the viewpoint;
            if xyrotation=0, the view is from the South toward North

        show_nodes : bool, default: False
            if True, the nodes and the edges are plotted;
            if False, only the edges are plotted

        figsize : tuple, default: (6, 3)
            contains the (width, height) dimension of the figure

        Examples
        --------
            >>> myKGraph.plot3()
            >>> myKGraph.plot3(1, zrotation=20, xyrotation=-30)
        """
        # 3D  plot

        if graph_type == 0:
            self._plot3(self.graph, zrotation, xyrotation, show_nodes, figsize)
            plt.title('original')
            plt.show()
        else:
            self._plot3(self.graph_simpl, zrotation, xyrotation, show_nodes, figsize)
            plt.title('simplified')
            plt.show()

        return

    def plot(self, xlabel='x', ylabel='y', figsize=(12, 5)):
        """
        Simple 2D map of the original and simplified karstic network.

        The two maps are ploted side by side. This function allows
        to check rapidly the data after an import for example.

        Parameters
        ----------
        xlabel : str, default: 'x'
            label for x-axis

        ylabel : str, default: 'y'
            label for x-axis

        figsize : tuple, default: (12, 5)
            contains the (width, height) dimension of the figure

        Returns
        -------
        fig: Figure object (matplotlib)

        Examples
        --------
            >>> myKGraph.plot()
        """
        fig = plt.figure(figsize=figsize)
        plt.subplot(121)
        nx.draw_networkx(self.graph,
                         pos=self.pos2d(),
                         with_labels=False,
                         node_size=0.1)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.subplot(122)
        nx.draw_networkx(self.graph_simpl,
                         pos=self.pos2d(),
                         with_labels=False,
                         node_size=0.1)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()

        return fig

    def plotxz(self, figsize=(12, 5)):
        """
        Calls method `plot` with keyword arguments xlabel='x', ylabel='z'.
        """
        fig = self.plot(xlabel='x', ylabel='z', figsize=figsize)

        return fig

    # ------
    # Member function written by Philippe Vernant 2019/11/25
    # Modified by Pauline Collon (aug. 2020) to weight density map by lenghts

    def stereo(self, weighted=True):
        """
        Density map of orientations and rose diagram of the karstic network.

        The two maps are ploted side by side.

        The density of orientations of the edges of the complete graph is
        ploted according to Schmidt\'s projection on lower hemisphere (stereo)
        on the left map ("3D orientations"), and as a rose diagram accounting only
        for the azimuth, i.e. orientation of the projection of the edges in the
        xy plane ("2D orientations") on the right map.

        By default, stereo and rose diagram are weighted by the length of the
        edges: for stereo, the length in 3D (real length), and for rose diagram,
        the length in 2D (length of the projection in the xy plane).

        Parameters
        ----------
        weighted : bool, default: True
            indicates if the maps are weighted by length (True), or if
            each edge has the same weight (False)

        Returns
        -------
        fig: Figure object (matplotlib)

        Examples
        --------
            >>> myKGraph.stereo()
            >>> myKGraph.stereo(weighted = False)
        """

        # Create an np.array of azimuths and dips
        # and lengths (projected(2d) and real (3d))
        azim = np.asarray(list((nx.get_edge_attributes(self.graph, 'azimuth')).values()))
        azim_not_Nan = azim[~np.isnan(azim)] # for rose diagram (exclude vertical edges)
        bearing_dc = np.nan_to_num(azim) # convert nan to zero (and inf to very large number)
        plunge_dc = np.array(list((nx.get_edge_attributes(self.graph, 'dip')).values()))
        if weighted:
            l2d = np.array(list((nx.get_edge_attributes(self.graph, 'length2d')).values()))
            l2d_not_Nan = l2d[~np.isnan(azim)] # for rose diagram
            l3d = np.array(list((nx.get_edge_attributes(self.graph, 'length')).values()))
        else:
            l2d_not_Nan = None
            l3d = None

        # Making colormap, based on Collon et al.(2017)
        # we saturate the colormap at 40%
        from matplotlib import colormaps
        from matplotlib.colors import ListedColormap
        from matplotlib.gridspec import GridSpec

        nbint = 15
        levels = np.linspace(0, 1, nbint)
        rainbow = colormaps['rainbow']
        newcolors = rainbow(levels)
        white = np.ones(4) # np.array([256 / 256, 256 / 256, 256 / 256, 1])
        newcolors[:1, :] = white
        newcmp = ListedColormap(newcolors)

        # Define the grid for plotting maps and the figure
        gs = GridSpec(nrows=20, ncols =2)
        fig = plt.figure(figsize=(16, 8))

        # ----- Stereo -----
        # Density map - Allows to consider almost vertical conduits
        # The data are weigthted by the real length of the segments (l3d)
        # Use the traditional "Schmidt" method : 1% count
        dc = fig.add_subplot(gs[:-2, 0], projection='stereonet')
        cdc = dc.density_contourf(plunge_dc,
                                  bearing_dc,
                                  measurement='lines',
                                  method='schmidt',
                                  levels=np.arange(0, nbint * 2 + 1, 2),
                                  extend='both',
                                  cmap=newcmp,
                                  weights=l3d)
        dc.set_title('Density map of orientations [Schmidt\'s projection]',
                     y=1.10,
                     fontsize=15)
        dc.grid()
        dc._polar.set_position(dc.get_position())
        dc.set_azimuth_ticks(np.arange(0, 351, 10))

        dc_cb = fig.add_subplot(gs[-1:, 0]) # axe for the colorbar, made invisible
        for spine in ["top", "bottom", "left", "right"]:
            dc_cb.spines[spine].set_visible(False)
            dc_cb.set_xticks([])
            dc_cb.set_yticks([])

        # Colorbar of the density map
        cbar = plt.colorbar(cdc, ax=dc_cb,
                            fraction=0.95,
                            pad=0.01,
                            orientation='horizontal')
        cbar.set_label('[%]')

        # ----- Rose diagram -----
        # The azimuth data are weighted by the projected length (l2d)
        bin_edges = np.arange(-5, 366, 10)
        number_of_strikes, bin_edges = np.histogram(azim_not_Nan,
                                                    bin_edges,
                                                    weights=l2d_not_Nan)
        number_of_strikes[0] += number_of_strikes[-1]
        half = np.sum(np.split(number_of_strikes[:-1], 2), 0)
        two_halves = np.concatenate([half, half])

        # Rose diagram on right hand side of picture
        rs = fig.add_subplot(gs[:-2,1], projection='polar')
        rs.bar(np.deg2rad(np.arange(0, 360, 10)),
               two_halves,
               width=np.deg2rad(10),
               bottom=0.0,
               color='.8',
               edgecolor='k')
        rs.set_theta_zero_location('N')
        rs.set_theta_direction(-1)
        rs.set_thetagrids(np.arange(0, 360, 10), labels=np.arange(0, 360, 10))
        rs.set_title('Rose Diagram of the cave survey segments',
                     y=1.10,
                     fontsize=15)

        fig.tight_layout()
        plt.show()

        return fig

    # end modif PV 2019/11/25

    # *************************************************************
    # Methods for export
    # *************************************************************

    # Export to csv files (Julien Straubhaar)
    def to_csv(
            self,
            basename,
            suffix_nodes='_nodes.csv',
            suffix_links='_links.csv',
            delimiter_nodes=';',
            delimiter_links=';',
            export_prop_nodes=True,
            # export_prop_links=True,
            verbose=True):
        """
        Exports the Kgraph to two csv files (nodes, and links).

        The file for nodes has the columns:
            - 'id', 'x', 'y'[, 'z', '<node_prop0>', '<node_prop1>', ...]

        The file for links has the columns:
            - '<node_idA>', '<node_idB>'

        where the two columns, '<node_idA>', '<node_idB>', are the node ids
        of the two extremities of a link (edge).

        Parameters
        ----------
        basename : str
            the base name (prefix) used for the input files

        suffix_nodes : str, default: '_nodes.csv'
            the input file containing the nodes is
            `basename``suffix_nodes`

        suffix_links : str, default: '_links.csv'
            the input file containing the links (edges) is
            `basename``suffix_links`

        delimiter_nodes : str, default:';'
            delimiter used in file for nodes

        delimiter_links : str, default:';'
            delimiter used in file for links

        export_prop_nodes : bool, default: True
            indicates if the properties associated to the nodes (attribute
            of "sequences of properties" named `.properties_name`, single
            properties named as `.properties_name_list`) are exported

        verbose : bool, default: True
            indicates if info is printed

        Examples
        --------
            >>> MyKGraph.to_csv("MyKarst")
        """

        # Files
        filename_nodes = f'{basename}{suffix_nodes}'
        filename_links = f'{basename}{suffix_links}'

        # ----- Set data frame for nodes -----
        try:
            nodes_list = list(self.graph.nodes())
            nodes_df = pd.DataFrame(nodes_list, columns=['id'])

            # Position
            tmp = nx.get_node_attributes(self.graph, 'pos')
            tmp = np.asarray(list(tmp.values()))
            if tmp.shape[1] == 2:
                nodes_df[['x', 'y']] = tmp
            else: # tmp.shape[1] should be equal to 3
                nodes_df[['x', 'y', 'z']] = tmp

            # Properties
            if export_prop_nodes and self.properties_name is not None:
                tmp = nx.get_node_attributes(self.graph, self.properties_name)
                tmp = np.asarray(list(prop.values()))
                nodes_df[self.properties_name_list] = tmp
        except:
            print(f'EXPORT ERROR: Could not set data frame for nodes')
            return

        # ----- Set data frame for links -----
        links = np.asarray(list(self.graph.edges())) # should have 2 columns
        try:
            links_df = pd.DataFrame(links, columns=['idA', 'idB'])
        except:
            print(f'EXPORT ERROR: Could not set data frame for links')
            return

        # Export
        try:
            nodes_df.to_csv(filename_nodes, index=False, sep=delimiter_nodes)
        except:
            print(f'EXPORT ERROR: Could not export csv file for nodes ({filename_nodes})')
            return

        try:
            links_df.to_csv(filename_links, index=False, sep=delimiter_links)
        except:
            print(f'EXPORT ERROR: Could not export csv file for links ({filename_links})')
            return

        if verbose:
            print(f"Graph successfully exported to csv files ({filename_nodes}, {filename_links}) !\n")
        return

    def to_pline(self, basename):
        """
        Exports the complete graph to Pline (GOCAD ASCII object).

        Manages the colocated vertices indicated by the mention "ATOM" in
        the ASCII file.

        `Warning`: this version does not export the properties on the nodes.

        Parameters
        ----------
        basename : str
            the base name of the file name used for the Pline file,
            the name contains no extension, it will be added by the function

        Examples
        --------
        The following command saves the file "MyKarst_exported.pl" :

            >>> myKGraph.to_pline("MyKarst")
        """
        # For a complete graph the list of Ilines corresponds to self.branches

        self._ilines_to_pline(self.branches, basename)

        return

    def simpleGraph_to_pline(self, basename):
        """
        Exports the simplified graph to pline (GOCAD ASCII object).

        Manages the colocated vertices indicated by the mention "ATOM"
        in the ASCII file.

        `Warning`: this version does not export the properties on the nodes.

        Parameters
        ----------
        basename : str
            the base name of the file name used for the Pline file,
            the name contains no extension, it will be added by the function

        Examples
        --------
        The following command saves the file "MyKarst_simpl_exported.pl" :

            >>> myKGraph.to_pline("MyKarst")
        """
        # For a simplified graph the list of Ilines will corresponds to
        # the edges of the simple graph
        # one iline is created for each edge, which is not exactly a branch
        # but does not prevent exportation

        # to clearly explicit it is the simplified graph
        basename = basename + "_simpl_"
        self._ilines_to_pline(self.list_simpl_edges, basename)

        return

    # *************************************************************
    # -------------------Computation for Analysis------------------
    # *************************************************************
    def basic_analysis(self):
        """
        Prints the basic statistics of a karstic network graph analysis.

        Examples
        --------
            >>> t = myKGraph.basic_analysis()
        """

        # On the complete graph
        nb_nodes_comp = nx.number_of_nodes(self.graph)
        nb_edges_comp = nx.number_of_edges(self.graph)

        # On the simplified graph
        nb_nodes = nx.number_of_nodes(self.graph_simpl)
        nb_edges = nx.number_of_edges(self.graph_simpl)
        nb_connected_components = nx.number_connected_components(
            self.graph_simpl)

        nb_cycles = nb_edges - nb_nodes + nb_connected_components

        # Compute all extremities and junction nodes (on the simple graph)
        nb_extremity_nodes = 0
        nb_junction_nodes = 0
        for i in self.graph_simpl.nodes():
            if self.graph_simpl.degree(i) == 1:
                nb_extremity_nodes += 1
            elif self.graph_simpl.degree(i) > 2:
                nb_junction_nodes += 1

        # Print these basics
        print(
            "\nThis network contains :\n",
            nb_nodes_comp,
            "nodes (stations) and ",
            nb_edges_comp,
            "edges.\n",
            "On the simplified graph, there are : ",
            nb_nodes,
            "nodes (stations) and ",
            nb_edges,
            "edges,\n",
            nb_extremity_nodes,
            "are extremity nodes (entries or exits) and ",
            nb_junction_nodes,
            "are junction nodes.\nThere is/are ",
            nb_connected_components,
            "connected component.s and ",
            nb_cycles,
            "cycle.s.\n")

        # Howard's parameters
        # (Howard, A. D., Keetch, M. E., & Vincent, C. L. (1970).
        # Topological and geometrical properties of braided patterns.
        # Water Resources Research, 6(6), 1674–1688.)
        # Rmq: All nodes of the simplified network have to be considered,
        # even those of degree 2 in the cycles or, it is not becoming
        # consistent with the exemples of braided rivers given by Howard.
        # This is indeed consistent with what has been done in
        # Collon, P., Bernasconi, D., Vuilleumier, C., & Renard, P. (2017).
        # Statistical metrics for the characterization of karst network
        # geometry and topology. Geomorphology, 283, 122–142.
        alpha = nb_cycles / (2 * (nb_nodes) - 5)
        beta = nb_edges / (nb_nodes)
        gamma = nb_edges / (3 * (nb_nodes - 2))
        print("\nHoward's parameter are (Howard, 1970) :",
              "\n alpha :", alpha,
              "\n beta :", beta,
              "\n gamma :", gamma)
        print("\nNote that this computation considers the node of degree 2",
              "necessary to loop preservations as Seed Nodes, in order to",
              "stay consistent with Howard's illustrations.")

    def mean_tortuosity(self):
        """
        Computes the mean tortuosity of a karstic network.

        Returns
        -------
        t : float
            mean tortuosity of the branches

        Examples
        --------
            >>> t = myKGraph.mean_tortuosity()
        """
        nb_of_Nan = np.isnan(self.br_tort).sum()

        if self.verbose:
            if nb_of_Nan != 0:
                print(
                    "\nWARNING: This network contains",
                    nb_of_Nan,
                    "looping branche.s, which is.are not considered for the ",
                    "mean tortuosity computation.")

        return np.nanmean(self.br_tort)

    def mean_length(self):
        """
        Computes the mean length of the branches of a karstic networkx

        Returns
        -------
        l : float
            mean length of the branches

        Examples
        --------
            >>> l = myKGraph.mean_length()
        """
        return np.mean(self.br_lengths)

    def coef_variation_length(self):
        """
        Computes the coefficient of variation of length of the branches of a
        karstic network.

        Returns
        -------
        cvl : float
            coefficient of variation of the length of the branches

        Examples
        --------
            >>> cvl = myKGraph.coef_variation_length()
        """

        # std is used with ddof=1 to use the estimate of std (divides by N-1)
        # The test below is to avoid an error with np.std computation
        # for networks having a single branch.
        if len(self.br_lengths) > 1:
            return np.std(self.br_lengths, ddof=1) / np.mean(self.br_lengths)
        else:
            return 0 # No

    def length_entropy(self, mode="default"):
        """
        Computes the entropy of lengths of the branches of a karstic network.

        The entropy of the branch lenghths, normalized in [0, 100] is computed
        as

        .. math::
            H=-\\sum_i p_i \\log_n(p_i)

        where :math:`p_i` is the probability mass in the `i`-th bin, and the
        base of logarithm, `n`, is the number of bins (see parameter `mode`
        below).

        Parameters
        ----------
        mode : str
            the mode style should be equal to "default" or "sturges";
            the property is normalized as compared to the
            maximum observed value

            - If mode="default", the entropy is computed according to the \
            implementation proposed in Collon et al. 2017 : \
            using a fixed bin number of 10 and ranging from 0 to 100
            - If mode="sturges" : alternative calculation using Sturges \
            rules to define the number of bins

        Returns
        -------
        entropy : float
            entropy of the lengths of the branches

        Examples
        --------
            >>> l_entrop = myKGraph.length_entropy()
        """

        v = self.br_lengths
        # In the paper of 2017, we normalize the length to get comparable
        # results between networks
        v = v[np.nonzero(v)]
        v_normalized = v / np.amax(v) * 100

        if (len(v) > 1):
            if mode == "sturges":
                # Sturges rule to define the number of bins fron nb of samples
                nbins = int(np.ceil(1 + np.log2(len(v_normalized))))
            else:
                # Fixed nb of bins to facilitate comparison,
                # as done in Collon et al 2017 and other papers on roads
                # orientation entropy
                nbins = 10

            # Pauline : range should be 0 and 100 or we get very different
            # results from what we should have
            # (I verify that 0 and 100 are included in the counts)
            counts, _ = np.histogram(v_normalized,
                                     bins=nbins,
                                     range=(0, 100))
            freq = counts / np.sum(counts)  # Computes the frequencies
            entropy = st.entropy(freq, base=nbins)
        else:
            entropy = 0  # v contains a single value - no uncertainty

        return entropy

    def orientation_entropy(self, mode="default", weighted=True):
        """
        Computes the entropy of orientation (2d) of the segments of a
        karstic network.

        The 2D orientation is considered, i.e. the orientation of the
        projection of the edges (links) in xy plane; the entropy of the
        distribution of the azimuth angles in degrees in [0, 180) is
        computed as

        .. math::
            H=-\\sum_i p_i \\log_n(p_i)

        where :math:`p_i` is the probability mass in the `i`-th bin, and the
        base of logarithm, `n`, is the number of bins (see parameter `mode`
        below).

        By default, the distribution of azimuth is weighted by the length of the
        projection of the edges in the xy plane.

        Parameters
        ----------
        mode : string
            the mode style should be equal to "default" or "sturges"

            - If mode="default", the entropy is computed according to \
            the implementation proposed in Collon et al. 2017 : \
            using a fixed bin number of 18 and ranging from 0 to 180
            - If mode="sturges" : alternative calculation using Sturges \
            rules to define the number of bins

        weighted : bool, default: True
            indicates if the distribution are weighted by length (True), or if
            each edge has the same weight (False)

        Returns
        -------
        entropy : float
            entropy of the segment orientation

        Examples
        --------
            >>> or_entropy = myKGraph.orientation_entropy()
        """

        # Get azimuths
        azim = np.array(list((nx.get_edge_attributes(self.graph, 'azimuth')).values()))
        # Removing NAN Azimuth values (vertical edges, length 2d is zero)
        ind = ~np.isnan(azim)
        azim = azim[ind]
        azim = azim % 180 # identify angle azimuth angles that differ from 180 (Julien Straubhaar)

        if weighted:
            # Get length 2d for weights
            weights = np.array(list((nx.get_edge_attributes(self.graph, 'length2d')).values()))
            weights = weights[ind]
        else:
            weights = None

        if len(azim) > 1:
            if mode == "sturges":
                # Sturges rule to define the number of bins fron nb of samples
                nbins = int(np.ceil(1 + np.log2(len(azim))))
            else:
                # Fixed nb of bins to facilitate comparison,
                # as done in Collon et al 2017 and other papers on roads
                # orientation entropy
                nbins = 18

            # Pauline : range should be 0 and 180 or we get very different
            # results from what we had (change the position of bin borders).
            # Also, I verified that it is consistent :
            # 0 and 180 are counted, not excluded
            counts, _ = np.histogram(azim,
                                     bins=nbins,
                                     range=(0, 180),
                                     weights=weights)
            freq = counts / np.sum(counts)  # Computes the frequencies
            return st.entropy(freq, base=nbins)
        else:
            return 0

    # added by Julien Straubhaar, 17 January 2025
    def orientation3d_entropy(self, nbins_azim=36, nbins_dip=6, weighted=True):
        r"""
        Computes the entropy of orientation of the segments of a
        karstic network.

        The 3D orientation is considered, i.e. orientation of an edge
        (link) is defined by the dip angle in degrees in [0, 90], and
        the azimuth angle in degrees in [0, 360).

        A pair of angles :math:`(\alpha, \theta)`, where :math:`\alpha`
        is the azimuth and :math:`\theta` the dip, defines a point on
        the lower hemisphere

        .. math::
            (\sin\alpha \cos\theta, \cos\alpha \cos\theta, -\sin\theta)

        The entropy of the joint distribution of the azimuth, dip angles
        is computed as

        .. math::
            H=-\\sum_i p_i \\log_n(p_i)

        where :math:`p_i` is the probability mass in the `i`-th bin, and the
        base of logarithm, `n`, is the number of bins.

        The bins cover the hemisphere, and every bin has the same area.
        Denoting :math:`n_\alpha` and :math:`n_\theta` the number of bins
        for azimuth and dip respectively (given as parameters), the total
        number of bins for the joint distribution is

        .. math::
            n = 1 + n_\alpha \cdot (n_\theta -1)

        where, with
        :math:`0=\alpha_0 < \alpha_1 <\cdots < \alpha_{n_\alpha}=360` and
        :math:`0=\theta_0 < \theta_1 <\cdots < \theta_{n_\theta}=90` the limits
        angles for azimuth and dip respectively, the 2D bins are

        .. math::
            \{(\alpha, \theta)\ :\ \theta_{n_{\theta-1}} \leqslant \theta\} \text{ "sphere cap"}\\
            \{(\alpha, \theta)\ :\ \alpha_{i} \leqslant \alpha < \alpha_{i+1}, \ \theta_{j} \leqslant \theta < \theta_{j+1}\}

        for :math:`i=0, \ldots, n_{\alpha-1}`, :math:`j=0, \ldots, n_{\theta-1}`.

        By default, the distribution of angles is weighted by the length of the
        of the edges.

        Parameters
        ----------
        nbins_azim : int, default: 36
            number of bins for azimuth angles (see above), at least 1

        nbins_dip : int, default: 6
            number of bins for dip angles (see above), at least 2

        weighted : bool, default: True
            indicates if the distribution are weighted by length (True), or if
            each edge has the same weight (False)

        Returns
        -------
        entropy : float
            entropy of the segment orientation

        Examples
        --------
            >>> or3d_entropy = myKGraph.orientation3d_entropy()
        """

        # Determine limits angles so that area of each bin is equal.
        #
        # With na = nbins_azim: alpha_i = i/(2pi), i=0, ..., na.
        #
        # With nt = nbins_dip: to determine theta_j, j=0, ..., nt.
        # Let
        #   theta_0=0, theta_{nt}=90
        # The area of a crown (on the hemisphere) defined
        # by an interval theta in [t1, t2], is equal to
        # sin(t2) - sin(t1).
        #
        # Then, with b = 1/nbins, and nbins = 1+na*(nt-1) the total
        # number of 2d bins, let
        #   theta_{nt-1} = arcsin(1-b)
        # such that the area of the sphere cap, for theta in
        # [theta_{nt_-1}, 90], is equal to b=1/nbins.
        #
        # Let
        #   theta_j = arcsin(j*na*b), j=0, ..., nt-1, (*)
        # such that the crown of the sphere, for theta in
        # [theta_{j}, theta_{j+1}], is equal to na*b.
        # Hence, as the crown will be divided in na parts (according
        # to bins for azimuth), each part will have an area of b.
        #
        # Note that (*) for j=0 and j=nt-1, give the same value
        # as the values f or theta_0 and theta_{nt-1} given before.
        #

        if nbins_azim < 1:
            print('ERROR: invalid bins for azimuth')
            return None
        if nbins_dip < 2:
            print('ERROR: invalid bins for dip')
            return None

        # Define bins limits
        nbins = 1 + nbins_azim*(nbins_dip-1)
        alphas = np.arange(nbins_azim+1)*2*np.pi/nbins_azim
        thetas = np.arcsin(np.arange(nbins_dip)*nbins_azim/nbins) # theta_{nbins_dip} = 90 not stored
        # thetas = np.hstack((np.arcsin(np.arange(nbins_dip)*nbins_azim/nbins), np.array([np.pi/2])))

        alphas = np.rad2deg(alphas)
        thetas = np.rad2deg(thetas)

        # Get azimuths and dips
        azim = np.array(list((nx.get_edge_attributes(self.graph, 'azimuth')).values()))
        dip = np.array(list((nx.get_edge_attributes(self.graph, 'dip')).values()))
        azim = np.nan_to_num(azim) # convert nan to zero (and inf to very large number

        if weighted:
            # Get length
            length = np.array(list((nx.get_edge_attributes(self.graph, 'length')).values()))

        # Compute probability mass in each bins (probability weighted by length of edges)
        counts = []
        for j in range(nbins_dip-1):
            ind = np.all((dip >= thetas[j], dip < thetas[j+1]), axis=0)
            if weighted:
                weights = length[ind]
            else:
                weights = None
            c, _ = np.histogram(azim[ind], bins=nbins_azim, range=(0, 360), weights=weights)
            counts.append(c)

        ind = dip >= thetas[nbins_dip-1]
        if weighted:
            c = np.array([np.sum(length[ind])])
        else:
            c = np.array([np.sum(ind)])
        counts.append(c)

        counts = np.hstack(counts)

        freq = counts / np.sum(counts)  # Computes the frequencies
        return st.entropy(freq, base=nbins)

    def mean_degree_and_CV(self):
        """
        Computes the average and the coefficient of variation of the degree.

        The computation is done on the simplified graph.

        Returns
        -------
        tuple
            meandeg, cvdeg : the mean and coefficient of variation
            of the node degrees

        Examples
        --------
            >>> meandeg, cvde = myKGraph.coef_variation_degree()
        """
        # Vector of degrees
        d = np.asarray(self.graph_simpl.degree())[:, 1]

        # Mean degree
        meandeg = np.mean(d)

        # Coefficient of variation of the degrees : std is used with ddof=1 to
        # use the estimate of std (divides by N-1)
        cvde = np.std(d, ddof=1) / np.mean(d)

        return meandeg, cvde

    def correlation_vertex_degree(self, cvde=None):
        """
        Computes the correlation of vertex degree.

        The computation is done on the simplified graph.

        Parameters
        ----------
        cvde : float (or bool: False), optional
            Optional input: coefficient of variation of the degree.
            If not provided (`None` or `False`), it is computed automatically internally.

        Returns
        -------
        cvd : float
            Correlation of Vertex Degree

        Examples
        --------
           >>> cvd = myKGraph.correlation_vertex_degree()
        """
        if cvde is None or not cvde:
            _, cvde = self.mean_degree_and_CV()

        # To avoid division by 0 when computing correlation coef
        if cvde != 0:
            cvd = nx.degree_pearson_correlation_coefficient(self.graph_simpl)
        else:
            cvd = 1

        return cvd

    def central_point_dominance(self):
        """
        Computes central point dominance.

        The computation is done on the simplified graph.

        Returns
        -------
        cpd : float
            central point dominance

        Examples
        --------
            >>> cpd = myKGraph.central_point_dominance()
        """
        bet_cen = nx.betweenness_centrality(self.graph_simpl)
        bet_cen = list(bet_cen.values())
        cpd = np.sum(max(bet_cen) - np.array(bet_cen)) / (len(bet_cen) - 1)

        return cpd

    def average_SPL(self, dist_weight=False):
        """
        Computes the average shortest path length.

        Notes
        -----

        The computation is done on the simplified graph.

        The function handles the case of several connected components
        which is not the case for the Networkx function
        "average_shortest_path_length".

        In case of several connected components, the average_SPL
        is the average of each SPL weighted by the number of nodes of each
        connected component.

        Parameters
        ----------
        dist_weight : bool, default: False
            if True, shortest path lengths are computed by weighting the
            edges by their length

        Returns
        -------
        aspl : float
            average shortest path length

        Examples
        --------
            >>> aspl = myKGraph.average_SPL()
        """

        sum_aspl = 0  # initialize the sum
        # Compute average spl on each connected component with Networkx
        for c in (self.graph_simpl.subgraph(c).copy()
                  for c in nx.connected_components(self.graph_simpl)):
            if not dist_weight:
                sum_aspl += nx.average_shortest_path_length(
                    c) * nx.number_of_nodes(c)
            else:
                sum_aspl += nx.average_shortest_path_length(
                    c, weight="length") * nx.number_of_nodes(c)

        av_SPL = sum_aspl / nx.number_of_nodes(self.graph_simpl)

        return av_SPL

    def characterize_graph(self, verbose=False):
        """
        Computes the set of metrics used to characterize a graph.

        Parameters
        ----------
        verbose : bool, default: False
            If True, the function displays information about the
            progress of the computation, and the results.

        Returns
        -------
        results : dict
            All the statistical metrics are stored in a dictionary. The
            keys of the dictionary are provided below with
            corresponding explanation.

            - `mean length` : mean length of the branches
            - `cv length` : coefficient of variation of length of branches
            - `length entropy` : entropy of the length of the branches
            - `mean tortuosity` : mean tortuosity of the branches
            - `orientation entropy` : entropy of the orientation (2d, azimuth) of the conduits
            - `orientation 3d entropy` : entropy of the orientation (3d, azimuth and dip) of the conduits
            - `aspl` : average shortest path length
            - `cpd` : central point dominance
            - `mean degree` : mean of the vertex degrees
            - `cv degrees` : coefficient of variation of vertex degrees
            - `correlation vertex degree` :  correlation of vertex degrees

        Examples
        --------
            >>> results = myKGraph.characterize_graph()
        """

        results = {}

        if verbose:
            print('Computing:')
            print(' - mean length', end='', flush=True)

        results["mean length"] = self.mean_length()

        if verbose:
            print(', cv length', end='', flush=True)

        results["cv length"] = self.coef_variation_length()

        if verbose:
            print(', length entropy', end='', flush=True)

        results["length entropy"] = self.length_entropy()

        if verbose:
            print(', mean tortuosity', end='', flush=True)

        results["tortuosity"] = self.mean_tortuosity()

        if verbose:
            print('', end='\n', flush=True)
            print(' - orientation entropy', end='', flush=True)

        results["orientation entropy"] = self.orientation_entropy()

        if verbose:
            print(', orientation 3d entropy', end='', flush=True)

        results["orientation 3d entropy"] = self.orientation3d_entropy()

        if verbose:
            print('', end='\n', flush=True)
            print(' - aspl', end='', flush=True)

        results["aspl"] = self.average_SPL()

        if verbose:
            print(', cpd', end='', flush=True)

        results["cpd"] = self.central_point_dominance()

        if verbose:
            print(', md, cv degree', end='', flush=True)

        md, cvde = self.mean_degree_and_CV()
        results["mean degree"] = md
        results["cv degree"] = cvde

        if verbose:
            print(', cvd', end='', flush=True)

        cvd = self.correlation_vertex_degree(cvde=cvde)
        results["correlation vertex degree"] = cvd

        if verbose:
            print('', end='\n', flush=True)
            print("--------------------------------------")
            for key in results.keys():
                print(f' {key:25s} = {results[key]:6.3f}')
                # print(" %25s = %5.3f" % (key, results[key]))
            print("--------------------------------------")

        return results

    # *************************************************************************
    # Non Public member functions of KGraph class
    # *************************************************************************

    # *******************************
    # Private functions for plots
    # *******************************
    def _plot2(self,
               G,
               with_labels=True,
               node_size=300,
               node_color='lightblue',
               show_ticks=False,
               axis_equal=False,
               figsize=(6, 3)):
        """
        NOT PUBLIC
        Plots a 2D view of a graph G that could be the simplified of the
        complete one.
        Requires self.pos2d() member function.
        Called by the plot2() public function.
        """

        # 2D  plot
        fig = plt.figure(figsize=figsize)

        nx.draw_networkx(G,
                         with_labels=with_labels,
                         pos=self.pos2d(),
                         node_size=node_size,
                         node_color=node_color)
        if show_ticks:
            plt.gca().tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        if axis_equal:
            plt.axis('equal')

        return fig

    def _plot3(self, G, zrotation=30, xyrotation=0, show_nodes=False, figsize=(6, 3)):
        """
        NOT PUBLIC
        Plots a 3D view of a graph G that could be the simplified or the
        complete one.
        Requires self.pos3d() member function.
        Called by the plot3() public function.
        """

        # 3D  plot
        # try:
        #     from mpl_toolkits.mplot3d import Axes3D
        # except ImportError:
        #     raise ImportError("karstnet.plot3 requires mpl_toolkits.mplot3d ")

        # fig = plt.figure(figsize=figsize)
        # ax = Axes3D(fig)

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(projection='3d')

        pos3d = self.pos3d()
        for e in G.edges():
            x = np.array((pos3d[e[0]][0], pos3d[e[1]][0]))
            y = np.array((pos3d[e[0]][1], pos3d[e[1]][1]))
            z = np.array((pos3d[e[0]][2], pos3d[e[1]][2]))

            # Plot the connecting lines
            ax.plot(x, y, z, c='black', alpha=0.5)

        if show_nodes:
            ax.scatter(*np.asarray(list(pos3d.values())).T, c='blue', alpha=0.5)

        # Set the view
        ax.view_init(zrotation, -xyrotation - 90)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        return fig

    # *******************************
    # Private function for export
    # *******************************
    def _ilines_to_pline(self, list_Iline, basename):
        """
        Writes a Pline (Gocad ascii object) from a list of ilines in a file.

        Used to export either complete or simplified graph.

        Manages the colocated vertices indicated by the mention "ATOM"
        in the ASCII file.

        WARNING: this version does not export (for the moment)
        the properties on the nodes.

        Parameters
        ----------
        list_Iline : list
            list of the ilines to write

        basename: str
            string containing the base name of output file;
            the output file name is `basename`_exported.pl

        Examples
        --------
            >>> myKGraph.to_Pline("MyKarst" )
        """

        # Pline file creation
        output_file_name = basename + '_exported.pl'
        f_pline = open(output_file_name, 'w')

        # Header writing
        f_pline.write('GOCAD PLine 1\n')
        f_pline.write('HEADER {\n')
        f_pline.write('name:' + output_file_name + '\n')
        f_pline.write('}\n')
        f_pline.write('GOCAD_ORIGINAL_COORDINATE_SYSTEM\n')
        f_pline.write('NAME Default\nAXIS_NAME "U" "V" "W"\n')
        f_pline.write('AXIS_UNIT "m" "m" "m"\n')
        f_pline.write('ZPOSITIVE Elevation\n')
        f_pline.write('END_ORIGINAL_COORDINATE_SYSTEM\n')
        f_pline.write('PROPERTY_CLASS_HEADER Z {\n')
        f_pline.write('is_z:on\n}\n')

        # Create an empty dictionary of nodes already written in one iline
        # key is the same as in dico_nodes, the node index
        # value is the corresponding number of vtrx in the iline file,
        # due to the specific numbering of this file format,
        # it is different of the node index
        dico_added_nodes = {}

        pos3d = self.pos3d()

        # To count vertices: in plines,
        # vertices are virtually duplicated in the counting
        cpt_vrtx = 1
        # Each branch would be an iline
        for iline in list_Iline:
            f_pline.write('ILINE\n')

            # Memorize counting state to write correctly the segments
            cpt_vrtx_deb = cpt_vrtx

            # Each node of a iline is written as a vertex or atom
            for node in iline:
                # First, verify that this node has not already been
                # added to choose between vrtx or atom
                if node not in dico_added_nodes:
                    f_pline.write('VRTX ' + str(cpt_vrtx) + ' ' +
                                  str(pos3d[node][0]) + ' ' +
                                  str(pos3d[node][1]) + ' ' +
                                  str(pos3d[node][2]) + '\n')
                    # Update dico_added_nodes to indicate that the node
                    # has already been declared
                    # and store the correct index in the pline domain
                    dico_added_nodes[node] = cpt_vrtx
                # if node is in dico_added_nodes, we must build an atom
                # refering to the vrtx number in the pline
                else:
                    f_pline.write('ATOM ' + str(cpt_vrtx) + ' ' +
                                  str(dico_added_nodes[node]) + '\n')
                # Update vrtx counting to treat the next node of the iline
                cpt_vrtx += 1
            # When all nodes of a branch have been written, write the list
            # of segments using new numbers
            for i in range(len(iline) - 1):
                f_pline.write('SEG ' + str(cpt_vrtx_deb + i) + ' ' +
                              str(cpt_vrtx_deb + i + 1) + '\n')
            # One Iline has been written, go to next one

        # All ilines have been written
        f_pline.write('END\n')

        if self.verbose:
            print('File created')

        # Close the file
        f_pline.close()
        return

    # *******************************
    # Private functions used by constructors
    # *******************************
    def _set_graph_lengths(self):
        """NON PUBLIC.
        Computes edge length at the creation of KGraph object.
        This function is called by all constructors.
        It updates graph.
        """

        pos = nx.get_node_attributes(self.graph, 'pos')

        # Creation of a dictionary to store the length of each edge
        length = {e:np.sqrt(np.sum((np.asarray(pos[e[1]])-np.asarray(pos[e[0]]))**2))
                  for e in self.graph.edges()}
        # length = {}
        # for e in self.graph.edges():
        #     dx = self.pos3d[e[0]][0] - self.pos3d[e[1]][0]
        #     dy = self.pos3d[e[0]][1] - self.pos3d[e[1]][1]
        #     dz = self.pos3d[e[0]][2] - self.pos3d[e[1]][2]
        #     length[e] = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        # Storing the length as an edge attribute
        nx.set_edge_attributes(self.graph, length, 'length')

        return

    def _simplify_graph(self):
        """
        Constructs a simplified graph by removing nodes of degree 2, while
        preserving the topology.

        Member function:
          Use self.graph (with its "length" attribute on edges) and self.pos3d()
          Use self.branches produced by self._get_allbranches()

        Returns:
        --------
        list_simpl_edges : list
            list of the simple edges (necessary for export to pline)

        Gs : networkx.Graph
            the simplified output graph
        """

        # Deals with cycles and loops to ensure that topology is not changed
        simpl_edges = _split_branches(self.branches)

        # Creates a new empty graph
        Gs = nx.Graph()

        # list of simpl_edges for export
        list_simpl_edges = []

        # Fills the graph with simpl_edges
        for i in simpl_edges:
            list_simpl_edges.append([i[0], i[-1]])

            # Compute the length of the current edge
            l_edge = np.sum(np.asarray([self.graph.edges[(node0, node1)]['length']
                                        for node0, node1 in zip(i[:-1], i[1:])]))
            # Notes:
            #  - `self.graph.edges[(node0, node1)]` gets the edge (node0, node1) or (node1, node0),
            #    i.e. order of endpoints does not matter
            #  - if the dictionary is first extracted: `length = nx.get_edge_attributes(self.graph, 'length')`,
            #    then order in the key (node0, node1) or (node1, node0) matters, i.e. if the first key is not found,
            #    then the second key must be used !
            #
            # ----- using the extracted dictionary `length` -----
            # l_edge = 0
            # for node0, node1 in zip(i[:-1], i[1:]):
            #     local_edge = (node0, node1)
            #     if length.__contains__(local_edge):
            #         br_len += length[local_edge]
            #     else:
            #         local_edge = (node1, node0)
            #         if length.__contains__(local_edge):
            #             br_len += length[local_edge]
            #         else:
            #             print("Warning: could not find ",
            #                   "1 edge when computing length")
            # -----

            # Add the edge corresponding to the simpl_edges, with attribute 'length'
            Gs.add_edge(i[0], i[-1], length=l_edge)

        # ----- Alternative -----
        # Creates the dictionary for length of edges
        edges_length = {}
        # Fills the graph with simpl_edges

        for i in simpl_edges:
            list_simpl_edges.append((i[0], i[-1]))

            # Compute the length of the current edge
            l_edge = np.sum(np.asarray([self.graph.edges[(node0, node1)]['length']
                                        for node0, node1 in zip(i[:-1], i[1:])]))

            edges_length[list_simpl_edges[-1]] = l_edge

        # Stores the results
        Gs.add_edges_from(list_simpl_edges)
        nx.set_edge_attributes(Gs, edges_length, 'length')
        # -----

        return list_simpl_edges, Gs

    # # ----- Alternative -----
    # def _simplify_graph(self):
    #     """
    #     Constructs a simplified graph by removing nodes of degree 2, while
    #     preserving the topology.

    #     Member function:
    #       Use self.graph (with its "length" attribute on edges) and self.pos3d()
    #       Use self.branches produced by self._get_allbranches()

    #     Returns:
    #     --------
    #     list_simpl_edges : list
    #         list of the simple edges (necessary for export to pline)

    #     Gs : networkx.Graph
    #         the simplified output graph
    #     """

    #     # Deals with cycles and loops to ensure that topology is not changed
    #     simpl_edges = _split_branches(self.branches)

    #     # Creates a new empty graph
    #     Gs = nx.Graph()

    #     # list of simpl_edges for export
    #     list_simpl_edges = []

    #     # Creates the dictionary for length of edges
    #     edges_length = {}

    #     # Fills the graph with simpl_edges

    #     for i in simpl_edges:
    #         list_simpl_edges.append((i[0], i[-1]))

    #         # Compute the length of the current edge
    #         l_edge = np.sum(np.asarray([self.graph.edges[(node0, node1)]['length']
    #                                     for node0, node1 in zip(i[:-1], i[1:])]))

    #         edges_length[list_simpl_edges[-1]] = l_edge

    #     # Stores the results
    #     Gs.add_edges_from(list_simpl_edges)
    #     nx.set_edge_attributes(Gs, edges_length, 'length')

    #     return list_simpl_edges, Gs
    # # ----- Alternative -----

    def _getallbranches(self):
        """
        Constructs the list of all branches of the karstic graph self_graph.
        Computes lengths and tortuosities.
        """

        # Initialisations
        target = []
        degreeTarget = []

        # Create one subgraph per connected components
        list_sub_gr = [self.graph.subgraph(c).copy()
                       for c in nx.connected_components(self.graph)]

        # Get list of nodes that are at the extremities of a branch, and their degree;
        # for loop with all nodes of degree 2, keep one node
        for sub_gr in list_sub_gr:
            d = dict(sub_gr.degree())
            node_id = np.asarray(list(d.keys()))
            node_degree = np.asarray(list(d.values()))
            ind = node_degree != 2

            if ind.sum() == 0:
                # all nodes of degree 2: add last one
                ind[-1] = True

            target.append(node_id[ind])
            degreeTarget.append(node_degree[ind])

        if len(target) == 0:
            # no target node
            if self.verbose:
                print("Warning: This network contains zero branch.\n")

            branches = []
            br_lengths = []
            br_tort = []
            return branches, np.array(br_lengths), np.array(br_tort)

        target = np.hstack(target)
        degreeTarget = np.hstack(degreeTarget)

        # Identifies all the neighbors of those nodes,
        # to create all the initial paths
        listStartBranches = [[i, n] for i in target for n in self.graph.neighbors(i)]

        # Follow all these initial paths to get all the branches
        branches = []
        for path in listStartBranches:
            go = True
            # Check all existing branches to avoid adding a branch twice
            # if starting from other extremity
            for knownbranch in branches:
                if ((path[0] == knownbranch[-1]) &
                        (path[1] == knownbranch[-2])):
                    go = False
                    break
            if go:
                new_branch = self._getbranch(path)
                # Be sure that starting and ending nodes of the new branch
                # do not correspond to ending and starting nodes respectively of
                # an existing branch: if is is the case, the order of nodes in
                # the new branch are reversed
                # [this could be removed, because this should not happen due to
                # the way `listStartBranches` is created]
                for knownbranch in branches:
                    if new_branch[0] == knownbranch[-1] and new_branch[-1] == knownbranch[0]:
                        new_branch = new_branch[::-1]
                        break
                # Append new branch
                branches.append(new_branch)

        # Compute the list of branch lengths and tortuosities
        br_lengths = []
        br_tort = []

        # length = nx.get_edge_attributes(self.graph, 'length') # not used see note in the for loop below
        for br in branches:

            # Computes the distance between extremities
            dist = np.sqrt(np.sum((np.asarray(self.graph.nodes[br[0]]['pos'])
                                   - np.asarray(self.graph.nodes[br[-1]]['pos']))**2))

            # Computes the length of the current branch
            br_len = np.sum(np.asarray([self.graph.edges[(node0, node1)]['length']
                                        for node0, node1 in zip(br[:-1], br[1:])]))
            # Notes:
            #  - `self.graph.edges[(node0, node1)]` gets the edge (node0, node1) or (node1, node0),
            #    i.e. order of endpoints does not matter
            #  - if the dictionary is extracted first: `length = nx.get_edge_attributes(self.graph, 'length')`,
            #    then order in the key (node0, node1) or (node1, node0) matters, i.e. if the first key is not found,
            #    then the second key must be used !
            #
            # ----- using the extracted dictionary `length` -----
            # br_len = 0
            # for node0, node1 in zip(br[:-1], br[1:]):
            #     local_edge = (node0, node1)
            #     if length.__contains__(local_edge):
            #         br_len += length[local_edge]
            #     else:
            #         local_edge = (node1, node0)
            #         if length.__contains__(local_edge):
            #             br_len += length[local_edge]
            #         else:
            #             print("Warning: could not find ",
            #                   "1 edge when computing length")
            # -----

            br_lengths.append(br_len)

            # Set the toruosity of the current branch
            if np.isclose(dist, 0.0):
                tort = np.nan
            else:
                tort = br_len / dist

            br_tort.append(tort)

        br_lengths = np.asarray(br_lengths)
        br_tort = np.asarray(br_tort)

        if self.verbose:
            nb_of_Nan = np.isnan(br_tort).sum()
            if nb_of_Nan:
                print(
                    "Warning: This network contains",
                    nb_of_Nan,
                    "looping branch.es.",
                    "Tortuosity is infinite on a looping branch.",
                    "The looping branches are not considered",
                    "for the mean tortuosity computation.\n")

        return branches, np.array(br_lengths), np.array(br_tort)


    # ***********
    # Functions related to branches of graphs
    # - a branch is defined between two nodes of degree != 2
    # - exception: loop with all nodes of degree 2, one node is
    #   retained for both extremities of the branch
    # ***********
    def _nextstep(self, path):
        """
        Works on self.graph
        Adds the next node to a path of self_graph along a branch.
        Stops when reaches a node of degree different from 2.
        """

        current = path[-1]
        # Checks first if the end of the path is already on an end
        if self.graph.degree(current) != 2:
            stopc = False
            return path, stopc

        # This is a security / it may be removed
        if len(path) > 1:
            old = path[-2]
        else:
            old = current

        # Among the neighbors search for the next one
        for nextn in self.graph.neighbors(current):
            if old != nextn:
                break

        # Add the next node to the path and check stopping criteria
        path.append(nextn)

        # Test for a closed loop / even if start node has degree = 2
        testloop = path[0] == path[-1]

        if (self.graph.degree(nextn) != 2) or testloop:
            stopc = False
        else:
            stopc = True

        return path, stopc

    def _getbranch(self, path):
        """
        Works on self.graph
        Constructs a branch from a starting node.
        """

        path, stopc = self._nextstep(path)
        while stopc:
            path, stopc = self._nextstep(path)
        return path

    # *******************************
    # Private functions used for orientations
    # *******************************
    def _set_graph_orientations(self):
        """NON PUBLIC.
        Computes edge length at the creation of KGraph object.
        This function is called by all constructors.
        It updates graph.
        """

        pos3d = self.pos3d()

        # Creation of a dictionary to store the projected length of each edge,
        # the dip and the azimuth
        length2d = {}
        dip = {}
        azimuth = {}
        for e in self.graph.edges():
            dx = pos3d[e[0]][0] - pos3d[e[1]][0]
            dy = pos3d[e[0]][1] - pos3d[e[1]][1]
            dz = pos3d[e[0]][2] - pos3d[e[1]][2]
            length2d[e] = np.sqrt(dx ** 2 + dy ** 2)

            if length2d[e] != 0:
                dip[e] = np.arctan(dz / length2d[e])  # returns in radians
                dip[e] = np.degrees(dip[e])
                azimuth[e] = np.pi / 2 - np.arctan2(dy, dx) # returns in radians
                azimuth[e] = np.degrees(azimuth[e]) % 360

                if (dz < 0):
                    # negative cave gradients yield positive plunges in a stereonet
                    dip[e] = -dip[e]
                else:
                    # positive cave gradients have bearings 180 apart for the stereonet
                    azimuth[e] = (azimuth[e] + 180) % 360

            else:  # case of nearly pure vertical segments
                azimuth[e] = np.nan
                # azimuth[e] = 0.0 #Convention
                dip[e] = 90  # degrees

        # Storing the length as an edge attribute
        nx.set_edge_attributes(self.graph, length2d, 'length2d')
        nx.set_edge_attributes(self.graph, azimuth, 'azimuth')
        nx.set_edge_attributes(self.graph, dip, 'dip')

        # azimuth: in [0, 360), from North, towards East (counter-clockwise)        -> bearings for stereonet
        # dip: in [0, 90], "plunging" direction, towards under the horizontal plane -> plunges for stereonet
        return

# -------------------END of KGraph class-------------------------------

# *********************************************************************
# NON public functions used by KGraph
# (can not be placed in another file)
# *********************************************************************


# def _pos_initialization(coordinates):
#     '''NON PUBLIC.
#     Creates a dictionary of 3d coordinates from 2d or 3d input coordinates.
#     If only x, y are provided, z is set to 0
#     '''

#     coord_are_3d = True
#     for key in coordinates.keys():
#         if len(coordinates[key]) == 3:
#             break  # we only check one value and let coord_are_3d to True
#         else:
#             coord_are_3d = False
#             break
#     pos3d = {}
#     pos2d = {}
#     #  if coordinates are 3d
#     if coord_are_3d:
#         pos3d = coordinates
#         for key, coord in coordinates.items():
#             pos2d[key] = [coord[0], coord[1]]

#     # if only x and y are provided, set a z value = 0
#     else:
#         pos2d = coordinates
#         for key, coord in coordinates.items():
#             pos3d[key] = [coord[0], coord[1], 0]

#     return pos2d, pos3d


# ******Functions used for graph simplification


def _split2(list_):
    """
    Splits a list in 2 sublists.
    """
    list_length = len(list_)
    if (list_length == 2):
        return (list_, [])
    else:
        midpoint = int(list_length / 2)
        return (list_[0:midpoint + 1], list_[midpoint:list_length])


def _split3(list_):
    """
    Splits a list in 3 sublists.
    """
    list_length = len(list_)
    if (list_length == 2):
        return (list_, [], [])
    elif (list_length == 3):
        l1, l2 = _split2(list_)
        return (l1, l2, [])
    else:
        k = int(list_length / 3.)
        return (list_[0:k + 1], list_[k:2 * k + 1], list_[2 * k:list_length])


def _split_branches(branches):
    """
    Split branches in cases of loop or cycles.
    """
    # Note: if two branches have the same endpoints, then
    # their starting nodes must be the same ones (and then also the
    # ending nodes); hence, any cycle with two junction nodes (of degree
    # greater than 2), is composed of two branches with the same
    # starting node (one of the junction node) and the same
    # ending node (the other junction node); this is guaranteed by
    # construction in method `_get_allbranches` of class:`KGraph`

    # Creates a dictionary to accelerate the search
    # of branches having the same extremities
    list_branches = dict()
    for i, b in enumerate(branches):
        key = (b[0], b[-1])
        if list_branches.__contains__(key):
            list_branches[key].append(i)
        else:
            list_branches[key] = [i]

    # Loop over the branches with same extremities and split them when required
    simpl_edges = []

    for key in list_branches:
        nbb = len(list_branches[key])
        # We test first if this is a loop (same start and end point)
        isloop = key[0] == key[1]

        # Simple case - no cycle but potential loop
        if nbb == 1:
            tmp = branches[list_branches[key][0]]
            if isloop:  # In the loop case, we need to split by 3 the branch
                tmp1, tmp2, tmp3 = _split3(tmp)
                simpl_edges.append(tmp1)
                if (len(tmp2) > 0):
                    simpl_edges.append(tmp2)
                    if (len(tmp3) > 0):
                        simpl_edges.append(tmp3)
            else:
                simpl_edges.append(tmp)

        # Several branches with same extremities - cycles and/or multiple loops
        else:
            if isloop:  # Case with multiple loops, we need to split by 3 each branch
                for i in range(nbb):
                    tmp = branches[list_branches[key][i]]
                    tmp1, tmp2, tmp3 = _split3(tmp)
                    simpl_edges.append(tmp1)
                    if (len(tmp2) > 0):
                        simpl_edges.append(tmp2)
                        if (len(tmp3) > 0):
                            simpl_edges.append(tmp3)
            else:  # Regular branches belonging to cycles, we need to split in 2 each branch
                for i in range(nbb):
                    tmp = branches[list_branches[key][i]]
                    tmp1, tmp2 = _split2(tmp)
                    simpl_edges.append(tmp1)
                    if (len(tmp2) > 0):
                        simpl_edges.append(tmp2)
                # # -----
                # # Note that the last branch could not be splitted, while preserving the topology
                # for i in range(nbb-1):
                #     tmp = branches[list_branches[key][i]]
                #     tmp1, tmp2 = _split2(tmp)
                #     simpl_edges.append(tmp1)
                #     if (len(tmp2) > 0):
                #         simpl_edges.append(tmp2)
                # # last branch
                # tmp = branches[list_branches[key][-1]]
                # simpl_edges.append(tmp)
                # # -----

    return simpl_edges
