import os, struct
import numpy as np
import numpy.linalg as la
from flags import Flags
#import scipy.spatial.distance as ssd

import matplotlib
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw
import networkx as nx

from collections import defaultdict

WAD_DIR = "../data/wad/"

class SpecialType(Flags):
    blocks_players_and_monsters = ()
    blocks_monsters = ()


# offsets for reading WAD
WAD_TYPE_END = 4
NUM_LUMPS_END = 8
DIR_OFF_END = 12
LUMP_ENT_SIZE = 16
LUMP_NAME_IDX = -8

# struct.unpack returns a tuple even if the data is only
# one element. This constant indexes the data in that tuple.
DATA_IDX = 0

# WAD spec constants
LINEDEF_SIZE = 14
VERTEX_SIZE = 4

def get_wad_paths():
    ''' Return a list of relative paths to all .wad or .WAD files
    found in the subdirectories of WAD_DIR
    '''
    return [WAD_DIR + "/" + d + "/" + f \
            for d in os.listdir(WAD_DIR) \
                for f in os.listdir(WAD_DIR+"/"+d) \
                    if "wad" in f or "WAD" in f]
def get_wad_data(wad):
    wad_data = None
    with open(wad, "rb") as f:
        wad_data = f.read()
    return wad_data

def get_wad_index(wad_data):
    wad_type = wad_data[:WAD_TYPE_END].decode("ASCII")
    num_lumps = struct.unpack("<L",
                              wad_data[WAD_TYPE_END:NUM_LUMPS_END])[DATA_IDX]
    dir_offset = struct.unpack("<L",
                               wad_data[NUM_LUMPS_END:DIR_OFF_END])[DATA_IDX]
    lump_index = {}

    lump_start = dir_offset
    lump_end = lump_start + LUMP_ENT_SIZE
    for i in range(int(num_lumps)):
        lump = wad_data[lump_start:lump_end]
        lump_offset = struct.unpack("<l", lump[:4])[DATA_IDX]
        lump_size = struct.unpack("<l", lump[4:LUMP_NAME_IDX])[DATA_IDX]
        # The name of the lump is zero-padded from the right
        lump_name = lump[LUMP_NAME_IDX:].decode('UTF8').rstrip("\x00")
        lump_index[lump_name] = (lump_offset, lump_size)
        lump_start = lump_end
        lump_end+=LUMP_ENT_SIZE

    return lump_index

out_edges = defaultdict(list)


def get_linedefs(linedefs_data):
    ''' Returns the linedefs of the wad which are represented by a tuple,
    (source_vertex, sink_vertex, flags, one_sided). The source and sink vertices
    are each doubles of coordinates in the plane and flags is a byte array
    which stores various attributes of the linedef. one_sided is a boolean
    which is true if the linedef has only one side.

    Also returns a dictionary which maps each node to its incident edges.

    @param: linedefs_data: byte array of linedefs data
    @return: linedefs, outedges: a list of tuples with linedef data, dict
    of incident edges for each node
    '''
    num_sides = int(len(linedefs_data)/LINEDEF_SIZE)
    print("Number of lines: ", num_sides)
    linedefs = []

    for i in range(num_sides):
        linedef_start = LINEDEF_SIZE*i
        vertex_start_idx = struct.unpack("<h",
                        linedefs_data[linedef_start:linedef_start+2])[DATA_IDX]
        vertex_end_idx = struct.unpack("<h",
                        linedefs_data[linedef_start+2:linedef_start+4])[DATA_IDX]
        flags = struct.unpack("<h",
                        linedefs_data[linedef_start+4:linedef_start+6])[DATA_IDX]
        back_sidedef = struct.unpack("<h",
                        linedefs_data[linedef_start+12:linedef_start+14])[DATA_IDX]
        one_sided = False
        if back_sidedef < 0:
            one_sided = True
        linedefs.append((vertex_start_idx, vertex_end_idx, flags, one_sided))

        out_edges[vertex_start_idx].append(vertex_end_idx)
        #if not one_sided:
        #    out_edges[vertex_end_idx].append(vertex_start_idx)

    return linedefs, out_edges


def get_vertex(vertex_string):
        x_coord = struct.unpack("<h",
                                vertex_string[0:2])[DATA_IDX]
        y_coord = struct.unpack("<h",
                                vertex_string[2:4])[DATA_IDX]
        return x_coord, y_coord


def get_vertexes(vertexes_string):
    num_vertexes = int(len(vertexes_string) / VERTEX_SIZE)
    vertexes = []
    max_coord = 0
    for i in range(num_vertexes):
        # Each sequence of 4 bytes is a vertex tuple
        vertex_start = 4*i
        x_coord, y_coord = \
            get_vertex(vertexes_string[vertex_start:vertex_start+VERTEX_SIZE])
        if abs(x_coord) > max_coord:
            max_coord = abs(x_coord)
        if abs(y_coord) > max_coord:
            max_coord = abs(y_coord)
        vertexes.append((x_coord, y_coord))
    return vertexes, max_coord


def chunk_data(offsets, wad_data):
    ''' Extracts the data for a particular lump from full block of wad data.

    @param: offsets: tuple of start and end offsets of the lump
    @param: wad_data: byte string of wad data
    @return: byte string of lump data
    '''
    data_start = offsets[0]
    data_end = data_start+offsets[1]
    return wad_data[data_start:data_end]


def draw_with_plt(vertexes, linedefs):
    x_idxs, y_idxs = map(list, zip(*vertexes))
    plt.scatter(x_idxs, y_idxs,s=.04)

    for linedef in linedefs:
        linedef_flags = linedef[2]
        if (1 & linedef_flags) | (1 & linedef_flags >> 7):
            vertex_start = vertexes[linedef[0]]
            vertex_end = vertexes[linedef[1]]
            xx, yy = map(list, zip(*[vertex_start, vertex_end]))
            plt.plot(xx, yy)
    plt.show()


def decode_wad(wad):
    wad_data = get_wad_data(wad)
    wad_index = get_wad_index(wad_data)

    vertexes, max_coord = get_vertexes(chunk_data(wad_index['VERTEXES'], wad_data))
    linedefs, out_edges = get_linedefs(chunk_data(wad_index['LINEDEFS'], wad_data))

    #incident_edges = find_incident_edges(out_edges, linedefs)

    BLACK = (0, 0, 0)
    WHITE = "#ffffff"
    RED = "#ff0000"
    OFFSET = max_coord+100
    MyImage = Image.new('RGB', (2*OFFSET, 2*OFFSET), BLACK)
    offset_vertexes = []

    for vertex in vertexes:
        # WAD files and Pillow do not share the same coordinate system
        offset_vertexes.append((OFFSET+vertex[0], OFFSET-vertex[1]))
    G = nx.Graph()
    G.add_nodes_from(offset_vertexes)
    MyDraw = ImageDraw.Draw(MyImage)

    edges = []

    # first draw all the lines
    for linedef in linedefs:
        linedef_flags = linedef[2]
        one_sided = linedef[3]
        if (1 & linedef_flags) | (1 & linedef_flags >> 7) | True:
            vertex_start = offset_vertexes[linedef[0]]
            vertex_end = offset_vertexes[linedef[1]]
            edges.append((vertex_start, vertex_end))
            #if not one_sided:
            #    edges.append((vertex_end, vertex_start))
            MyDraw.line([vertex_start, vertex_end], fill=WHITE)

    G.add_edges_from(edges)
    paths = set()
    #print(len(list(nx.simple_cycles(G))))
    for vertex in list(G.nodes):
        try:
            print("vertex: ", vertex)
            path=nx.find_cycle(G,source=vertex,orientation='ignore')
            path_list = []
            for l in list(path):
                path_list.extend([l[0], l[1]])
            incident_vertices = [offset_vertexes[i] for i in
                                 out_edges[offset_vertexes.index(vertex)]]
            print(vertex,incident_vertices,str(path_list) )
            MyDraw.polygon(path_list, fill=WHITE, outline=RED)
        except Exception:
            print("Exception for node: ", vertex)

    MyImage.show()
    #draw_with_plt(vertexes, linedefs)
    return wad_index

if __name__=="__main__":
    print(decode_wad(get_wad_paths()[0]))
