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

from node import get_angle

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
    linedefs = []
    adj_list = defaultdict(list)

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

        adj_list[vertex_start_idx].append(vertex_end_idx)
        adj_list[vertex_end_idx].append(vertex_start_idx)

    return linedefs, adj_list


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

def get_index(vertex, vertexes):
    for i in range(len(vertexes)):
        if np.array_equal(vertexes[i], vertex):
            return i
    print(vertex, vertexes)
    return None

def get_next_edge(cur_edge, ordered_incident_vertices):
    tail_vertex = cur_edge[0]
    head_vertex = cur_edge[1]
    head_incident_vertices = ordered_incident_vertices[head_vertex]
    next_hop = get_next_hop(tail_vertex, head_incident_vertices)
    return (head_vertex, next_hop)


def get_next_hop(tail_vertex, head_incident_edges):
    tail_vertex_idx = get_index(tail_vertex, head_incident_edges)
    next_hop_idx = (tail_vertex_idx + 1) % len(head_incident_edges)
    return head_incident_edges[next_hop_idx]


def find_covering_face(line, ordered_incident_vertices):
    cur_edge = line
    face = [cur_edge]
    found_face = False
    while not found_face:
        next_edge = get_next_edge(cur_edge, ordered_incident_vertices)
        if next_edge == line:
            found_face = True
        else:
            face.append(next_edge)
            cur_edge = next_edge
    return face


def init_ordered_incident_vertices(vertexes, linedefs, adj_list):
    ordered_incident_edges = {}
    for i, vertex in enumerate(vertexes):
        edges = adj_list[get_index(vertex, vertexes)]
        incident_vertices = [vertexes[j] for j in adj_list[i]]
        if not incident_vertices:
            continue
        #cur_vec = np.array(vertex) - np.array(incident_vertices[0])
        #cyc_vertex_list = [(0.0,incident_vertices[0])]
        cyc_vertex_list = []
        for i in range(0, len(incident_vertices)):
            next_vec = np.array(vertex) - np.array(incident_vertices[i])
            #angle = get_angle(cur_vec, next_vec)
            angle = np.arctan2(next_vec[1], next_vec[0])
            cyc_vertex_list.append((angle, incident_vertices[i]))
            cur_vec = next_vec
        #sorted(cyc_vertex_list, reverse=False)
        cyc_vertex_list.sort()

        ordered_incident_edges[vertex] = [x[1] for x in cyc_vertex_list]
    return ordered_incident_edges

def decode_wad_test():
    vertexes = [(200,200), (200,300), (300,300), (300,200),(200,500),(300,500),
               (0,600),(100,600),(0,900),(100,900)]
    max_coord = 1000
    linedefs =\
    [(0,1,None,None),(1,2,None,None),(2,3,None,None),(3,0,None,None),
    (1,4,None,None),(2,5,None,None),(4,6,None,None),(5,7,None,None),(6,7,None,None),
    (8,9,None,None),(7,9,None,None),(8,9,None,None),(6,8,None,None)]
    adj_list = {0:[1,3],1:[0,2,4],2:[1,3,5],3:[0,2],4:[1,6],5:[2,7],
                6:[4,7,8],7:[5,6,9],8:[6,9],9:[7,8]}
    ordered_incident_vertices = init_ordered_incident_vertices(vertexes, linedefs,
                                                         adj_list)
    linedefs_by_vertices = [((vertexes[line[0]], vertexes[line[1]]), line[2]) \
                            for line in linedefs]
    return vertexes, max_coord, linedefs_by_vertices, ordered_incident_vertices

def decode_wad(wad):
    wad_data = get_wad_data(wad)
    wad_index = get_wad_index(wad_data)

    vertexes, max_coord = get_vertexes(chunk_data(wad_index['VERTEXES'], wad_data))
    linedefs, adj_list = get_linedefs(chunk_data(wad_index['LINEDEFS'], wad_data))
    ordered_incident_vertices = init_ordered_incident_vertices(vertexes, linedefs,
                                                         adj_list)
    linedefs_by_vertices = [((vertexes[line[0]], vertexes[line[1]]), line[2]) \
                            for line in linedefs]
    return vertexes, max_coord, linedefs_by_vertices, ordered_incident_vertices

BLACK = (0, 0, 0)
WHITE = "#ffffff"
RED = "#ff0000"

def translate_edge_list(edge_list, offset):
    translated_edges = [(offset+x[0],offset-x[1])\
                        for x in edge_list]
    return translated_edges


def main(wad_path):
    vertexes, max_coord, linedefs, ordered_incident_vertices = \
            decode_wad(wad_path)

    #vertexes, max_coord, linedefs, ordered_incident_vertices = \
    #        decode_wad_test()

    MyImage = Image.new('RGB', (2*max_coord, 2*max_coord), BLACK)
    MyDraw = ImageDraw.Draw(MyImage)
    faces = []
    line_is_wall = {}

    for linedef in linedefs:
        line = linedef[0]
        linedef_flags = linedef[1]
        if (1 & linedef_flags):# | (1 & linedef_flags >> 7):
            line_is_wall[line] = True
            line_is_wall[(line[1],line[0])] =True
            faces.append(find_covering_face(line, ordered_incident_vertices))
        else:
            line_is_wall[line] = False
            line_is_wall[(line[1],line[0])] = False
            faces.append(find_covering_face((line[1],line[0]),
                                            ordered_incident_vertices))
            faces.append(find_covering_face(line, ordered_incident_vertices))
    already_drawn = {}
    out_of_bounds = []

    for face in faces:
        edge_list = []
        hash_string = str(sorted(face))
        if hash_string in already_drawn:
            continue
        already_drawn[hash_string] = True

        white_space = False
        for edge in face:
            edge_list.append(edge[0])
            if not line_is_wall[edge]:
                white_space = True
        translated_edge_list = translate_edge_list(edge_list, max_coord)

        if not white_space:
            out_of_bounds.append(translated_edge_list)
            #MyDraw.polygon(translated_edge_list, fill=RED, outline=WHITE)
        else:
            MyDraw.polygon(translated_edge_list, fill=WHITE,outline=RED)
            #pass

    for nontraversable_polygon in out_of_bounds:
        MyDraw.polygon(nontraversable_polygon, fill=RED, outline=WHITE)

    MyImage.show()


if __name__=="__main__":
    main(get_wad_paths()[2])
