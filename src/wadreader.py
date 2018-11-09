import os, struct
import numpy as np
import numpy.linalg as la

import matplotlib
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw

from flags import Flags

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
    num_sides = int(len(linedefs_data)/LINEDEF_SIZE)
    linedefs = []

    for i in range(num_sides):
        linedef_start = LINEDEF_SIZE*i
        vertex_start_idx = struct.unpack("<h",
                        linedefs_data[linedef_start:linedef_start+2])[DATA_IDX]
        vertex_end_idx = struct.unpack("<h",
                        linedefs_data[linedef_start+2:linedef_start+4])[DATA_IDX]
        flags = struct.unpack("<h",
                        linedefs_data[linedef_start+4:linedef_start+6])[DATA_IDX]
        linedefs.append((vertex_start_idx, vertex_end_idx, flags))
    return linedefs


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


def get_subsectors(subsectors_string):
    prev_end = 0
    subsector = 0
    subsectors = {}
    while prev_end < len(subsectors_string):
        seg_ct = struct.unpack("<h",
                            subsectors_string[prev_end:prev_end+2])[DATA_IDX]
        segs = []
        start = prev_end+2
        end = prev_end+(2*seg_ct)

        for i in range(start, end, 2):
            seg_start = i
            seg_end = seg_start+2
            seg = struct.unpack("<h",
                            subsectors_string[seg_start:seg_end])[DATA_IDX]
            segs.append(seg)

        subsectors[subsector] = segs
        prev_end += seg_ct*2
        subsector+=1

    return subsectors


def chunk_data(offsets, wad_data):
    ''' Extracts the data for a particular lump from full block of wad data.

    @param: offsets: tuple of start and end offsets of the lump
    @param: wad_data: byte string of wad data
    @return: byte string of lump data
    '''
    data_start = offsets[0]
    data_end = data_start+offsets[1]
    return wad_data[data_start:data_end]

def find_interior_point(vertex_start, vertex_end):
    wall =  np.array([(vertex_end[0]-vertex_start[0]),
           (vertex_end[1]-vertex_start[1])])
    mid = np.array([(vertex_end[0]-vertex_start[0])/2,
           (vertex_end[1]-vertex_start[1])/2])
    orth_norm_wall = np.array([wall[1], -wall[0]])/la.norm(wall)

    return orth_norm_wall+mid


def decode_wad(wad):
    wad_data = get_wad_data(wad)
    wad_index = get_wad_index(wad_data)

    vertexes, max_coord = get_vertexes(chunk_data(wad_index['VERTEXES'], wad_data))
    linedefs = get_linedefs(chunk_data(wad_index['LINEDEFS'], wad_data))
    subsectors = get_subsectors(chunk_data(wad_index['SSECTORS'], wad_data))

    BLACK = (0, 0, 0)
    WHITE = "#ffffff"
    OFFSET = max_coord+100
    MyImage = Image.new('RGB', (2*OFFSET, 2*OFFSET), BLACK)
    offset_vertexes = []

    for vertex in vertexes:
        # WAD files and Pillow do not share the same coordinate system
        offset_vertexes.append((OFFSET+vertex[0], OFFSET-vertex[1]))

    MyDraw = ImageDraw.Draw(MyImage)

    # first draw all the lines
    for linedef in linedefs:
        linedef_flags = linedef[2]
        vertex_start = offset_vertexes[linedef[0]]
        vertex_end = offset_vertexes[linedef[1]]

        MyDraw.line([vertex_start, vertex_end], fill=WHITE)

    #for linedef in linedefs:
    #    linedef_flags = linedef[2]
    #    if (1 & linedef_flags) | (1 & linedef_flags >> 7):
    #        vertex_start = vertexes[linedef[0]]
    #        vertex_end = vertexes[linedef[1]]
    #        interior_point = find_interior_point(vertex_start, vertex_end)
    #        offset_interior_point = (OFFSET+interior_point[0],
    #                                 OFFSET-interior_point[1])
    #        ImageDraw.floodfill(MyImage, offset_interior_point, (255,255,255,255))

    for subsector, segs in subsectors.items():
        for seg in segs:
            print(seg)
            linedef = linedefs[seg]
            vertex_start = offset_vertexes[linedef[0]]
            vertex_end = offset_vertexes[linedef[1]]
            MyDraw.line([vertex_start, vertex_end], fill=WHITE)

    MyImage.show()
    #x_idxs, y_idxs = map(list, zip(*vertexes))
    #plt.scatter(x_idxs, y_idxs,s=.04)

    #for linedef in linedefs:
    #    linedef_flags = linedef[2]
    #    if (1 & linedef_flags) | (1 & linedef_flags >> 7):
    #        vertex_start = vertexes[linedef[0]]
    #        vertex_end = vertexes[linedef[1]]
    #        xx, yy = map(list, zip(*[vertex_start, vertex_end]))
    #        plt.plot(xx, yy)
    #plt.show()

    return wad_index


if __name__=="__main__":
    print(decode_wad(get_wad_paths()[0]))
