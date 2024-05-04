import numpy as np
import matplotlib.pyplot as plt
from itertools import count
from shutil import which
from subprocess import check_call # used to launch gmsh
import sys
# plt.rcParams.update({
#     'figure.figsize':(6,6),'figure.dpi':80,'font.size': 9,
#     'grid.linestyle': '--', 'lines.linewidth': 1})

_gmsh_installed = which('gmsh')
PI = np.pi


def formatting(editor, opspy=''):
    global _format
    global _editor
    _editor = editor

    if editor == 'python':
        fflo = ', {:6.4f}'
        fint = ', {:3d}'
        emp = ', {}'
        if opspy != '' and not opspy.endswith('.'):
            opspy += '.'
        
        _format = {
        'circ':		opspy+'patch("circ"{} {} {})\n'.format(', {}', fint*2, fflo*6),
        'rect':		opspy+'patch("rect"{} {} {})\n'.format(', {}', fint*2, fflo*4),
        'fib':		opspy+'fiber({:.4f}, {:.4f}, {:.6f}, {})\n',
        'sec':		opspy+'section("Fiber", {}',
        'S02':      opspy+'uniaxialMaterial("Steel02"{})\n'.format(emp*7),
        'S04':   opspy+'uniaxialMaterial("Steel4"{}, "-iso"{})\n'.format(emp*3, fflo*5),
        'kentpark':	opspy+'uniaxialMaterial("Concrete01"{})\n'.format(emp*5),
        'C04':	opspy+'uniaxialMaterial("Concrete04"{})\n'.format(emp + fflo*4),
        'C07':opspy+'uniaxialMaterial("Concrete07"{})\n'.format(emp + ', {}'*8)# duzelt

        }


    elif editor == 'tcl':
        fflo = ' {:6.4f}'
        fint = ' {:3d}'
        emp = ' {}'
        
        _format = {
        'circ':		'\tpatch circ {} {} {}\n'.format('{}', fint*2, fflo*6),
        'rect':		'\tpatch rect {} {} {}\n'.format('{}', fint*2, fflo*4),
        'fib':		'\tfiber {} {} {}\n'.format(fflo*2, ' {:10.6f}', fint),
        'sec':		'section Fiber {}',
        'S02':      'uniaxialMaterial Steel02 {})\n'.format(emp*7),
        'S04':      'uniaxialMaterial Steel4 {} -iso {}\n'.format(emp*3, fflo*5),
        'kentpark':	'uniaxialMaterial Concrete01 {} {} {} {} {}\n',
        'C04':      'uniaxialMaterial Concrete04 {}\n'.format(emp + fflo*4),
        'C07':      'uniaxialMaterial Concrete07 {}\n'.format(emp + fflo*8)
        }

_editor = 'python'
formatting(_editor, opspy='ops')

class Geometry:
    """
    Makes a geometry.

    Args:
        fname (str): geometry name for this Geometry, this is a file prefix.

    Attributes:
        eshape (str): {'tri', 'quad'}, elements' shape.
            'tri'=triangle; CPS3 or CPE3 in terms of FEM(Finite Element Method).
            'quad'=quadrangle, CPS4 or CPE4 in terms of FEM.
        mesh_size (float): length of elelments's face, for dividing section in
            size of length/mesh_size.
        mesher (str): {'builtin', 'gmsh'}
            'builtin': generate mesh for the rectangle, Tshape and circle sections,
                with its own algorithm.
            'gmsh': an open source 3D finite element mesh generator, needs to be
                installed, for more information visit https://gmsh.info.
        elements (dict in dict): store elements info in grouped form: id as key;
            vertices, center point and area of elements as values
            elements['group1'][el_id] = [[[x1, y1], ..., [xn, yn]], [center_x, center_y], area].
        shape (str): {'rectangle', 'Tshape', 'circle'}, geometrical shape of section,
            needs for Reinforcement and Material object.
    """

    def __init__(self, fname):
        self._fname = fname
        self._b, self._h = None, None # for Reinforcement
        self._D, self._d3 = None, None
        self.elements = {'confined':{}, 'unconfined':{}}


    def set_elements_from_inp(self, inp_fname, part_names, shape='', **dimensions):
        """
        Defines elements dictionary from inp file as to be key, part_names is used.
        Args:
            inp_fname (str): file name (without extension) of which have inp extension
            part_names (str): name of the physical surface which is defined when meshing done.
            shape (str): {'rectangle, Tshape, circle'}, geometrical shape of section.
            **dimensions (float):must be given in set of following,
                b, h, d3: width, height, cover thickness;
                D, d3: diameter, cover thickness,
                This is to be used in instances of Reinforcement and Material classes.
        Note: element type must be 3 noded triangle(CPS3, CPE3) or 4 noded quadrangle(CPS4, CPE4)
        """

        self.shape = shape
        for k, v in dimensions.items():
            if v!=None:
                setattr(self, '_'+k, v)

        for k, v in dimesions.items():
            setattr(self, k, v)

        self.__read_inp(part_names, fname=inp_fname)


    def set_mesh_info(self, eshape, mesh_size, mesher='builtin'):
        """
        Set mesh information for generation of mesh.
        Args:
            eshape (str): {'tri', 'quad'}, elements' shape.
                'tri'=triangle; CPS3 or CPE3 in terms of FEM(Finite Element Method).
                'quad'=quadrangle, CPS4 or CPE4 in terms of FEM.
            mesh_size (float): length of elelments's face, for dividing section
                in size of length/mesh_size
            mesher (str): {'builtin', 'gmsh'}
                'builtin': generate mesh for the rectangle, Tshape and circle
                    sections, with its own algorithm
                'gmsh': an open source 3D finite element mesh generator, needs
                    to be installed, for more information visit https://gmsh.info.
        """

        self.eshape = eshape
        self.mesh_size = mesh_size
        self.mesher = mesher

        if mesher=='gmsh' and not _gmsh_installed:
            self.mesher = 'builtin'
            print('gmsh is not installed, if mesher will be given as gmsh, mesher\
            will be changed from gmsh to builtin. If you want to use gmsh visit \
            https://gmsh.info, install.')


    def Rectangle(self, b, h, d3):
        """
        Generates mesh and store elements info in dict of confined and
            unconfined in elements dict: id as key; vertices, center point
            and area of elements as values.
            elements['group1'][el_id] = [[[x1, y1], ..., [xn, yn]], [center_x, center_y], area]
        Args:
            b (float): width of section
            h (float): height of section
            d3 (float): cover thickness
        """

        self._b, self._h, self._d3 = b, h, d3
        self.shape = 'rectangle'
        self.height = [b, h]
        self.section_center = [b/2, h/2]

        if self.mesher=='gmsh':
            self._section_part = {
            'Point': [[d3, d3], [b - d3, d3], [d3, h - d3], [b - d3, h - d3],
                                                [0, 0], [b, 0], [0, h], [b, h]],
            'Line': [[1, 2], [3, 4], [1, 3], [2, 4], [5, 6], [7, 8], [5, 7], [6, 8]],
            'Line Loop': [[1, 4, -2, -3],
                          [5, 8, -6, -7, 1, 4, -2, -3]],
            'Physical Surface': {'confined':[1], 'unconfined':[2]}
            }


        elif self.mesher=='builtin':
            self._section_part = {
                    'confined'  : [[d3, d3, b - d3, h - d3]],
                    'unconfined': [[0, 0, b, d3],
                                   [0, h - d3, b, h],
                                   [0, d3, d3, h - d3],
                                   [b - d3, d3, b, h - d3]]
            }
            
                    
    def Circle(self, D, d3):
        """
        Generates mesh and store elements info in dict of confined and
            unconfined in elements dict: id as key; vertices, center point and
            area of elements as values.
            elements['group1'][el_id] = [[[x1, y1], ..., [xn, yn]], [center_x, center_y], area]
        Args:
            D (float): diameter of section
            d3 (float): cover thickness
        """

        self._D, self._d3 = D, d3
        self.shape = 'circle'
        self.height = [D, D]
        self.section_center = [D/2, D/2]


        
        if self.mesher == 'gmsh':
            x1 = (D/2-d3)*np.cos(PI/3)
            y1 = (D/2-d3)*np.sin(PI/3)
            x2 = D/2*np.cos(PI/3)
            y2 = D/2*np.sin(PI/3)
            c = D/2
            
            self._section_part = {
            'Point': [[c, c], [D-d3, c], [c-x1, c+y1], [c-x1, c-y1], [D, c],
                                                [c-x2, c+y2], [c-x2, c-y2]],

            'Line': [[2, 1, 3], [3, 1, 4], [4, 1, 2], [5, 1, 6], [6, 1, 7], [7, 1, 5]],

            'Line Loop': [[1, 2, 3], [4, 5, 6, 1, 2, 3]],

            'Physical Surface': {'confined': [1], 'unconfined': [2]}
            }


    def Tshape(self, b, b1, b2, h, hf, d3):
        """
        Generates mesh and store elements info in dict of confined and
            unconfined in elements dict: id as key; vertices, center point and
            area of elements as values.
            elements['group1'][el_id] = [[[x1, y1], ..., [xn, yn]], [center_x, center_y], area]
        Args:
            b (float): width of the section's body part
            b1 (float): width of left arm of the section
            b2 (float): width of right arm of the section
            hf (float): height of up part of the section or height of floor
            h (float): height of section

            --------------------------   -      -
            |                        |   |hf    |
            --------          --------   -      |
              b1   |          |   b2            |
                   |          |                 | h
                   |          |                 |
                   |          |                 |
                   |          |                 |
                   ------------                 -
                        b
        """

        self._b, self._b1, self._b2 = b, b1, b2
        self._h, self._hf, self._d3 = h, hf, d3
        self.shape = 'Tshape'
        self.height = [b1 + b + b2, h]
        h2 = h - hf
        
        area = [(b1 + b + b2)*hf, b*h2]
        xc = area[0]*(b1 + b + b2)/2 + area[1]*(b1 + b/2)
        yc = area[0]*(h2 + hf/2) + area[1]*h2/2
        self.section_center = np.array([xc, yc])/sum(area)

        if self.mesher=='gmsh':
            self._section_part = {
            'Point':[[b1 + d3, d3], [b1 + b - d3, d3], [0, h2 + d3],
                         [b1 + d3, h2 + d3], [b1 + b - d3, h2 + d3],
                     [b1 + b + b2, h2 + d3], [0, h - d3], [b1 + b + b2, h - d3],
                                                  [b1, 0], [b1 + b, 0], [0, h2],
                     [b1, h2], [b1 + b, h2], [b1 + b + b2, h2], [0, h], [b1 + b + b2, h]],

            'Line': [[1, 2], [3, 4], [5, 6], [7, 8], [1, 4], [2, 5], [3, 7], [6, 8],
                     [9, 10], [11, 12], [13, 14], [15, 16], [9, 12], [10, 13],
                     [11, 3], [14, 6], [7, 15], [8, 16]],

            'Line Loop': [[1, 6, 3, 8, -4, -7, 2, -5],
                          [10, -13, 9, 14, 11, 16, -3, -6, -1, 5, -2, -15],
                          [4, 18, -12, -17]],

            'Physical Surface': {'confined': [1], 'unconfined': [2, 3]}
            }


        elif self.mesher=='builtin':
            self._section_part = {
                    'confined'  : [[b1 + d3, d3, b1 + b - d3, h2 + d3],
                                    [0, h2 + d3, b1 + b + b2, h - d3]],
                    'unconfined': [[b1, 0, b1 + b, d3],
                                   [0, h2, b1 + d3, h2 + d3],
                                   [b1 + b - d3, h2, b1 + b + b2, h2 + d3],
                                   [0, h - d3, b1 + b + b2, h],
                                   [b1, d3, b1 + d3, h2],
                                   [b1 + b - d3, d3, b1 + b, h2]]
            }


    def T2shape(self, bf, tf, tw, h):
        """
        Generates mesh and store elements info in dict of confined and
            unconfined in elements dict: id as key; vertices, center point and
            area of elements as values.
            elements['group1'][el_id] = [[[x1, y1], ..., [xn, yn]], [center_x, center_y], area]
        Args:
            bf (float): flange width
            tf (float): flange thickness
            tw (float): web thickness
            h (float): height

             _________________    ___      ___
            |______     ______|   _|_  tf   |
                   |   |                    |
                   |   |                    |
                   |   |                    | h
                   |   |                    |
             ______|   |______              |
            |_________________|            _|__

                   |tw |
            |-------bf--------|
        """

        self.shape = 'T2shape'
        self.height = [bf, h]
        self.section_center = [bf/2, h/2]
        b2 = (bf - tw)/2

        
        if self.mesher=='gmsh':
            self._section_part = {
            'Point':[[0, 0], [bf, 0], 
                     [0, tf], [b2, tf], [b2 + tw, tf], [bf, tf],
                     [0, h - tf], [b2, h - tf], [b2 + tw, h - tf], [bf, h - tf],
                     [0, h], [bf, h]],

            'Line': [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12],
                     [1, 3], [2, 6], [4, 8], [5, 9], [7, 11], [10, 12]],
            
            'Line Loop': [[1, 8, -3, 10, 5, 12, -6, -11, 4, -9, -2, -7]],

            'Physical Surface': {'single': [1]}
            }


        elif self.mesher=='builtin':
            self._section_part = {
                    'single': [[0, 0, bf, tf],
                               [b2, tf, b2 + tw, h-tf],
                               [0, h - tf, bf, h]] 
            }

        self.elements = {'single':{}}


    def generate_mesh(self):
        """Generates mesh"""
        
        if self.mesher == 'gmsh':
            if self.shape == 'circle':
                self.__circle_gmsh(self._D, self._d3, self.eshape, self.mesh_size)
            
            else:
                self.__mesh_gmsh(self._section_part, self.eshape, self.mesh_size)
                self.__read_inp(list(self._section_part.keys()))
            self.__rearrange()

        elif self.mesher=='builtin':
            if self.shape == 'circle':
                if self.eshape == 'tri':
                    self.__circle_builtin_tri(self._D, self._d3, self.mesh_size)
                else:
                    self.__circle_builtin_quad(self._D, self._d3, self.mesh_size)
                self.__rearrange()
            
            else:
                self.__mesh_builtin_rec(self._section_part, self.eshape, self.mesh_size)


    def plot(self, elnum=False, ax=None, legend=False, save=False, ecolor=None):
        """
        Plot elements.
        Args:
            elnum (bool): if True, elements id will be shown on the elements.
            ax (None or matplotlib axis): for setting matplotlib by yourself
            legend (bool): if True elements keys will shown as label.
            save (bool): if True figure will be saved.
        """

        elements = self.elements
        nel = len(elements) 
        if not ax:
            fig, ax = plt.subplots()
            ax.axis('equal')
            ax.axis('off')
            show = True 
        else:
            show = False

        ax.plot()
        color_list = ['g', 'r', 'm', 'c', 'y', 'b', 'k']
        alpha_list = [0.3, 0.45, 0.5, 0.5, 0.5, 0.5, 0.5]
        fc_list = ['gray', 'gray', 'y', 'c', 'm', 'r', 'g']
        
        if not ecolor:
            ec_list = fc_list
        else:
            ec_list = [ecolor for i in range(nel)]

        for j, i in enumerate(elements.keys()):

            for key, nodes in elements[i].items():
                p = plt.Polygon(nodes.T, fill=True, ec=ec_list[j%(nel + 1)],
                                fc=fc_list[j%(nel + 1)], ls='-',
                                alpha=alpha_list[j%(nel + 1)])
                ax.add_patch(p)

        if legend:
            for j, i in enumerate(elements.keys()):
                ax.plot([], [], label=i, ms=10, marker='o', mfc=color_list[j],
                        alpha=alpha_list[j], c=fc_list[j], mew=6)
            plt.legend(labelspacing=1.5)

        if save:
            plt.savefig(self._fname+'_geo.png', bbox_inches="tight", pad_inches = 0)

        if show:
            plt.show()

    # NOTE: metreye cevrim kismini kontrol et
    def to_ops(self, fibTag, GJ=None, **matTags):
        """
        m
        """
        
        if _editor == 'python':
            GJ_txt = 1*(GJ != None)*f', "-GJ", {GJ}' + ')\n'
        else:
            GJ_txt = 1*(GJ != None)*f' -GJ {GJ}' + ' {\n'
        ops_txt = _format['sec'].format(fibTag) + GJ_txt 
        tab_str = '\t'*(_editor == 'tcl')
        
        if self.mesher == 'gmsh' and self.shape != 'circle':
            print('mesher must be "builtin"')
            return None
        
        if self.shape == 'circle':
            D = self._D
            d3 = self._d3
            r = D/2 - d3 # radius of confined concrete
            n_cover = d3/self.mesh_size
            n_cover = int(n_cover) + 1*(n_cover != int(n_cover))
            n_circ = int(PI*D/(self.mesh_size)) + 1
            n_core = int((r - d3)/self.mesh_size) + 1
            
            D = D/1e3
            d3 = d3/1e3
            r = r/1e3
            args_dict = {
                'unconfined': [n_circ, n_cover, D/2, D/2, r - d3, r, 0, 360],
                'confined': [n_circ, n_core, D/2, D/2, 0., r - d3, 0, 360]
            }
            
            for key, values in args_dict.items():
                tag = matTags.get(key)
                if tag == None:
                    tag = 'tag'
                ops_txt += f'{tab_str}# {key.lower()}\n'
                ops_txt += _format['circ'].format(tag, *values)

        else:
            for key, values in self._section_part.items():
                tag = matTags.get(key)
                if tag == None:
                    tag = 'tag'
                
                ops_txt += f'{tab_str}# {key.lower()}\n'
                for vert in values:
                    x1, y1, x2, y2 = vert
                    new_vert = [y1, x1, y2, x2]
                    nx = (x2 - x1)/self.mesh_size
                    nx = int(nx) + 1*(nx != int(nx))
                    
                    ny = (y2 - y1)/self.mesh_size
                    ny = int(ny) + 1*(ny != int(ny))
                   
                    new_vert = [i/1e3 for i in new_vert] 
                    ops_txt += _format['rect'].format(tag, ny, nx, *new_vert)

        return ops_txt
    

    def __circle_builtin_tri(self, D, d3, mesh_size): # NOTE: cover_el_number' a bk
        """
        Meshes the section and sets elements dict using built-in mesh algorithm
            for triangular element shape."""

        r = D/2 - d3 # radius of confined concrete
        cover_el_number = d3//mesh_size
        cover_el_number = 1 if cover_el_number==0 else cover_el_number

        rad_unconfined = np.linspace(PI, -PI, int(PI*D/(d3/cover_el_number)) + 1)
        y_unconfined = [D/2+(D/2-d3*n/cover_el_number)*np.sin(rad_unconfined)
                                          for n in range(cover_el_number + 1)]
        x_unconfined = [D/2+(D/2-d3*n/cover_el_number)*np.cos(rad_unconfined)
                                          for n in range(cover_el_number + 1)]


        iter0 = count(1)

        part1 = {'unconfined':[x_unconfined, y_unconfined, len(x_unconfined)-1]}
        for key, [x, y, lenx] in part1.items():

            for ii in range(lenx):

                for i in range(len(x[ii+1])-1):
                    xyArray = np.array([[x[ii][i], x[ii+1][i+1], x[ii+1][i]],
                                        [y[ii][i], y[ii+1][i+1], y[ii+1][i]]])
                    self.elements[key][next(iter0)] = xyArray

                    xyArray = np.array([[x[ii][i], x[ii][i+1], x[ii+1][i+1]],
                                        [y[ii][i], y[ii][i+1], y[ii+1][i+1]]])
                    self.elements[key][next(iter0)] = xyArray



        if (int(2*r/mesh_size) + 1)%2 == 1:
            leny_1 = int(2*r/mesh_size) + 1

        else:
            leny_1 = int(2*r/mesh_size) + 2

        y_confined_1 = np.linspace(d3, 2*r + d3, leny_1)
        y_confined_1 = np.delete(y_confined_1, [0, len(y_confined_1) - 1])

        x_arc_right = (r**2 - (y_confined_1 - r - d3)**2)**0.5 + r + d3
        x_arc_left = -(r**2 - (y_confined_1 - r - d3)**2)**0.5 + r + d3

        x_el_number_confined = [len(y_confined_1) + 1 - i
                                for i in range(0, leny_1//2 - 1)]

        x_el_number_confined = [*list(reversed(x_el_number_confined)),
                                        leny_1, *x_el_number_confined]

        x_confined_1 = np.array([np.linspace(s, e, x_el_number_confined[i])
                    for i, [s, e] in enumerate(zip(x_arc_left, x_arc_right))],
                                 dtype=object)



        alpha = np.arctan((x_confined_1[-1][-1]-r)/(y_confined_1[-1]-r))
        rad_confined = np.linspace(PI/2 + alpha*0.8, PI/2 - alpha*0.8,
                                    len(x_confined_1[-1]), endpoint=False)

        rad_confined = np.delete(rad_confined, 0)
        y_arc_up = r + d3 + r*np.sin(rad_confined)
        y_arc_down = r + d3 - r*np.sin(rad_confined)
        x_arc_updown = r + d3 + r*np.cos(rad_confined)

        y_confined = np.array([[y_confined_1[i]
                    for y in range(len(x_confined_1[i]))]
                    for i in range(len(x_confined_1))], dtype=object)

        y_confined = np.array([y_arc_down, *y_confined, y_arc_up], dtype=object)
        x_confined = np.array([x_arc_updown, *x_confined_1, x_arc_updown], dtype=object)


        part2 = {'confined':[x_confined, y_confined, len(x_confined)-1]}
        for key, [x, y, lenx] in part2.items():

            for ii in range(lenx):

                if y[ii][0] - d3>=r:
                    for i in range(len(x[ii+1])):
                        xyArray = np.array([[x[ii][i], x[ii][i+1], x[ii+1][i]],
                                            [y[ii][i], y[ii][i+1], y[ii+1][i]]])

                        self.elements[key][next(iter0)] = xyArray

                        if i!=len(x[ii+1])-1:
                            xyArray = np.array([[x[ii][i+1], x[ii+1][i+1], x[ii+1][i]],
                                                [y[ii][i+1], y[ii+1][i+1], y[ii+1][i]]])

                            self.elements[key][next(iter0)] = xyArray

                else:
                    for i in range(len(x[ii])):
                        xyArray = np.array([[x[ii+1][i], x[ii+1][i+1], x[ii][i]],
                                            [y[ii+1][i], y[ii+1][i+1], y[ii][i]]])

                        self.elements[key][next(iter0)] = xyArray


                        if i!=len(x[ii])-1:
                            xyArray = np.array([[x[ii][i], x[ii][i+1], x[ii+1][i+1]],
                                                [y[ii][i], y[ii][i+1], y[ii+1][i+1]]])
                            
                            self.elements[key][next(iter0)] = xyArray


    def __circle_builtin_quad(self, D, d3, mesh_size):
        """
        Meshes the section and sets elements dict using built-in mesh algorithm
            for quadrilateral element shape."""

        iter0 = count(1)
        r = D/2 - d3
        elnum = r/mesh_size
        elnum = int(elnum) + 1*(elnum != int(elnum))
        elnum1 = elnum//2
        elnum2 = elnum - elnum1

        ref1 = r*elnum1/elnum
        dx = r/elnum
        d45 = (r*np.cos(PI/4) - ref1)/elnum2

        elnum_cov = d3/mesh_size
        elnum_cov = int(elnum_cov) + 1*(elnum_cov != int(elnum_cov))
        
        # rectangle part mesh
        ref2 = D/2 - ref1
        ILxy = [ref2,  ref2, ref2 + 2*ref1, ref2 + 2*ref1]
        mesh_object = self.__rectangle_part_mesh(
                *ILxy, 'quad', nx=2*elnum1-2, ny=2*elnum1-2)
        
        for [nodes, [centerX, centerY], area] in mesh_object:
            self.elements['confined'][next(iter0)] = nodes
        

        # arc part mesh
        nodesx = [0]*(elnum2 + elnum_cov + 1)
        nodesy = [0]*(elnum2 + elnum_cov + 1)

        for i in range(elnum2 + elnum_cov + 1):
            if i < elnum2:
                x1 = ref1 + dx*i
                y1 = 0
                x2 = ref1 + d45*i
                y2 = x2

            else:
                x1 = r + d3*(i - elnum2)/elnum_cov
                y1 = 0
                x2 = x1*np.cos(PI/4)
                y2 = x2
   
            y = np.linspace(y1, y2, elnum2)

            if i == 0:
                x = x1*np.ones(elnum2)

            else:
                ar = np.square(np.array([[x1, y1], [x2, y2]]))
                res = np.linalg.solve(ar, [1, 1])
                a2, b2 = 1/res
                x = (a2*(1 - y**2/b2))**0.5 # ellipse

            nodesx[i] = np.append(x, y[::-1][1:])
            nodesy[i] = np.append(y, x[::-1][1:])

            nodesx[i] = np.append(nodesx[i], -nodesx[i][::-1][1:])
            nodesy[i] = np.append(nodesy[i], nodesy[i][::-1][1:])

            nodesx[i] = D/2 + np.append(nodesx[i], nodesx[i][::-1][1:])
            nodesy[i] = D/2 + np.append(nodesy[i], -nodesy[i][::-1][1:])

        nodesx, nodesy = np.array(nodesx), np.array(nodesy)

        for i in range(elnum2 + elnum_cov):
            if i < elnum2:
                key = 'confined'

            else:
                key = 'unconfined'
            for j in range(8*elnum2 - 8):
                (x1, x2), (x3, x4) = nodesx[i:i+2, j:j+2]
                (y1, y2), (y3, y4) = nodesy[i:i+2, j:j+2]

                self.elements[key][next(iter0)] = np.array([[x1, x2, x4, x3], 
                                                       [y1, y2, y4, y3]])


    def __rectangle_part_mesh(self, Ix, Iy, Lx, Ly, eshape, nx=-1, ny=-1, mesh_size=-1):
        """
        Yield [[coorX, coorY], [centerX, centerY], area]
        Args:
            Ix, Iy (float): left down corner vertices' coordinates.
            Lx, Ly (float): right upper corner vertices' coordinates.
            nx (int): number of element along x-axis.
            ny (int): number of element along y-axis.
        """

        if mesh_size != -1 and nx == -1 and ny == -1:
            nx = (Lx - Ix)/mesh_size
            nx = int(nx) + (nx != int(nx))
            ny = (Ly - Iy)/mesh_size
            ny = int(ny) + (ny != int(ny))

        dx = (Lx - Ix)/nx
        dy = (Ly - Iy)/ny
        area = abs(dx*dy)/(1 + (eshape=='tri'))

        for yi in range(ny):
            for xi in range(nx):
                x1, x2 = Ix + dx*xi, Ix + dx*(xi + 1)
                y1, y2 = Iy + dy*yi, Iy + dy*(yi + 1)

                if eshape == 'tri':
                    ver = np.array([[[x1, x2, x1], [y1, y1, y2]],
                                    [[x2, x2, x1], [y1, y2, y2]]])

                    for side in range(2):
                        xc = x1 + dx*(side + 1)/3
                        yc = y1 + dy*(side + 1)/3

                        yield [ver[side], [xc, yc], area]

                else: # quad mesh
                    xc = x1 + dx/2
                    yc = y1 + dy/2

                    yield [np.array([[x1, x2, x2, x1], [y1, y1, y2, y2]]),
                                    [xc, yc], area]


    def __mesh_builtin_rec(self, section_dict, eshape, mesh_size):
        """
        m
        """
        
        n = 1
        self.elements = {}      # element tag and nodes pair
        self.eCenters = {}      # element center
        self.eAreas = {}        # element area
        section_center = [0, 0]
        section_area = 0        # total section area

        for part, pvertices in section_dict.items():
            self.elements[part] = {}
            pcenter_list = [[], []]
            parea_list = []

            for vert in pvertices:
                # finding center of section
                edX = vert[2] - vert[0]     # dimension x
                edY = vert[3] - vert[1]     # dimension y
                eArea = abs(edX*edY)        # part area
                ecX = (vert[2] + vert[0])/2 # center x
                ecY = (vert[3] + vert[1])/2 # center y

                section_area += eArea
                section_center[0] += ecX*eArea
                section_center[1] += ecY*eArea

                # meshing
                mesh_object = self.__rectangle_part_mesh(
                        *vert, eshape, mesh_size=mesh_size)

                for [nodes, [centerX, centerY], area] in mesh_object:
                    self.elements[part][n] = nodes
                    pcenter_list[0].append(centerX)
                    pcenter_list[1].append(centerY)
                    parea_list.append(area)
                    n += 1

            self.eCenters[part] = np.array(pcenter_list)
            self.eAreas[part] = np.array(parea_list)

        if not hasattr(self, 'section_center') :
            self.section_center = [i/section_area for i in section_center]


    def __mesh_gmsh(self, gmsh_dict, eshape, mesh_size):
        """
        Meshes the section. Write a geo file as a input file to the gmsh mesh
            generator, and an inp file is generated which include meshed section's
            elements. The files is saved with prefix which is the given fname.
        Args:
            gmsh_dict (dict): consist of elementary entities and physical
                surface to create geo file.
        """

        fname = self._fname
        points_g = []
        lines_g = []
        curves_g = []
        physicalsurfaces_g = []

        if eshape=='quad':
            mesh_size = 2*mesh_size

        for i, p in enumerate(gmsh_dict['Point'], start=1):
            text = 'Point(%i) = {%f, %f, %f, %f};'%(i, *p, 0, 1)
            points_g.append(text)

        for i, l in enumerate(gmsh_dict['Line'], start=1):
            if len(l) == 3:
                text = 'Circle(%i) = {'%(i)+', '.join(str(ll) for ll in l)+'};'

            else:
                text = 'Line(%i) = {'%(i)+', '.join(str(ll) for ll in l)+'};'

            lines_g.append(text)

        for i, c in enumerate(gmsh_dict['Line Loop'], start=1):
            text = 'Line Loop(%i) = {'%(i) + ', '.join(str(cc) for cc in c) + '};'
            text += '\nPlane Surface(%i) = {%i};'%(i, i)
            curves_g.append(text)

        for key, ps in gmsh_dict['Physical Surface'].items():
            text = "Physical Surface('%s') = {"%(key)+", ".join(str(i) for i in ps)+"};"
            physicalsurfaces_g.append(text)

        mesh = ['Mesh.CharacteristicLengthMin = %f;'%(mesh_size),
                'Mesh.CharacteristicLengthMax = %f;'%(mesh_size)]

        if eshape == 'quad':
            mesh.append('Mesh.RecombineAll = 1; //turns on quads')
            mesh.append('Mesh.SubdivisionAlgorithm = 1; // quadrangles only')

        fgeo = open('%s.geo'%fname, 'w')
        aspects = [points_g, lines_g, curves_g, physicalsurfaces_g, mesh]
        fgeo.write('\n'.join([i for res in aspects for i in res]))
        fgeo.close()
        runstr = "%s %s -2 -o %s"%('gmsh', fname+'.geo', fname+'.inp')
        timeout = 20
        check_call(runstr, timeout=timeout, shell=True)


    def __read_inp(self, part_names, fname=-1):
        """
        Reads in the mesh from a inp file. All elements assigned to their
            physical surfaces are in grouped in elements dict.

        Args:
            fname (str or -1): file name to read, must not include '.inp' extension.
                if fname is -1, then the given fname will be used.
            part_names (str in list or -1): the list of names which are physical
                surfaces's names.
                if part_names is -1, then will be taken as ['confined', 'unconfined'].
        """

        if fname == -1:
            fname = self._fname

        f = open(fname+'.inp', 'r')
        mode = None
        set_name = None
        set_type = None

        Dict_NodeIDs = {}
        Dict_ElemIDs = {}
        Dict_Set = {i: [] for i in part_names}

        # read in input file
        for line in f:
            if line[0] != '*':
                if mode == 'nmake':
                    L = line.split(',')
                    L = [a.strip() for a in L]
                    nnum, x, y = int(L[0]), float(L[1]), float(L[2])
                    Dict_NodeIDs[nnum] = (x, y)

                elif mode == 'emake':
                    L = line.split(',')
                    L = [int(a.strip()) for a in L]
                    enum = L[0]
                    Dict_ElemIDs[enum] = L[1:]

                elif mode == 'set':
                    L = line.split(',')
                    L = [a.strip() for a in L]
                    L = [int(a) for a in L if a != '']
                    if set_type == 'E':
                        if set_name in part_names:
                            for a in L:
                                if a in Dict_ElemIDs:
                                    Dict_Set[set_name].append(a)

            # mode setting
            if '*Node' in line or '*NODE' in line:
                mode = 'nmake'

            elif '*Element' in line or '*ELEMENT' in line:
                L = line.split(',') # split it based on commas
                e = L[1].split('=')
                etype = e[1]

                # exclude T elements made in gmsh
                if etype[0] != 'T':
                    e = L[2].split('=')
                    set_name = e[1].strip()
                    set_type = 'E'
                    mode = 'emake'

                else:
                    mode = None

            elif '*ELSET' in line:
                L = line.split(',')
                e = L[1].split('=')
                set_name = e[1].strip()
                set_type = 'E'
                mode = 'set'
        f.close()


        self.elements = {i: {} for i in part_names}
        for part, pelements in Dict_Set.items():
            for el in pelements:
                nodes = Dict_ElemIDs[el]
                xyArray = np.array([[Dict_NodeIDs[a][i] for a in nodes]
                                                    for i in range(2)])
                self.elements[part][el] = xyArray


    def __rearrange(self):
        """

        """
        builtin_shapes = ['rectangle', 'circle', 'Tshape', 'T2shape']
        lenvert = 3 + (self.eshape=='quad')
        self.eCenters = {}
        self.eAreas = {}

        # rearranging data for numpy multification
        for part, values in self.elements.items():
            pcenter_list = [[], []]
            parea_list = []

            for el, [x, y] in values.items():
                pcenter_list[0].append(np.sum(x)/lenvert)
                pcenter_list[1].append(np.sum(y)/lenvert)
                
                add = x[-1]*y[0] - x[0]*y[-1]
                parea_list.append(0.5*abs(x[:-1]@y[1:] - x[1:]@y[:-1] + add))

            self.eCenters[part] = np.array(pcenter_list)
            self.eAreas[part] = np.array(parea_list)


        if not hasattr(self, 'section_center') :
            self.section_center = np.array([0., 0.])
            section_area = 0
            
            for part in self.elements.keys():
                pAreas = np.sum(self.eAreas[part])
                section_area += pAreas
                self.section_center += self.eCenters[part]@self.eAreas[part]
            
            self.section_center = self.section_center/section_area


    def __set_quad_info_gmsh(self, node_list):
        """
        Return vertices, center point and area of quadrilateral elements.
            Faster when setting from inp file.
        Args:
            node_list (array with shape 4x2): the vertices"""

        [[xi, yi], [xj, yj], [xk, yk], [xl, yl]] = node_list
        return [node_list,
                [(xi+xj+xk+xl)/4, (yi+yj+yk+yl)/4],
                0.5*abs(xi*(yj-yl) + xj*(yk-yi) + xk*(yl-yj) + xl*(yi-yk))]



class Reinforcement: # mander_wi2 hesabi kontrol edilmelidir.
    """
    Makes a reinforcement.

    Args:
        fname (str): reinforcement name for this Reinforcement, this is a file prefix.
        phiw (float) diameter of transversal rebar.
        origin (tuple): (x, y), reference coordinate.
        shape (str): shape of section {'rectangle', 'Tshape' and 'circle'}
        **dimensions (float):must be given in set of following,
            b, h, d3: width, height, cover thickness;
            D, d3: diameter, cover thickness

    Attributes:
        rebar_dict (dict):
        sumA (float): sum of the area of longitudinal rebars
        """

    def __init__(self, fname, phiw, origin, shape='', **dimensions):
        self.rebar_dict = {}
        self._phiw = phiw
        self._origin = origin
        self._mander_wi2 = 0
        self._fname = fname

        for k, v in dimensions.items():
            if v!=None:
                setattr(self, '_'+k, v)

        self._shape = shape
        self._sumA = 0

        if self._shape in ['rectangle', 'Tshape']:
            bc = self._b - 2*self._d3 - phiw
            hc = self._h - 2*self._d3 - phiw
            self._core_dimensions = [bc, hc]

        elif self._shape == 'circle':
            self._core_dimensions = self._D - 2*self._d3 - phiw

        else:
            print("shape must be in ['rectangle', 'Tshape' and 'circle']")

    def to_ops(self, rebarTag):
        """Return OpenSees commands"""
        
        ops_txt = '\t'*(_editor == 'tcl') + '# rebar\n'
        for x, y, phi, A in self.rebar_dict.values():
            ops_txt += _format['fib'].format(y/1e3, x/1e3, A/1e6, rebarTag)
        return ops_txt + '}\n'*(_editor == 'tcl')

    @property
    def sumA(self):
        """Return sum of the area of longitudinal rebars"""

        self._sumA = sum(i[3] for i in self.rebar_dict.values())
        return self._sumA


    def positioning_from_string(self, rebar, sep_layer='|', 
                                sep_in_layer='+', sep_num_phi='@'):
        """
        Generate 2d rebar array for Reinforcement's
            positioning_rectangular_phi method.
        Args:
            rebar(str): representing rebar plane in str format, see Example.
            sep_layer(str): a sign to seperate layers vertically.
            sep_in_layer(str): a sign to seperate rebar group according to
                dimension of rebar.
            sep_num_phi(str): a sign to seperate number of rebar and dimension of them.
        Example:
            rebar_from_string('2@12+2@16|2@12|2@16',
                              sep_layer='|', sep_in_layer='+', sep_num_phi='@')
            >>> [[12.0, 16.0, 16.0, 12.0],
                [12.0, 12.0],
                [16.0, 16.0]]
        """
        rebar = rebar.split(sep_layer)
        rebar_array = [[] for _ in range(len(rebar))]
        for ind_i, i in enumerate(rebar):
            layer_str_list = i.split(sep_in_layer)
            layers = [j.split(sep_num_phi) for j in layer_str_list]
            layers = [[int(n), int(p)] for n, p in layers]
            
            if len(layers)==1:
                num, phi = layers[0]
                rebar_array[ind_i] = [phi for _ in range(num)]

            else:
                for num, phi in layers[1:]:
                    [rebar_array[ind_i].append(phi) for _ in range(num)]
                num, phi = layers[0]

                for _ in range(num):
                    if _%2==0:
                        rebar_array[ind_i].insert(0, phi)

                    else:
                        rebar_array[ind_i].append(phi)
        
        self.positioning_rectangular_phi(rebar_array)
        return rebar_array


    def positioning_rectangular_column(self, phi, nx, ny):
        """
        Positions rebars in the rectangular pattern and set rebar_dict attribute.
        More convenient for usual column sections' reinforcement plane.
        Args:
            phi (float): diameter of all rebars
            nx (int): number of rebars along x-axis.
            ny (int): number of rebars along y-axis."""

        self.rebar_dict = {}
        b, h, d3, phiw, xc, yc = self._b, self._h, self._d3, self._phiw, *self._origin
        As0 = PI*(phi)**2/4

        lx = b-2*(d3+phiw)-phi
        dlx = lx/(nx-1)

        ly = h-2*(d3+phi/2+phiw)
        dly = ly/(ny-1)

        # for mander material model
        self._mander_wi2 = (2*(nx-1))*((dlx-phi)**2) + (2*(ny-1))*((dly-phi)**2)

        iter0 = count(1)
        for y in [1, ny]:

            for x in range(1, nx+1):
                self.rebar_dict[next(iter0)] = [xc+d3+phi/2+phiw+(x-1)*dlx,
                                                yc+d3+phi/2+phiw+(y-1)*dly,
                                                phi, As0]

        for x in [1, nx]:

            for y in range(2, ny):
                self.rebar_dict[next(iter0)] = [xc+d3+phi/2+phiw+(x-1)*dlx,
                                                yc+d3+phi/2+phiw+(y-1)*dly,
                                                phi, As0]


    def positioning_rectangular_beam(self, rebar_list, where=''):
        """
        Positions rebars in the rectangular pattern and set rebar_dict attribute.
        More convenient for usual beam sections' reinforcement plane.
        Args:
            phi (float): diameter of all rebars

            rebar_list(array with shape Nx2): [[number, phi], [number2, phi2], *];
                number must be at least 2, else use add_rebar method. N represent
                number of layer along y-axis.
            where (str): {up, down} if N is 1, specification is required."""

        self.rebar_dict = {}
        rebar_list = rebar_list[::-1]
        b, h, d3, phiw = self._b, self._h, self._d3, self._phiw
        xc, yc = self._origin
        iter0 = count(1)
        ny = len(rebar_list)
        phi_first = rebar_list[0][1]

        if ny == 1:
            if where == 'down':
                dly, y0, sign = 0, 0, 1

            elif where == 'up':
                dly, y0, sign = 0, h, -1

        else:
            phi_last = rebar_list[-1][1]
            ly = h - 2*(d3 + phiw) - (phi_first + phi_last)/2
            dly, sign, y0 = ly/(ny - 1), 1, 0

        wi2 = 0 # for mander material model
        for y, [nx, phi] in enumerate(rebar_list):
            As0 = PI*(phi)**2/4
            lx = b - 2*(d3 + phi/2 + phiw)
            dlx = lx/(nx - 1)

            if y in [0, ny - 1]:
                wi2 += (nx - 1)*(dlx - phi)**2

            if y < ny-1:
                wi2 += 2*(dly - (phi + rebar_list[y+1][1])/2)**2

            for x in range(nx):
                self.rebar_dict[next(iter0)] = [xc + d3 + phi/2 + phiw + x*dlx,
                                yc + sign*(d3 + phi_first/2 + phiw + dly*y) + y0,
                                phi, As0]
        self._mander_wi2 = wi2
        if ny == 1:
            self._mander_wi2 = 0.0


    def positioning_rectangular_phi(self, rebar_list):
        """
        Positions rebars in rectangular pattern and set rebar_dict attribute.
        More convenient for unusual reinforcement plane or for beam which have
            rebars with different diameter on the same line or not can be solved
            by other positiong methods.
        Args:
            rebar_list(array with shape Nx(+2)) : [[phi1, phi2, phi3 *],
                                                    [phi4, phi5, phi6 phi7, *], *]
            N represent number of layer along y-axis. N represent number of layer
            along y-axis. +2 means the all layer have to include at least two phi
            and is not restricted. phi1, phi2 ... is diameter of individual rebars.
        """

        self.rebar_dict = {}
        rebar_list = rebar_list[::-1]
        b, h, d3, phiw = self._b, self._h, self._d3, self._phiw
        xc, yc = self._origin
        ny = len(rebar_list)
        ly = h - 2*(d3 + phiw)
        dly = ly/(ny - 1)

        wi2 = 0
        iter0 = count(1)
        for y, phis in enumerate(rebar_list):

            double = isinstance(phis[0], (list, tuple, np.ndarray))
            if double:
                phis2 = phis[1]
                phis = phis[0]
                len_double = len(phis2)

            nx = len(phis)
            lx = b - 2*(d3 + phiw) - (phis[0] + phis[-1])/2
            dlx = lx/(nx - 1)

            if y == 0:
                for x, phi in enumerate(phis):
                    As0 = PI*(phi)**2/4
                    xloc = xc + d3 + phis[0]/2 + phiw + x*dlx
                    yloc = yc + d3 + phi/2 + phiw
                    self.rebar_dict[next(iter0)] = [xloc, yloc, phi, As0]

                    if x < nx - 1:
                        wi2 += (dlx - (phi + phis[x+1])/2)**2

                    if double:
                        start_ind = nx - len_double - 1*(nx != len_double)

                        if start_ind <= x < len_double + start_ind:
                            phi2 = phis2[x - start_ind]
                            yloc += (phi + phi2)/2
                            As2 = PI*phi2**2/4
                            self.rebar_dict[next(iter0)] = [xloc, yloc, phi2, As2]


            elif y == ny - 1:
                for x, phi in enumerate(phis):
                    As0 = PI*(phi)**2/4
                    xloc = xc + d3 + phis[0]/2 + phiw + x*dlx
                    yloc = yc + h - (d3 + phi/2 + phiw)
                    self.rebar_dict[next(iter0)] = [xloc, yloc, phi, As0]

                    if x < nx - 1:
                        wi2 += (dlx - (phi + phis[x+1])/2)**2

                    if double:
                        start_ind = nx - len_double - 1*(nx != len_double)
                        if start_ind <= x < len_double + start_ind:
                            phi2 = phis2[x - start_ind]
                            yloc -= (phi + phi2)/2
                            As2 = PI*phi2**2/4
                            self.rebar_dict[next(iter0)] = [xloc, yloc, phi2, As2]


            else:
                for x, phi in enumerate(phis):
                    As0 = PI*(phi)**2/4
                    xloc = xc + d3 + phis[0]/2 + phiw + x*dlx
                    yloc = yc + d3 + phiw + dly*y
                    self.rebar_dict[next(iter0)] = [xloc, yloc, phi, As0]


        phi_sum_y = 0
        for i in range(ny):

            if isinstance(rebar_list[i][0], (list, tuple, np.ndarray)):
                phi_sum_y += rebar_list[i][0][0] + rebar_list[i][0][-1]

            else:
                phi_sum_y += rebar_list[i][0] + rebar_list[i][-1]

        wi2 += 2*(ny - 1)*((ly - phi_sum_y/2)/(ny - 1))**2
        self._mander_wi2 = wi2


    def positioning_circular(self, phi, n):
        """
        Positions number of n rebar along circumference of circular section.
            Args:
                phi (float): diameter of all rebars.
                n (int): number of longitudinal rebars."""

        self.rebar_dict = {}
        D, d3, phiw = self._D, self._d3, self._phiw
        rad = np.linspace(-PI, PI, n, endpoint=False)
        As0 = PI*(phi)**2/4
        r = D/2 - d3 - phiw - phi/2
        xcor = D/2 + r*np.cos(rad)
        ycor = D/2 + r*np.sin(rad)

        for ind, [x, y] in enumerate(zip(xcor, ycor), start=1):
            self.rebar_dict[ind] = [x, y, phi, As0]


    def add_rebar(self, x, y, phi):
        """Adds rebar by coordinates to the section.
        Args:
            x, y (float): coordinates.
            phi (float): diameter of rebar."""

        As0 = PI*(phi)**2/4

        if len(self.rebar_dict)!=0:
            ind = max(self.rebar_dict.keys()) + 1

        else:
            ind = 1

        self.rebar_dict[ind] = [x, y, phi, As0]


    def add_rebar_below(self, id_rebar, phi):
        """
        Adds rebar below rebar_dict[id_rebar] to double it. To learn rebar id
            you can call plot method (to be sure show_id_phi='id' or 'both') or
            print rebar_dict attribute.
        Args:
            id_rebar (int): rebars are stored in rebar_dict(dictionary) attribute
                an all rebar have a id number as a key in the dict.
            phi (float): diameter of rebar."""

        As0 = PI*(phi)**2/4
        ind = max(self.rebar_dict.keys()) + 1
        x = self.rebar_dict[id_rebar][0]
        y = self.rebar_dict[id_rebar][1]-(self.rebar_dict[id_rebar][2] + phi)/2
        self.rebar_dict[ind] = [x, y, phi, As0]


    def add_rebar_above(self, id_rebar, phi):
        """
        Adds rebar above rebar_dict[id_rebar] to double it. To learn rebar id
            you can call plot method (Be sure show_id_phi='id' or 'both') or print
            rebar_dict attribute.
        Args:
            id_rebar (int): rebars are stored in rebar_dict(dictionary) attribute
                an all rebar have a id number as a key in the dict.
            phi (float): diameter of rebar."""

        As0 = PI*(phi)**2/4
        ind = max(self.rebar_dict.keys()) + 1
        x = self.rebar_dict[id_rebar][0]
        y =  self.rebar_dict[id_rebar][1] + (self.rebar_dict[id_rebar][2] + phi)/2

        self.rebar_dict[ind] = [x, y, phi, As0]


    def calc_manderwi2(self, *rebar_ids):
        """
        Calculates mander_wi2 parameter to use as a varible in Material object
            if Mander method will be called. This method must be used in situation of
            add_rebar method is called for adding rebar to the surrounding of section.
        Args:
            *rebar_ids(int): Longitudinal rebar ids surrounding the section.
                Must be given in sequence and closed. After calling plot method
                the rebar ids can be easily seen.
            Example:
            r = sa.Reinforcement('C01', 8, origin=(0, 0), shape='Tshape', b=b, h=h, d3=d3)
            rebars = [[40.0, 40.0, 14], [150, 40, 14], [260.0, 40.0, 14],
                      [39.0, 461.0, 12], [150.0, 461.0, 12], [261.0, 461.0, 12]]
            for x, y, phi in rebars:
                r.add_rebar(x, y, phi)
            r.plot(show_id_phi='id')
            r.calc_manderwi2(1, 2, 3, 6, 5, 4, 1)"""

        rebar = self.rebar_dict
        wi2 = 0

        for i in range(len(rebar_ids)-1):
            i1, i2 = rebar_ids[i], rebar_ids[i+1]
            p12 = (rebar[i1][2] + rebar[i2][2])/2
            x1, y1, x2, y2 = rebar[i1][0], rebar[i1][1], rebar[i2][0], rebar[i2][1]
            length = ((x2 - x1)**2 + (y2 - y1)**2)**0.5 - p12
            wi2 += length**2

        self._mander_wi2 = wi2


    def plot(self, ax=None, show_id_phi='phi', save=False):
        """Plots all rebar.
        Args:
            ax (None or matplotlib axis): for setting matplotlib by yourself
            show_id_phi (str or None): {'id', 'phi', 'both', None} shows
                the given choice(s) on figure.
                None: the following options are not shown.
                'id': shows rebars id's.
                'phi': shows diameter of the rebars.
                'both': shows id and diameters of the rebars.
            save (bool): if True figure will be saved."""

        if not ax:
            fig, ax = plt.subplots()
            ax.axis('equal')
            ax.axis('off')
            show = True
        else:
            show = False

        phiw = self._phiw
        if self._shape in ['rectangle', 'Tshape']:
            b, h, d3, xc, yc = self._b, self._h, self._d3, *self._origin
            section = plt.Polygon([[xc, yc],
                                   [xc + b, yc],
                                   [xc + b, yc + h],
                                   [xc, yc + h]], color='gray', alpha=0.3)

            transverse1 = plt.Polygon([[xc + d3, yc + d3],
                                       [xc + b - d3, yc + d3],
                                       [xc + b - d3, yc + h - d3],
                                       [xc + d3, yc + h - d3]],
                                       color='b', fill=False, lw=0.5)

            transverse2 = plt.Polygon([[xc + d3 + phiw, yc + d3 + phiw],
                                       [xc + b - d3 - phiw, yc + d3 + phiw],
                                       [xc + b - d3 - phiw, yc + h - d3 - phiw],
                                       [xc + d3 + phiw, yc + h - d3 - phiw]],
                                       color='b', fill=False, lw=0.5)

        elif self._shape == 'circle':
            D, d3, xc, yc = self._D, self._d3, *self._origin
            section = plt.Circle((xc + D/2, yc + D/2),
                                  D/2, color='gray', alpha=0.3)
            transverse1 = plt.Circle((xc + D/2, yc + D/2),
                                      D/2 - d3, color='b', fill=False, lw=0.5)
            transverse2 = plt.Circle((xc + D/2, yc + D/2),
                                      D/2 - d3 - phiw, color='b', fill=False, lw=0.5)

        patches = [section, transverse1, transverse2]
        for p in patches:
            ax.add_patch(p)

        for item, [x, y, phi, a] in self.rebar_dict.items():
            reb = plt.Circle((x, y), phi/2, fc='b')
            ax.add_patch(reb)
            text = {'id':str(item), 'phi':str(phi),
                    'both':str(item)+'.'+str(phi), None:''}

            ax.text(x - phi/4, y - phi/4, text[show_id_phi], color='white', weight='bold')
        ax.plot()
        # plt.title('Longitudinal Reinforcement')

        if save:
            plt.savefig(self._fname+'_rebar.png', bbox_inches="tight", pad_inches = 0)

        if show:
            plt.show()



class Material: # ultimate_strain
    """
    Construct material object.
    Args:


    Attribures:


    """

    def __init__(self):
        self.material_model = ''
        self._mat_function = None


    def Hognested(self, fc, Ec):
        """
        Construct Hognested(1951) concrete stress-strain model.
        Args:
            fc (float): unconfined cylindrical strength of concrete specimen.
        """

        self._mat_function_args = [fc, Ec]
        self.material_model = 'Hognested'
        self._mat_function = self.__Hognested
        self.fc = fc
        self.stress = self.__Hognested
   

    def modifiedKentPark(self, fc, confined=False, eps_co=0.002,
                                    eps_cu=0.004, s=None, Ast=None):
        """
        Construct Modified Kent-Park concrete stress-strain model.
        Args:
            fc (float): unconfined cylindrical strength of concrete specimen.
            confined (boolean): if True, the necessary arguments are following
                confined keyword.
            eps_co (float): strain at maximum strength of unconfined concrete.
        """

        if confined:
            # grouped according to having defult value of -1
            self.material_model = 'Confined modified Kent and Park'

        else:
            self.material_model = 'Unconfined modified Kent and Park'

        self._mat_function_args = [fc, confined, eps_co, eps_cu, s, Ast]
        self._mat_function = self.__modifiedKentPark

    def Mander(self, fc, Ec, eps_co=0.002, confined=False,\
                s=None, Ast=None, fyh=-1, transverse_type='circular'):
        """
        Construct Mander J.B., Priestley M.J.N., and Park R. (1988) concrete
            stress-strain model.
        Args:
            fc (float): unconfined cylindrical strength of concrete specimen.
            Ec (float): initial tangent modulus.
            confined (boolean): if True, the necessary arguments are following
                confined keyword.
            eps_co (float): strain at maximum strength of unconfined concrete.
            s (float): hoop spacing.
            Ast (list(float) or float): Area of transverse reinforcement.
                if shape rectangle, Ast = [Asx, Asy]
                if shape circle, Ast = Asp,
            transverse_type (str): {'spiral', 'circular'} for circular section.
        """

        if confined:
            self.material_model = 'Confined Mander'

        else:
            self.material_model = 'Unconfined Mander'

        self._mat_function_args = [fc, Ec, confined, eps_co, s, Ast, fyh, transverse_type]
        self.stress = self.__Mander


    def Steel(self, quality='S420', Es=2e5, fsy=-1, eps_sh=-1, eps_su=-1, fsu=-1):
        """
        Construct steel stress-strain model.
        Args:
            quality (str): {'S420', 'S220'}
                                    fsy,  eps_sh, eps_su,  fyu
                default = {'S220': [220,  0.011,   0.16,   275],
                           'S420': [420,  0.008,   0.08,   550]}
            Es (float): initial tangent modulus.
            fsy (float): yield strength.
            eps_sh (float): strain at the end of yield plateau.
            eps_su (float): ultimate strain.
            fsu (float): ultimate strength.
        """

        self.material_model = 'Steel, ' + quality

        default = {'S220': [220, 0.011, 0.16, 275],
                   'S420': [420, 0.008, 0.08, 550]}

        values = [fsy, eps_sh, eps_su, fsu]

        for i, v in enumerate(values):
            if v == -1:
                values[i] = default[quality][i]


        self.fsy, eps_sh, eps_su, fsu = values
        self.ultimate_strain = eps_su
        eps_sy = self.fsy/Es
        self._mat_function_args = [self.fsy, eps_sy, eps_sh, eps_su, fsu, Es]
        self.stress = self.__Steel

        
        Esh = 2*(fsu - self.fsy)/(eps_su - eps_sh)
        rhoi = (fsu - self.fsy)/self.fsy*1.05
        lyp = (eps_sh - eps_sy)/(eps_sy)

        ops_args = ['{}', self.fsy*1e3, Es*1e3, Esh/Es*0.9, rhoi, 0.0008, 2.773, lyp]
        self._ops_txt = _format['S04'].format(*ops_args)

        #ops_args = ['{}', self.fsy*1e3, Es*1e3, 0.02, 20, 0.925, 0.15]
        #self._ops_txt = _format['S02'].format(*ops_args)


    def to_ops(self, matTag):
        return self._ops_txt.format(matTag)


    def __Hognested(self, eps_c):
        """
        Return stress at the eps_c strain level.
        Args:
            eps_c list(float): strain
        """

        fc, Ec = self._mat_function_args
        eps_co = -2*fc/Ec
        eps_cu = -0.0038
        self.ultimate_strain = eps_cu

        sigma_c  = []
        for eps in eps_c:
            if eps >= 0:
                sigma_c.append(0)

            elif eps_co <= eps < 0:
                sigma_c.append(-fc*(2*eps/eps_co - (eps/eps_co)**2))

            elif eps_cu <= eps<eps_co:
                sigma_c.append(-fc + (eps - eps_co)/(eps_cu - eps_co)*0.15*fc)

            elif eps < eps_cu:
                sigma_c.append(0)

        return np.array(sigma_c) # Ec


    def __modifiedKentPark(self, eps_c): # rho_s' e bak
        """
        Return stress at the eps_c strain level.
        Args:
            eps_c list(float): strain
        """

        fc, confined, eps_co, eps_cu = self._mat_function_args[:4]
        eps_50u = -(3 + 0.285*fc)/(142*fc - 1000)
        eps_co, eps_cu = -eps_co, -eps_cu
        sigma_c = []

        if confined:

            s, Ast = self._mat_function_args[4:]
            fyh, phiw, bc, hc = self._mat_function_args2
            rho_s = Ast*2*(bc+hc)/(s*bc*hc)
            #*************************************
            K = 1 + rho_s*fyh/fc
            eps_coc = K*eps_co
            fcc =  K*fc
            #*************************************
            eps_50h = -0.75*rho_s*(bc/s)**0.5
            Z_bc = 0.5/(eps_50u + eps_50h - eps_coc)
            eps_c20 = 0.8/Z_bc + eps_coc  # NOTE: isaret hatasi olabilir.
            self.ultimate_strain = eps_c20
            
            self._args = [eps_coc, fcc, eps_c20, Z_bc]
            self.stress = self.__modifiedKentPark_confined
            
            ops_args = ['{}', -fcc*1e3, -eps_coc, -0.2*fcc*1e3, -eps_c20]
        
        if not confined:
            Z_bd = 0.5/(eps_50u - eps_co)
            self.ultimate_strain = eps_cu
            
            self._args = [eps_co, fc, eps_cu, Z_bd]
            self.stress = self.__modifiedKentPark_unconfined
            
            ops_args = ['{}', -fc*1e3, -eps_co, 0., -eps_cu]
        
        self._ops_txt = _format['kentpark'].format(*ops_args)
        
        return self.stress(eps_c)


    def __modifiedKentPark_confined(self, eps_c):

        eps_coc, fcc, eps_c20, Z_bc = self._args 
        sigma_c = []
        for eps in eps_c:
            if eps >= 0:
                sigma_c.append(0)

            elif eps_coc <= eps < 0:
                sigma_c.append(-fcc*(2*eps/eps_coc - (eps/eps_coc)**2))

            elif eps_c20 <= eps < eps_coc:
                sigma_c.append(-fcc*(1 - Z_bc*(eps - eps_coc)))

            elif eps < eps_c20:
                sigma_c.append(-0.2*fcc)
   
        return np.array(sigma_c)

    def __modifiedKentPark_unconfined(self, eps_c):

        eps_co, fc, eps_cu, Z_bd = self._args
        sigma_c = []
        for eps in eps_c:
            if eps >= 0:
                sigma_c.append(0)

            elif eps_co <= eps < 0:
                sigma_c.append(-fc*(2*eps/eps_co - (eps/eps_co)**2))

            elif eps_cu <= eps < eps_co:
                sigma_c.append(-fc*(1 - Z_bd*(eps - eps_co)))

            elif eps < eps_cu:
                sigma_c.append(0.)
    
        return np.array(sigma_c)


    def __Mander(self, eps_c):
        """
        Return stress at the eps_c strain level.
        Args:
            eps_c list(float): strain
        """
        fc, Ec, confined, eps_co, s, Ast, fyh_d, transverse_type = self._mat_function_args


        fct = 0.35*(fc)**0.5
        eps_ct = 2*fct/Ec
        
        if confined:
            fyh, eps_su, shape, mander_wi2, As, phiw, core_dimensions = self._mat_function_args2
            if fyh_d != -1:
                fyh = fyh_d

            if shape in ['rectangle', 'Tshape']:
                bc, hc = core_dimensions
                rho_cc = As/(bc*hc)

                alpha_e = (1 - mander_wi2/(6*bc*hc))*(1 - s/(2*bc))*(1 - s/(2*hc))
                rho_st = [Ast[0]/(s*hc), Ast[1]/(s*bc)]
                rho_s = sum(rho_st)/2
                coeff = 0.04
                omega_we = alpha_e*min(rho_st)*fyh/fc

            elif shape == 'circle':
                rho_cc = As/(PI*core_dimensions**2/4)

                if transverse_type=='spiral':
                    power = 1

                elif transverse_type=='circular':
                    power = 2

                alpha_e = (1 - s/(2*core_dimensions))**power
                rho_s = 2*Ast/(core_dimensions*s)
                coeff = 0.07
                omega_we = alpha_e*rho_s*fyh/fc

            # TBDY-2018
            self._ecGO = 0.0035 + coeff*omega_we**0.5

            ke = alpha_e/(1 - rho_cc)
            fe = ke*fyh*rho_s
            lambda_c = 2.254*(1 + 7.94*fe/fc)**0.5 - 2*fe/fc - 1.254
            if lambda_c < 1:
                lambda_c = 1

            fcc = lambda_c*fc
            eps_cc = eps_co*(5*lambda_c - 4)
            eps_cu = 0.004 + 1.4*(2*rho_s)*fyh*eps_su/fcc
            Esec = fcc/eps_cc
            r = Ec/(Ec - Esec)
            self.ultimate_strain = -eps_cu
            self._args = [eps_cc, eps_cu, fcc, r]
            self.stress = self.__mander_confined
            
            xn = eps_cu/eps_cc
            #ops_args = ['{}', -fcc*1e3, -eps_cc, -eps_cu, Ec*1e3] # C04
            ops_args = ['{}', -fcc*1e3, -eps_cc, Ec*1e3, fct*1e3, eps_ct, 2., xn, r] # C07
            self._ops_txt = _format['C07'].format(*ops_args)
        
        elif not confined: # unconfined
            Esec = fc/eps_co
            r = Ec/(Ec - Esec)
            self.ultimate_strain = -0.005
            x35 = 0.0035/eps_co
            f35 = fc*x35*r/(r - 1 + x35**r)
            self._args = [eps_co, fc, f35, r]
            self.stress = self.__mander_unconfined
             
            #ops_args = ['{}', -fc*1e3, -eps_co, -0.0035, Ec*1e3] #C04
            ops_args = ['{}', -fc*1e3, -eps_co, Ec*1e3, fct*1e3, eps_ct, 2., 1.75, 3.8]
            self._ops_txt = _format['C07'].format(*ops_args)

        return self.stress(eps_c)


    def __mander_confined(self, eps_c):
        eps_cc, eps_cu, fcc, r = self._args
        sigma_c = np.zeros(len(eps_c))
        for i, eps in enumerate(eps_c):
            #if -eps_cu <= eps < 0:
            if eps < 0:
                x = -eps/eps_cc
                sigma_c[i] = -fcc*x*r/(r - 1 + x**r)

        return sigma_c

    def __mander_unconfined(self, eps_c):
        eps_co, fc, f35, r = self._args
        sigma_c = np.zeros(len(eps_c))
        for i, eps in enumerate(eps_c):
            if -0.0035 < eps < 0:
                x = -eps/eps_co
                sigma_c[i] = -fc*x*r/(r - 1 + x**r)

            elif -0.005 <= eps <= -0.0035:
                sigma_c[i] = -f35*(0.005 + eps)/0.0015

        return sigma_c


    def __Steel(self, eps_s):
        """
        TBDY 7B.2
        """

        fsy, eps_sy, eps_sh, eps_su, fsu, Es = self._mat_function_args

        sigma_s = []
        sign = np.sign(eps_s)
        for eps in np.abs(eps_s):
            if eps <= eps_sy:
                fs = Es*eps

            elif eps_sy < eps <= eps_sh:
                fs = fsy

            else:
            #elif eps_sh < eps <= eps_su:
                fs = fsu - (fsu - fsy)*(eps_su - eps)**2/(eps_su - eps_sh)**2

            #else:
            #    fs = 0

            sigma_s.append(fs)

        return np.array(sigma_s)*sign



class SectionAnalysis:
    """
    m
    """
    def __init__(self, fname):
        self.fname = fname
        self.parts_mats_dict = {}


    def make_geo(self):
        """assign self.geo attribute to Geometry class"""
        self.geo = Geometry(self.fname)


    def make_rebar(self, phiw, origin=-1):
        """
        assign self.rebar attribute Reinforcement class, do not forget to 
        assign rebar material to the rebar part
        origin: a tuple (x, y), for rebars to be positioned according to 
        reference point, if geo.elements created by Geometry's method, do not change.
        """
        g = self.geo
        if origin==-1:
            if g.shape=='Tshape':
                orig = (g._b1, 0)
            else:
                orig = (0, 0)
        else:
            orig = origin
        self.dimensions_dict = {'b':g._b, 'h':g._h, 'D':g._D, 'd3':g._d3}
        self.rebar = Reinforcement(self.fname, phiw, orig, shape = g.shape, **self.dimensions_dict)


    def make_material(self, *material_attr): # AttributeError
        """assign self.material_attr to Material class"""

        for mat in material_attr:
            if hasattr(self, mat):
                print("'%s' attribute exist, will be replaced"%(mat))
            setattr(self, mat, Material())


    def to_ops(self, fibTag, noRedefineMatTags=[], GJ=None, **matTags):
        """
        m
        """

        ops_txt = f'# {self.fname}\n'
        tag_count = count(1) # with itertools method
        
        for part, material in self.parts_mats_dict.items():
            tag = matTags.get(part)
            if tag == None:
                tag = 'tag%d'%next(tag_count)
                matTags[part] = tag

            if part not in noRedefineMatTags:
                ops_txt += f'# {part.lower()}\n' +  material.to_ops(tag)
        
        ops_txt += self.geo.to_ops(fibTag, GJ=GJ, **matTags)
        
        if 'rebar' in self.parts_mats_dict:
            ops_txt += self.rebar.to_ops(matTags['rebar'])
        else:
            ops_txt += ('}\n'*_editor == 'tcl')

        return ops_txt


    def assign_materials_to_parts(self, *mats_parts):
        """
        assign material to parts
        args = [material_instance_1, self.geo.elements.key_1, <limit_state>], *
        """

        self._mats_parts = mats_parts
        parts = list(self.geo.elements.keys())
        
        r = None
        if hasattr(self, 'rebar'):
            parts.append('rebar')
            r = self.rebar

        material_model_args = {
        'Confined modified Kent and Park': '[fyh, r._phiw, *r._core_dimensions]',
        'Confined Mander': '[fyh, eps_su, r._shape, r._mander_wi2, r.sumA,  r._phiw, r._core_dimensions]'
        }

        # assigning stop limits
        self.strainLimitUser = {}
        for mps in mats_parts:
            mat, part = mps[:2]
            m = 'self.' + mat
            m = eval(m)
            if isinstance(m, Material) and part in parts:
                self.parts_mats_dict[part] = m

            if part == 'rebar':
                fyh, eps_su = m.fsy, m.ultimate_strain

            try:
                limit = mps[2]

            except IndexError:
                limit = None
            self.strainLimitUser[part] = limit


        for part, mat in self.parts_mats_dict.items():
            model = mat.material_model
            if model in material_model_args:
                if r == None:
                    sys.exit('Define Reinforcement object to use confined concrete models')
                mat._mat_function_args2 = eval(material_model_args[model])

            mat.stress([0.0])

            if self.strainLimitUser[part] == -1:
                self.strainLimitUser[part] = mat.ultimate_strain
        
        self._center_dict = self.geo.eCenters
        self._area_dict = self.geo.eAreas

        rebar_lst = self.rebar.rebar_dict.values()
        self._center_dict['rebar'] = np.array([v[:2] for v in rebar_lst]).T
        self._area_dict['rebar'] = np.array([v[3] for v in rebar_lst])


    def _get_critic_coor(self, part, strain_sign, direction):
        if strain_sign == None:
            return

        sign = np.sign(direction)
        dir_ind = abs(direction) - 2

        if sign*strain_sign < 0:
            return self._center_dict[part][dir_ind].max()
            
        elif sign*strain_sign > 0:
            return self._center_dict[part][dir_ind].min()


    def get_strain(self, coor=None, part=None, strain_sign=None):
        sign = np.sign(self.direction)
        dir_ind = abs(self.direction) - 2
        section_center = self.geo.section_center[dir_ind]

        if coor == None and (part == None or strain_sign == None):
            sys.exit('Define coor or part and strain_sign')

        if coor == None:
            coor = self._get_critic_coor(part, strain_sign, self.direction)

        else:
            coor = coor[dir_ind]

        return self.center_strain - sign*self.curvature*(coor - section_center)
            

    def _MomentCurvature(self, compression_strain, axial_load, direction=3):
        """

        """

        self.direction = direction
        sign = np.sign(direction)
        dir_ind = abs(direction) - 2

        h1 = self.geo.height[dir_ind]
        section_center = self.geo.section_center[dir_ind]


        if sign == 1:
            def strain_func(depth):
                return eps*(coor - h1 + depth)/depth
        else:
            def strain_func(depth):
                return eps*(-coor + depth)/depth
        
        count_mat_func = 0 # number of calling material's stress function
        compression_strain2 = [] # created for realistic c values

        # phi, moment calculation
        Moment = [0]
        Phi = [0]
        brk = 0
        tolerance = 100 # for F
        parts_mats_items = self.parts_mats_dict.items()
        
        # finding root of the equations
        #i = 0
        #while brk == 0 and i < len(compression_strain):
        #    eps = compression_strain[i]
        #    i += 1
        center_strain = [0]
        for eps in compression_strain:
            c0 = h1*0.1
            c1 = h1*0.5
            n = 1
            Fel = []
            for part, material in parts_mats_items:
                coor = self._center_dict[part][dir_ind]
                strain = strain_func(c0)
                sigma_var = material.stress(strain) # MPa
                Fel.append(sigma_var@self._area_dict[part]) # MPa*mm**2 = N
                count_mat_func+=1
            f0 = sum(Fel)

            condition = True
            while condition:
                Fel = []
                for part, material in parts_mats_items:
                    coor = self._center_dict[part][dir_ind]
                    strain = strain_func(c1)
                    sigma_var = material.stress(strain) # MPa
                    Fel.append(sigma_var@self._area_dict[part]) # MPa*mm**2 = N
                    count_mat_func+=1
                f1 = sum(Fel)
                if f1 != f0:
                    c2 = c1 + (axial_load - f1)*(c1 - c0)/(f1 - f0)
                    c0, c1 = c1, c2
                    condition = abs(axial_load - f1) > tolerance and n < 100
                    f0 = f1
                    n += 1
                else:
                    c2 = (c0 + c1)/2
                    condition = False

            brk = 0
            phi = abs(eps/c2)
            if 0 < c2 < h1 and n < 100 and phi > Phi[-1]:
                moment = 0
                force = 0
                for part, material in parts_mats_items:
                    coor = self._center_dict[part][dir_ind]
                    strain = strain_func(c2)
                    limit_strain = self.strainLimitUser[part]
                    if limit_strain:
                        if limit_strain > 0 and np.any(strain > limit_strain):
                            brk = 1
                            print('m end', part, limit_strain,  max(strain))
                            print('m end', part, min(strain))

                        elif limit_strain < 0 and np.any(strain < limit_strain):
                            brk = 1
                            print('m end', part, min(strain))
                    sigma_var = material.stress(strain) # MPa
                    moment += (sigma_var*self._area_dict[part])@(section_center - coor) # Nmm

                    force += sigma_var@self._area_dict[part]
                    count_mat_func+=1

                if brk == 0:
                    Phi.append(phi) # 1/m
                    Moment.append(moment)
                    compression_strain2.append(eps)
                    
                    coor = section_center # for strain at center
                    center_strain.append(strain_func(c2))
                else:
                    break
        
        
        self.curvature, self.moment = np.array([Phi, Moment])
        self.center_strain = np.array(center_strain)
        
        return self.curvature, self.moment


    def MomentCurvature(self, phis, axial_load, direction=3):
        """

        """

        self.direction = direction
        sign = np.sign(direction)
        dir_ind = abs(direction) - 2

        h1 = self.geo.height[dir_ind]
        section_center = self.geo.section_center[dir_ind]

        tolerance = 100 # for F
        parts_mats_items = self.parts_mats_dict.items()

        
        def feps0(deformation):
            Fel = []
            for part, material in parts_mats_items:
                sigma_var = material.stress([deformation])[0]
                Fel.append(sigma_var*np.sum(self._area_dict[part]))
            return sum(Fel) 


        def sumForce(depth):
            Fel = []
            for part, material in parts_mats_items:
                coor = self._center_dict[part][dir_ind]
                strain =  -phi*(coor - depth)
                sigma_var = material.stress(strain) # MPa
                Fel.append(sigma_var@self._area_dict[part]) # MPa*mm**2 = N
            return  sum(Fel)


        phis = sign*phis
        Moment = []
        Phi = []
        center_strain = []
        
        brk = 0
        eps_height = [None, h1, 0.][sign]
        c2 = 0.2*h1
        c0 = 0.05*h1
        c1 = 0.95*h1

        for phi in phis:

            c0 = 0.05*h1
            c1 = 0.95*h1
            c2 = self.Secant(c0, c1, axial_load, sumForce, 100, 100)
            if c2 != None and (0 < c2 < h1):
                moment = 0
                for part, material in parts_mats_items:
                    coor = self._center_dict[part][dir_ind]
                    strain = -phi*(coor - c2)
                    
                    limit_strain = self.strainLimitUser[part]
                    if limit_strain:
                        if limit_strain > 0 and np.any(strain > limit_strain):
                            brk = 1
                            print(part, limit_strain, max(strain))

                        elif limit_strain < 0 and np.any(strain < limit_strain):
                            brk = 1
                            print(part, limit_strain, min(strain))
                    
                    sigma_var = material.stress(strain) # MPa
                    moment += (sigma_var*self._area_dict[part])@(section_center - coor) # Nmm
                

                Phi.append(phi) # 1/mm
                Moment.append(moment)
                center_strain.append(-phi*(section_center -  c2))

                c0 = 0.9*c2
                c1 = 1.1*c2

            if brk == 1:
                break


        self.curvature, self.moment = np.array([Phi, Moment]) 
        self.center_strain = np.array(center_strain)

        return np.array([Phi, Moment])


    def Secant(self, x0, x1, y, func, error, Nmax):
        """ m """
        
        step = 1
        condition=True
        f0 = func(x0)
        
        while condition: 
            f1 = func(x1)
            if f1 != f0:
                x2 = x1 + (y - f1)*(x1 - x0)/(f1 - f0)
                x0, x1 = x1, x2
                f0 = f1
                step += 1
                condition = abs(y - f1) > error and step < Nmax
            else:
                x2 = (x0 + x1)/2
                condition = False 
        #print('Required root is: %3.8f %.1f step: %s'%(x2, f1, step))
        if step < Nmax - 1:
            return x2


    def __PM(self, eps_co, eps_sy, direction=3, total_step=50):
        """

        """
        parts_mats_items = self.parts_mats_dict.items()

        sign = np.sign(direction)
        lens = int((total_step - 4)/7)
        dir_ind = abs(direction) - 2

        
        sorted_ind = sign - (sign==1)
        sorted_rebar = sorted(rebar_lst, key=lambda x: x[dir_ind])

        h1 = self.geo.height[dir_ind]
        h2 = sort_coor[sorted_ind]

        H = h2 - h1*(sign==1)

        eps_c_list = np.array([
            *[eps_co for i in range(4*lens + 2)],
            *np.linspace(eps_co, eps_co/2, 2*lens, endpoint=False),
            *np.linspace(eps_co/2, 0, lens, endpoint=False),
            eps_sy/2, eps_sy
            ])

        eps_s_list = np.array([
            eps_co, eps_co/2,
            *np.linspace(eps_co/3, 0, lens, endpoint=False),
            *np.linspace(0, eps_sy, 3*lens, endpoint=False),
            *[eps_sy for i in range(3*lens + 2)]
            ])

        Moment = []
        Force = []

        n = 0
        for eps_c, eps_s in zip(eps_c_list, eps_s_list):
            # plt.plot([eps_s, eps_c], [0, h1], 'r--.')
            # plt.plot([0, 0], [0, h1])
            slope = (eps_s - eps_c)/H

            moment = 0
            force = 0
            for part, material in parts_mats_items:
                coor = self._center_dict[part][:, dir_ind]
                strain = slope*(coor - h2) + eps_s
                sigma_var = material.stress(strain) # MPa
                force_array = sigma_var*self._area_dict[part]

                force += sum(force_array)
                moment += force_array@(h1/2.0 - coor) # Nmm

            Moment.append(moment/10**6)
            Force.append(force/10**3)
            # print(n, eps_c, eps_s)
            n += 1
        # plt.show()
        return np.array([Moment, Force])


