
import matplotlib.pyplot as plt
import openseespy.opensees as ops
import numpy as np
import rayir


def MC(b, h, d3, phiw, s, narm_xy, rebar_list, fc):

    #Ec = 3250*fc**0.5 + 14000 #  TS500-Denk.3.2
    Ec = 5000*fc**0.5 # TBDY
    A0 = (np.pi*phiw**2)/4
    Ast = [i*A0 for i in narm_xy]
    mesh_size = d3


    model = rayir.SectionAnalysis('aaa')

    model.make_geo()
    model.geo.set_mesh_info('tri', d3, mesher='builtin')
    model.geo.Rectangle(b, h, d3)
    model.geo.generate_mesh()

    model.make_rebar(phiw)
    model.rebar.positioning_rectangular_beam(rebar_list)
    #model.rebar.positioning_rectangular_column(16, 3, 3)


    model.make_material('mat_unconfined', 'mat_confined', 'mat_reinforcement' )
    model.mat_unconfined.Mander(fc, Ec, confined=False)
    model.mat_confined.Mander(fc, Ec, confined=True, s=s, Ast=Ast)
    model.mat_reinforcement.Steel(quality='S420')

    es_cr, ec_cr = -1, -1
    model.assign_materials_to_parts(['mat_reinforcement', 'rebar', es_cr],
                                    ['mat_unconfined', 'unconfined'],
                                    ['mat_confined', 'confined', ec_cr])

    return model

if __name__ == '__main__':
    curv_impact = np.linspace(0, 0.2, 150) 

    b, h, d3 = 400, 600, 30
    phiw, s = 8, 100
    arms = [2, 2]

    Ast = [i*np.pi*phiw**2/4 for i in arms]
    rebar = [[3, 14], [3, 16]]
    fc = 30
    P = b*h*fc*0.1
    model = MC(b, h, d3, phiw, s, arms, rebar, fc)



    curvature, moment = model.MomentCurvature(curv_impact*1e-3, -P, direction=3)
    xs, ys = rayir.bilinear_response(curvature, moment, method='UTC-40')


    plt.plot(curvature*1e3, moment*1e-6, 'b-', label='rayir')
    plt.legend()
    plt.show()
